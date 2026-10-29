"""Stage 07, manual editing: the tree labels corrected by hand in segfix (https://pypi.org/project/segfix/).

usage: s07_segfix.py prepare [PLOT ...] [--buffer=15] [--force]
       s07_segfix.py status  [PLOT ...]
       s07_segfix.py import  [PLOT ...] [--allow-unfinished]

prepare  writes a segfix project from the transferred labels (07_trees/<PLOT>_07_merged_2cm_treeid_tiles):
           07_trees/<PLOT>_07_segfix/<PLOT>_07_segfix.las   every labelled point, and the unlabelled points inside the plot box
                                                            plus --buffer m; all 2 cm fields, with a signed treeID
                                                            (> 0 tree, 0 unassigned, -1 noise) that segfix finds on its own
           07_trees/<PLOT>_07_segfix/segfix_project.json    the segfix project manifest
           07_trees/<PLOT>_07_segfix/pipeline_segfix.json       which merge tiles the points came from, in order
         and adds the project to segfix's recent list. Open it with `segfix` and work through the trees, starting with
         those at the top of 07_trees/<PLOT>_07_review.csv; Space marks a tree done, Ctrl+S saves (segfix writes only the
         labels, in place). The point order never changes, which is what lets the import split the file back into tiles.
status   trees reviewed (segfix's Done list in <cloud>.segfix.json) and labels changed so far.
import   reads the saved project back: 07_trees/<PLOT>_07_trees_edited_tiles/*.laz (same tiles as the merge, treeid
         int32) and <PLOT>_07_segfix_import.json (trees added, removed and changed, points relabelled, noise). From then
         on the review and stages 08-09 read the edited labels (project.tree_tiles). Refused while trees are not marked
         done, unless --allow-unfinished.
Plots without manual trees have no transferred labels to start from; segment them first (e.g. `sylva trees --segment`
on the height-normalised plot cloud) and import that file into segfix by hand.
"""
import copy, json, os, sys, time
from pathlib import Path
import numpy as np
import project
from project import ROOT, log

LABEL = "treeID"


def project_dir(site): return ROOT / site / "07_trees" / f"{site}_07_segfix"
def cloud_path(site): return project_dir(site) / f"{site}_07_segfix.las"


def transfer_tiles(site):
    import glob
    return sorted(glob.glob(str(ROOT / site / "07_trees" / f"{site}_07_merged_2cm_treeid_tiles" / "*.laz")))


def header_for(first, bounds):
    """A LAS 1.4 header with the tiles' point format and extra fields, the transfer's treeid replaced by a signed treeID."""
    import laspy
    h = laspy.LasHeader(point_format=first.point_format.id, version="1.4")
    for d in first.point_format.extra_dimensions:
        if d.name.lower() != "treeid": h.add_extra_dim(laspy.ExtraBytesParams(name=d.name, type=d.dtype))
    h.add_extra_dim(laspy.ExtraBytesParams(name=LABEL, type=np.int32, description="tree id; 0 unassigned, -1 noise"))
    h.scales = np.array([0.001, 0.001, 0.001]); h.offsets = np.floor(np.asarray(bounds[:3], float)); return h


def prepare(site, buffer=15.0, force=False):
    import laspy
    tiles = transfer_tiles(site)
    if not tiles: raise FileNotFoundError(f"{site}: no transferred labels; run s07_trees.py first")
    pd = project_dir(site)
    if pd.exists() and not force: raise FileExistsError(f"{pd} exists (it may hold edits); --force replaces it")
    pd.mkdir(parents=True, exist_ok=True)
    for f in pd.iterdir():
        if f.is_file(): f.unlink()
    x0, y0, x1, y1 = project.plot_bounds(site); x0, y0, x1, y1 = x0 - buffer, y0 - buffer, x1 + buffer, y1 + buffer
    t0 = time.time(); src = [laspy.open(t).header for t in tiles]
    lo = np.min([h.mins for h in src], axis=0); hdr = header_for(src[0], lo)
    std = [d for d in src[0].point_format.standard_dimension_names if d not in ("X", "Y", "Z")]
    extra = [d.name for d in src[0].point_format.extra_dimensions if d.name.lower() != "treeid"]
    parts, n_all, n_out, n_trees = [], 0, 0, 0
    with laspy.open(str(cloud_path(site)), mode="w", header=hdr) as w:
        for t in tiles:
            las = laspy.read(t); lab = np.asarray(las["treeid"]).astype(np.int64); x, y = np.asarray(las.x), np.asarray(las.y)
            keep = (lab > 0) | ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)); n = int(keep.sum()); n_all += len(lab)
            parts.append({"tile": Path(t).name, "points": n})
            if not n: continue
            rec = laspy.ScaleAwarePointRecord.zeros(n, header=hdr)
            rec.x, rec.y, rec.z = x[keep], y[keep], np.asarray(las.z)[keep]
            for d in std + extra: rec[d] = np.asarray(las[d])[keep]
            rec[LABEL] = lab[keep].astype(np.int32); w.write_points(rec); n_out += n; n_trees += int((lab[keep] > 0).sum())
    project.jsave({"source": str(ROOT / site / "07_trees" / f"{site}_07_merged_2cm_treeid_tiles"), "data_file": cloud_path(site).name,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, pd / "segfix_project.json")
    project.jsave({"site": site, "label_field": LABEL, "plot_box_buffer_m": buffer, "box": [x0, y0, x1, y1], "tiles": parts,
                "points": n_out, "labelled_points": n_trees, "merged_points": n_all, "prepared": time.strftime("%Y-%m-%d %H:%M")}, pd / "pipeline_segfix.json")
    register(pd)
    log(f"{site}: segfix project {pd.name}: {n_out:,} of {n_all:,} points ({n_trees:,} labelled), {cloud_path(site).stat().st_size / 1e9:.1f} GB ({time.time() - t0:.0f} s)")


def register(pd):
    """Put the project at the top of segfix's recent list (~/.config/segfix/registry.json), as segfix's own startup dialog does."""
    reg = Path.home() / ".config" / "segfix" / "registry.json"
    try:
        r = json.load(open(reg)) if reg.exists() else {"entries": []}
        r["entries"] = [e for e in r["entries"] if e.get("path") != str(pd)]
        r["entries"].insert(0, {"path": str(pd), "type": "workspace", "last_opened": time.strftime("%Y-%m-%dT%H:%M:%S")})
        reg.parent.mkdir(parents=True, exist_ok=True); project.jsave(r, reg)
    except (OSError, ValueError) as e: log(f"could not add {pd} to segfix's recent list ({e}); open it with File > Open Project")


def read_labels(site):
    """treeID of the saved project (segfix edits it in place), and the transfer's labels for the same points."""
    import laspy
    with laspy.open(str(cloud_path(site))) as r: new = np.concatenate([np.asarray(c[LABEL]) for c in r.chunk_iterator(20_000_000)]).astype(np.int64)
    meta = json.load(open(project_dir(site) / "pipeline_segfix.json")); x0, y0, x1, y1 = meta["box"]; old = []
    for p in meta["tiles"]:
        if not p["points"]: continue
        las = laspy.read(str(ROOT / site / "07_trees" / f"{site}_07_merged_2cm_treeid_tiles" / p["tile"]))
        lab = np.asarray(las["treeid"]).astype(np.int64); x, y = np.asarray(las.x), np.asarray(las.y)
        old.append(lab[(lab > 0) | ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))])
    old = np.concatenate(old)
    if len(old) != len(new): raise RuntimeError(f"{site}: the segfix cloud has {len(new):,} points, the tiles {len(old):,}; was it prepared from other tiles?")
    return old, new, meta


def done_trees(site):
    p = Path(str(cloud_path(site)) + ".segfix.json")
    return {int(t) for t in json.load(open(p)).get("done", [])} if p.exists() else set()


def compare(old, new, done):
    ids_old = set(np.unique(old[old > 0]).tolist()); ids_new = set(np.unique(new[new > 0]).tolist())
    ch = old != new; touched = set(np.unique(old[ch & (old > 0)]).tolist()) | set(np.unique(new[ch & (new > 0)]).tolist())
    return {"trees_before": len(ids_old), "trees_after": len(ids_new), "trees_added": sorted(ids_new - ids_old), "trees_removed": sorted(ids_old - ids_new),
            "trees_changed": len(touched & ids_new & ids_old), "points_relabelled": int(ch.sum()), "noise_points": int((new == -1).sum()),
            "trees_done": len(done & ids_new), "trees_not_done": sorted(ids_new - done)}


def status(site):
    if not cloud_path(site).exists(): log(f"{site}: no segfix project (s07_segfix.py prepare {site})"); return
    old, new, _ = read_labels(site); s = compare(old, new, done_trees(site))
    log(f"{site}: {s['trees_done']} of {s['trees_after']} trees done; {s['points_relabelled']:,} points relabelled, {s['trees_changed']} trees changed, "
        f"{len(s['trees_added'])} added, {len(s['trees_removed'])} removed, {s['noise_points']:,} noise points")
    return s


def import_(site, allow_unfinished=False):
    import laspy
    old, new, meta = read_labels(site); s = compare(old, new, done_trees(site))
    if s["trees_not_done"] and not allow_unfinished:
        raise RuntimeError(f"{site}: {len(s['trees_not_done'])} trees not marked done in segfix (first: {s['trees_not_done'][:10]}); finish them or pass --allow-unfinished")
    out = project.edited_tiles_dir(site); tmp = out.with_name(out.name + ".part")
    if tmp.exists():
        for f in tmp.iterdir(): f.unlink()
    tmp.mkdir(exist_ok=True)
    with laspy.open(str(cloud_path(site))) as r:
        hdr = r.header
        for p in meta["tiles"]:
            if not p["points"]: continue
            las = laspy.LasData(header=copy.deepcopy(hdr), points=r.read_points(p["points"]))
            lab = np.asarray(las[LABEL]).astype(np.int32)
            las.add_extra_dim(laspy.ExtraBytesParams(name="treeid", type=np.int32, description="tree id; 0 unassigned, -1 noise")); las["treeid"] = lab
            las.remove_extra_dim(LABEL); las.write(str(tmp / p["tile"]))
    if out.exists():
        for f in out.iterdir(): f.unlink()
        out.rmdir()
    os.replace(tmp, out)
    rec = {"site": site, "project": str(project_dir(site)), "imported": time.strftime("%Y-%m-%d %H:%M"), "tiles": str(out.name), **s}
    project.jsave(rec, ROOT / site / "07_trees" / f"{site}_07_segfix_import.json")
    project.record_run(site, "07_segfix_import", trees=s["trees_after"], relabelled=s["points_relabelled"], not_done=len(s["trees_not_done"]))
    log(f"{site}: imported {s['trees_after']} trees ({s['trees_done']} done) into {out.name}; {s['points_relabelled']:,} points relabelled")


def done(site):
    """True when the edited labels are imported and newer than the segfix cloud (no unsaved-into-pipeline edits)."""
    j = ROOT / site / "07_trees" / f"{site}_07_segfix_import.json"
    return j.exists() and (not cloud_path(site).exists() or j.stat().st_mtime >= cloud_path(site).stat().st_mtime)


if __name__ == "__main__":
    p = project.parser(__doc__, sites=False); p.add_argument("action", choices=["prepare", "status", "import"])
    p.add_argument("sites", nargs="*", metavar="PLOT"); p.add_argument("--buffer", type=float, default=15.0)
    p.add_argument("--force", action="store_true"); p.add_argument("--allow-unfinished", action="store_true"); a = p.parse_args()
    sites = a.sites or [s for s in project.plots() if project.has_trees(s)]
    if a.action == "prepare": ok = project.for_sites(prepare, sites, "07_segfix", buffer=a.buffer, force=a.force)
    elif a.action == "status": ok = project.for_sites(status, sites, "07_segfix")
    else: ok = project.for_sites(import_, sites, "07_segfix", allow_unfinished=a.allow_unfinished)
    sys.exit(0 if ok else 1)
