"""Stages 08 (QSM) and 09 (leaf/wood) for the manual trees: a raycloudtools QSM and a sylva QSM per tree, and a sylva
leaf/wood split with leaf angles and leaf area.

usage: s08_tree_models.py [SITE ...] [--workers=6] [--only=12,34] [--no-rct] [--no-sylva] [--limit=N] [--meshes-only]

Trees come from tern.tree_tiles (the segfix-edited labels once imported, else the transfer); each point's ray starts at
the scanner it came from (point_source_id -> scan position in the merge's matrices).
  08_qsm/<SITE>_08_rct/<SITE>_08_tree_<id>_trees.txt      raycloudtools `rayextract trees` on the tree's ray cloud with
                                                           the plot ground mesh (from the 1 m DTM); + _trees_mesh.ply
  08_qsm/<SITE>_08_sylva/<SITE>_08_tree_<id>_qsm.csv      sylva.qsm.build_qsm on the tree's wood points (1 cm thinned),
                                                           breast-height radius prior from 06_stems or the tree's own slab
  09_leafwood/<SITE>_09_tree_<id>.laz                      the tree's points with wood (1/0), height above the DTM, treeid
  <SITE>_08_rct_trees.csv, <SITE>_08_sylva_trees.csv, 09_leafwood/<SITE>_09_leafwood.csv, 08_qsm/<SITE>_08_models_manifest.json
  08_qsm/<SITE>_08_plot_rct_mesh.ply, _plot_sylva_mesh.ply, _plot_ground_mesh.ply   all trees of each tool, a tree the same colour in both
Resumable: a tree is redone only when its points changed (count and coordinate checksum) or a step is missing; trees
removed in segfix lose their outputs. Leaf/wood: sylva.leaves.classify_leaf_wood (graph-based, Tian & Li 2022).
"""
import colorsys, csv, json, re, subprocess, time, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import tern
from tern import ROOT, log

PLY_DT = np.dtype([("x", "f8"), ("y", "f8"), ("z", "f8"), ("t", "f8"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")])
PLY_HDR = ("ply\nformat binary_little_endian 1.0\ncomment TERN TLS manual tree, ray starts at the scanner of each point\nelement vertex {n}\n"
           "property double x\nproperty double y\nproperty double z\nproperty double time\nproperty float nx\nproperty float ny\nproperty float nz\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\nend_header\n")


def write_ground_mesh(dtm, path):
    """The DTM as a triangle mesh in raycloudtools' layout (double xyz + rgba vertices, int face lists)."""
    z = dtm.data; rows, cols = z.shape; res = dtm.resolution
    X, Y = np.meshgrid(dtm.xmin + (np.arange(cols) + 0.5) * res, dtm.ymin + (np.arange(rows) + 0.5) * res)
    ok = np.isfinite(z); vid = np.full(z.shape, -1, np.int64); vid[ok] = np.arange(ok.sum())
    V = np.zeros(ok.sum(), dtype=np.dtype([("x", "f8"), ("y", "f8"), ("z", "f8"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")]))
    V["x"], V["y"], V["z"] = X[ok], Y[ok], z[ok]; V["r"], V["g"], V["b"], V["a"] = 120, 100, 80, 255
    a, b, c, d = vid[:-1, :-1], vid[:-1, 1:], vid[1:, 1:], vid[1:, :-1]; q = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    F = np.concatenate([np.stack([a[q], b[q], c[q]], 1), np.stack([a[q], c[q], d[q]], 1)])
    faces = np.zeros(len(F), dtype=np.dtype([("n", "i4"), ("i", "i4", 3)])); faces["n"] = 3; faces["i"] = F
    hdr = (f"ply\nformat binary_little_endian 1.0\ncomment ground mesh from the 1 m DTM\nelement vertex {len(V)}\nproperty double x\nproperty double y\nproperty double z\n"
           f"property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\nelement face {len(faces)}\nproperty list int int vertex_indices\nend_header\n")
    with open(path, "wb") as f: f.write(hdr.encode()); f.write(V.tobytes()); f.write(faces.tobytes())
    return len(V), len(faces)


def collect_trees(site, only=None):
    """{tree id: (xyz float32, scan index)} of every labelled point."""
    from sylva import io
    xs, ids, ps = [], [], []
    for t in tern.tree_tiles(site):
        c = io.read(t); lab = c.attrs["treeid"].astype(np.int64); m = lab > 0
        if only: m &= np.isin(lab, list(only))
        if m.any(): xs.append(c.xyz[m].astype(np.float32)); ids.append(lab[m]); ps.append(c.attrs["point_source_id"][m].astype(np.int32))
    xyz = np.concatenate(xs); tid = np.concatenate(ids); psid = np.concatenate(ps); order = np.argsort(tid, kind="stable")
    xyz, tid, psid = xyz[order], tid[order], psid[order]
    starts = np.r_[0, np.flatnonzero(tid[1:] != tid[:-1]) + 1]; ends = np.r_[starts[1:], len(tid)]
    return {int(tid[s]): (xyz[s:e], psid[s:e]) for s, e in zip(starts, ends)}


def signature(xyz):
    """Point count and a coordinate checksum (mm), order-independent: a tree is redone when this changes."""
    return f"{len(xyz)}:{int(np.round(xyz.astype(np.float64) * 1000).sum()) & 0xFFFFFFFFFFFF:x}"


def parse_rct_trees(path):
    """Height, base diameter, volume, length and segments from a raycloudtools tree file. A manual tree that rct splits into
    several (a multi-stem clump) is summed; height is the tallest, base diameter the largest; rct_trees_found says how many."""
    lines = [l for l in open(path) if l.strip() and not l.startswith("#")]
    if len(lines) < 2: return None
    cols = lines[0].strip().split(","); n = len(cols); ix = {c: k for k, c in enumerate(cols)}; out = []
    for line in lines[1:]:
        seg = np.array([float(v) for v in line.replace(",", " ").split()]).reshape(-1, n)
        xyz = seg[:, [ix["x"], ix["y"], ix["z"]]]; rad = seg[:, ix["radius"]]; par = seg[:, ix["parent_id"]].astype(int)
        has = par >= 0; L = np.linalg.norm(xyz[has] - xyz[par[has]], axis=1); rm = 0.5 * (rad[has] + rad[par[has]])
        out.append((len(seg), xyz[:, 2].max() - xyz[:, 2].min(), 2 * rad[0], np.sum(np.pi * rm ** 2 * L), L.sum()))
    o = np.array(out)
    return {"rct_trees_found": len(out), "rct_segments": int(o[:, 0].sum()), "rct_height_m": float(o[:, 1].max()), "rct_base_diameter_m": float(o[:, 2].max()),
            "rct_volume_m3": float(o[:, 3].sum()), "rct_total_length_m": float(o[:, 4].sum())}


def stem_prior(xyz, h, wood, stems):
    """Breast-height radius prior: the detected stem (06_stems) within 0.6 m of the tree base, capped at 3 % of the tree's height,
    else the spread of its wood points in the 1.0-1.6 m slab (or its lowest fifth), capped at 2 %."""
    low = xyz[np.argsort(xyz[:, 2])[:max(20, len(xyz) // 100)]]; base = low[:, :2].mean(axis=0); H = float(h.max())
    if stems is not None and len(stems):
        d = np.linalg.norm(stems[:, :2] - base, axis=1); k = int(np.argmin(d))
        if d[k] < 0.6 and stems[k, 2] > 0: return float(min(stems[k, 2] / 2, 0.03 * H)), "06_stems"
    sel = wood & (h > 1.0) & (h < 1.6)
    if sel.sum() < 30: sel = wood & (h < max(0.4, 0.2 * H))
    if sel.sum() < 30: sel = h < max(0.4, 0.2 * H)
    q = xyz[sel, :2]; ev = np.sqrt(np.linalg.eigvalsh(np.cov((q - q.mean(axis=0)).T)))[::-1] if len(q) > 3 else np.array([0.05, 0.05])
    return float(np.clip(min(1.5 * ev.mean(), 0.02 * H), 0.01, 1.0)), "slab spread"


def one_tree(args):
    site, tid, xyz, psid, origins, dtm_d, do_rct, do_sylva, stems = args
    from sylva import PointCloud, Raster, filters, io, leaves, qsm
    t0 = time.time(); res = {"treeid": tid, "n_points": int(len(xyz)), "signature": signature(xyz)}
    rct_dir = ROOT / site / "08_qsm" / f"{site}_08_rct"; syl_dir = ROOT / site / "08_qsm" / f"{site}_08_sylva"; lw_dir = ROOT / site / "09_leafwood"
    try:
        dtm = Raster(dtm_d["data"], dtm_d["xmin"], dtm_d["ymin"], dtm_d["resolution"]); gz = dtm.sample(xyz[:, 0].astype(np.float64), xyz[:, 1].astype(np.float64))
        gz = np.where(np.isfinite(gz), gz, np.nanmedian(gz) if np.isfinite(gz).any() else xyz[:, 2].min()); h = (xyz[:, 2] - gz).astype(np.float32)
        res["height_m"] = H = float(h.max())
        if do_rct:
            ply = rct_dir / f"{site}_08_tree_{tid}.ply"; txt = rct_dir / f"{site}_08_tree_{tid}_trees.txt"; txt.unlink(missing_ok=True)
            a = np.zeros(len(xyz), PLY_DT); a["x"], a["y"], a["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]; a["t"] = np.arange(len(xyz), dtype=np.float64)
            a["nx"], a["ny"], a["nz"] = (origins[psid] - xyz).T.astype(np.float32); a["r"], a["g"], a["b"], a["a"] = 200, 120, 40, 255
            with open(ply, "wb") as f: f.write(PLY_HDR.format(n=len(a)).encode()); f.write(a.tobytes())
            # plain mode: --segmented inflates the trunk of multi-stem clumps; no --max_diameter (below rct's 0.9 it runs 50-300x slower)
            r = subprocess.run(["apptainer", "exec", tern.RCT_SIF, "rayextract", "trees", ply.name, "ground_mesh.ply"], cwd=rct_dir, capture_output=True, text=True, timeout=3600)
            if not txt.exists(): raise RuntimeError("rayextract trees wrote no tree file: " + (r.stdout + r.stderr).strip().splitlines()[-1][:200])
            ply.unlink(); (rct_dir / f"{site}_08_tree_{tid}_segmented.ply").unlink(missing_ok=True)
            s = parse_rct_trees(txt); res.update(s or {"rct_segments": 0}); res["rct_mode"] = "plain"; res["rct_seconds"] = round(time.time() - t0, 1)
            if s and s["rct_base_diameter_m"] > 0.06 * H: res["rct_flag"] = "trunk diameter over 6 % of height: multi-stem clump, model unreliable"
        if do_sylva:
            t1 = time.time(); cloud = PointCloud(xyz.astype(np.float64)); wood = leaves.classify_leaf_wood(cloud, voxel_size=0.02, method="gbs")
            io.write(cloud.with_attrs(wood=wood.astype(np.uint8), height=h, treeid=np.full(len(cloud), tid, np.uint32), point_source_id=psid.astype(np.uint16)),
                     str(lw_dir / f"{site}_09_tree_{tid}.laz"))
            res.update(n_wood=int(wood.sum()), n_leaf=int((~wood).sum()), wood_fraction=float(wood.mean()))
            if (~wood).sum() >= 500:
                lad = leaves.leaf_angle_distribution(cloud.xyz[~wood])
                res.update(leaf_angle_mean_deg=float(np.degrees(lad.mean)), leaf_angle_std_deg=float(np.degrees(lad.std)), leaf_beta_a=float(lad.beta_a),
                           leaf_beta_b=float(lad.beta_b), leaf_chi=float(lad.chi), leaf_area_m2=float(leaves.leaf_area_density(cloud.xyz[~wood], voxel_size=0.25).total_area))
            res["leafwood_seconds"] = round(time.time() - t1, 1); t2 = time.time()
            qsm_csv = syl_dir / f"{site}_08_tree_{tid}_qsm.csv"; qsm_csv.unlink(missing_ok=True)
            if wood.sum() >= 200:
                wc = filters.voxel_downsample(PointCloud(cloud.xyz[wood]), 0.01)
                base = wc.xyz[np.argsort(wc.xyz[:, 2])[:max(20, len(wc) // 200)]][:, :2].mean(axis=0)
                r0, src = stem_prior(cloud.xyz, h, wood, stems); res.update(sylva_prior_radius_m=round(r0, 4), sylva_prior=src)
                model = qsm.build_qsm(wc, base_xy=base, base_radius=r0, max_radius=max(0.05, 2.0 * r0)); model.to_csv(qsm_csv)
                res.update({"sylva_" + k: (float(v) if v is not None else None) for k, v in model.summary().items()})
            else: res["sylva_note"] = "too few wood points"
            res["sylva_seconds"] = round(time.time() - t2, 1)
        res["seconds"] = round(time.time() - t0, 1)
    except Exception:
        res["error"] = traceback.format_exc().strip().splitlines()[-1][:300]
    return res


def scan_origins(site):
    """Scanner position per point_source_id, from the merge's matrices."""
    M, src = tern.matrices(site); j = json.load(open(ROOT / site / "03_registration" / f"{site}_03_transforms.json"))
    origins = np.zeros((max(x["index"] for x in j["scans"]) + 1, 3)); seen = set()
    for x in j["scans"]:
        n = x["name"] if x["name"] in M else x["name"].split("#")[0]
        if x["index"] not in seen and n in M: origins[x["index"]] = M[n][:3, 3]; seen.add(x["index"])
    return origins, src


def remove_tree(site, tid):
    for p in (ROOT / site / "08_qsm" / f"{site}_08_rct").glob(f"{site}_08_tree_{tid}_*"): p.unlink()
    for p in ((ROOT / site / "08_qsm" / f"{site}_08_sylva" / f"{site}_08_tree_{tid}_qsm.csv"), ROOT / site / "09_leafwood" / f"{site}_09_tree_{tid}.laz"): p.unlink(missing_ok=True)


def run(site, workers=6, only=None, do_rct=True, do_sylva=True, limit=None):
    if not tern.setting(site, "trees") and not tern.edited_tiles_dir(site).exists(): log(f"{site}: no trees"); return
    t0 = time.time(); q = tern.stage(site, "08_qsm"); lw = tern.stage(site, "09_leafwood")
    rct_dir, syl_dir = q / f"{site}_08_rct", q / f"{site}_08_sylva"; rct_dir.mkdir(exist_ok=True); syl_dir.mkdir(exist_ok=True)
    origins, src = scan_origins(site); dtm = tern.load_dtm(site); dtm_d = {"data": dtm.data, "xmin": dtm.xmin, "ymin": dtm.ymin, "resolution": dtm.resolution}
    nv, nf = write_ground_mesh(dtm, rct_dir / "ground_mesh.ply")
    trees = collect_trees(site, only); labels = "segfix edit" if tern.edited_tiles_dir(site).exists() else "transfer"
    rows_ = list(csv.DictReader(open(ROOT / site / "06_stems" / f"{site}_06_stems.csv")))
    stems = np.array([[float(r["x"]), float(r["y"]), float(r["dbh"])] for r in rows_]) if rows_ else None
    mf = q / f"{site}_08_models_manifest.json"
    man = tern.jload(mf) or {"site": site, "trees": {}}; man.update(matrices=src, labels=labels, ground_mesh=f"{nv} vertices, {nf} faces from the 1 m DTM")
    if not only:                                                         # trees removed in segfix
        for t in [t for t in man["trees"] if int(t) not in trees]: remove_tree(site, t); del man["trees"][t]
    def same(t):
        r = man["trees"].get(str(t), {}); return r.get("signature", f"{r.get('n_points')}") in (signature(trees[t][0]), str(len(trees[t][0])))
    def rct_ok(t): return same(t) and "rct_volume_m3" in man["trees"].get(str(t), {}) and (rct_dir / f"{site}_08_tree_{t}_trees.txt").exists()
    def sylva_ok(t): return same(t) and (lw / f"{site}_09_tree_{t}.laz").exists() and ("sylva_dbh_m" in man["trees"][str(t)] or "sylva_note" in man["trees"][str(t)])
    todo = [t for t in sorted(trees) if (do_rct and not rct_ok(t)) or (do_sylva and not sylva_ok(t))][:limit]
    log(f"{site}: {len(trees)} trees ({labels}), {sum(len(v[0]) for v in trees.values()):,} points; {len(todo)} to do on {workers} workers")
    jobs = [(site, t, trees[t][0], trees[t][1], origins, dtm_d, do_rct and not rct_ok(t), do_sylva and not sylva_ok(t), stems) for t in todo]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, r in enumerate(pool.map(one_tree, jobs, chunksize=1), 1):
            old = man["trees"].get(str(r["treeid"]), {})
            if old.get("signature", r["signature"]) == r["signature"] or str(old.get("n_points")) == r["signature"].split(":")[0]:
                for key, v in old.items():                              # keep the results of a step that was not redone
                    if key not in r and ((key.startswith("rct_") and "rct_mode" not in r) or (not key.startswith("rct_") and "leafwood_seconds" not in r)): r[key] = v
            man["trees"][str(r["treeid"])] = r; tern.jsave(man, mf)
            if "error" in r: log(f"{site}: tree {r['treeid']}: ERROR {r['error'][:150]}")
            elif k % 20 == 0 or k == len(jobs): log(f"{site}: {k}/{len(jobs)} trees ({(time.time() - t0) / 60:.0f} min)")
    rows = [man["trees"][k] for k in sorted(man["trees"], key=int)]
    def write_csv(path, keys):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows({k: r.get(k, "") for k in keys} for r in rows)
    write_csv(q / f"{site}_08_rct_trees.csv", ["treeid", "n_points", "rct_trees_found", "rct_segments", "rct_height_m", "rct_base_diameter_m", "rct_volume_m3", "rct_total_length_m", "rct_flag", "rct_seconds", "error"])
    write_csv(q / f"{site}_08_sylva_trees.csv", ["treeid", "n_points", "n_wood", "sylva_n_cylinders", "sylva_total_volume_m3", "sylva_stem_volume_m3", "sylva_branch_volume_m3", "sylva_total_length_m",
                                                 "sylva_max_branch_order", "sylva_dbh_m", "sylva_prior_radius_m", "sylva_prior", "sylva_note", "sylva_seconds", "error"])
    write_csv(lw / f"{site}_09_leafwood.csv", ["treeid", "n_points", "n_wood", "n_leaf", "wood_fraction", "height_m", "leaf_area_m2", "leaf_angle_mean_deg", "leaf_angle_std_deg",
                                              "leaf_beta_a", "leaf_beta_b", "leaf_chi", "leafwood_seconds", "error"])
    ok = [r for r in rows if "error" not in r]
    man["totals"] = {"trees": len(rows), "ok": len(ok), "errors": len(rows) - len(ok), "rct_volume_m3": round(sum(r.get("rct_volume_m3") or 0 for r in ok), 3),
                     "sylva_volume_m3": round(sum(r.get("sylva_total_volume_m3") or 0 for r in ok), 3), "leaf_area_m2": round(sum(r.get("leaf_area_m2") or 0 for r in ok), 1),
                     "seconds": round(time.time() - t0)}
    tern.jsave(man, mf); plot_meshes(site); tern.record_run(site, "08_tree_models", labels=labels, **man["totals"]); log(f"{site}: {man['totals']}")


def read_ply_mesh(path):
    with open(path, "rb") as f:
        hdr = []
        while True:
            l = f.readline().decode().strip(); hdr.append(l)
            if l == "end_header": break
        nv = int(next(l for l in hdr if l.startswith("element vertex")).split()[-1]); nf = int(next(l for l in hdr if l.startswith("element face")).split()[-1])
        props, in_v = [], False
        for l in hdr:
            if l.startswith("element vertex"): in_v = True; continue
            if l.startswith("element"): in_v = False
            if in_v and l.startswith("property"): props.append(l.split()[1:])
        types = {"double": "f8", "float": "f4", "uchar": "u1", "int": "i4", "uint": "u4"}; dt = [(n, types[t]) for t, n in props]
        V = np.frombuffer(f.read(nv * np.dtype(dt).itemsize), dtype=dt)
        F = np.frombuffer(f.read(nf * 16), dtype=[("n", "i4"), ("i", "i4", 3)]) if nf else None
    return np.stack([V["x"], V["y"], V["z"]], 1).astype(np.float64), (F["i"].astype(np.int64) if F is not None else np.zeros((0, 3), np.int64))


def colour(tid):
    r, g, b = colorsys.hsv_to_rgb((tid * 0.61803398875) % 1.0, 0.65, 0.95); return np.array([r * 255, g * 255, b * 255], np.uint8)


def write_mesh(path, parts):
    nv = sum(len(v) for v, f, c in parts); nf = sum(len(f) for v, f, c in parts)
    V = np.zeros(nv, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    F = np.zeros(nf, dtype=[("n", "u1"), ("i", "i4", 3)]); F["n"] = 3; ov = of = 0
    for v, f, c in parts:
        sl = slice(ov, ov + len(v)); V["x"][sl], V["y"][sl], V["z"][sl] = v[:, 0], v[:, 1], v[:, 2]; V["r"][sl], V["g"][sl], V["b"][sl] = c
        F["i"][of:of + len(f)] = f + ov; ov += len(v); of += len(f)
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {nv}\nproperty float x\nproperty float y\nproperty float z\n"
           f"property uchar red\nproperty uchar green\nproperty uchar blue\nelement face {nf}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as fh: fh.write(hdr.encode()); fh.write(V.tobytes()); fh.write(F.tobytes())


def plot_meshes(site):
    """Every tree's rct mesh and sylva QSM (12-sided tubes) merged per tool, with the ground mesh; project frame."""
    from sylva import qsm
    q = ROOT / site / "08_qsm"; rd = q / f"{site}_08_rct"; sd = q / f"{site}_08_sylva"; rct, syl = [], []
    for p in sorted(rd.glob(f"{site}_08_tree_*_trees_mesh.ply")):
        v, f = read_ply_mesh(p)
        if len(f): rct.append((v, f, colour(int(re.search(r"_tree_(\d+)_trees_mesh", p.name).group(1)))))
    for p in sorted(sd.glob(f"{site}_08_tree_*_qsm.csv")):
        m = qsm.QSM.from_csv(p)
        if len(m): v, f, *_ = m.mesh(sides=12, contiguous=True); syl.append((np.asarray(v, float), np.asarray(f, np.int64), colour(int(re.search(r"_tree_(\d+)_qsm", p.name).group(1)))))
    write_mesh(q / f"{site}_08_plot_rct_mesh.ply", rct); write_mesh(q / f"{site}_08_plot_sylva_mesh.ply", syl)
    gv, gf = read_ply_mesh(rd / "ground_mesh.ply"); write_mesh(q / f"{site}_08_plot_ground_mesh.ply", [(gv, gf, np.array([150, 130, 110], np.uint8))])
    log(f"{site}: plot meshes: {len(rct)} rct trees, {len(syl)} sylva trees")


def done(site):
    m = tern.jload(ROOT / site / "08_qsm" / f"{site}_08_models_manifest.json")
    if not tern.setting(site, "trees"): return True
    if not m or "totals" not in m: return False
    j = ROOT / site / "07_trees" / f"{site}_07_segfix_import.json"               # an import newer than the models means redo
    return not j.exists() or Path(ROOT / site / "08_qsm" / f"{site}_08_models_manifest.json").stat().st_mtime >= j.stat().st_mtime


if __name__ == "__main__":
    p = tern.parser(__doc__); p.add_argument("--workers", type=int, default=6); p.add_argument("--only", default=""); p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-rct", action="store_true"); p.add_argument("--no-sylva", action="store_true"); p.add_argument("--meshes-only", action="store_true"); a = p.parse_args()
    if a.meshes_only: tern.for_sites(plot_meshes, a.sites, "08")
    else: tern.for_sites(run, a.sites, "08", workers=a.workers, only={int(v) for v in a.only.split(",") if v}, do_rct=not a.no_rct, do_sylva=not a.no_sylva, limit=a.limit)
