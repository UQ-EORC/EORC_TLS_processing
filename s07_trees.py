"""Stage 07: tree labels on the 2 cm merge, from the manual trees or an automatic segmentation, and a review list for editing.

usage: s07_trees.py [PLOT ...] [--review-only] [--r-label=0.05] [--r-grow=0.04] [--r-near=0.15] [--grow-passes=2] [--no-clusters]

A plot with manual trees (manual_trees in project.toml) has them transferred; any other plot is segmented automatically
(segment_trees = false in project.toml turns that off). Either way the labels then go to segfix for checking (07e).

Transfer. The trees are one point cloud per tree (project.manual_tree_files: manual_trees, optionally listed in
manual_trees_table), in the frame of the matrices in manual_trees_frame (default: the merge's own). Where the merge uses
other matrices (e.g. a coreg set), each tree is moved by the rigid block motion under which it fits the merged cloud best.
Then, tile by tile, a merged point takes the id of the nearest tree point within r_label; for a few passes an unlabelled
point within r_grow of a labelled one and within r_near of any tree point takes that label; and small clusters of
unlabelled canopy that sit inside one tree's crown are joined to it (attach_clusters).
  07_trees/<PLOT>_07_merged_2cm_treeid_tiles/, <PLOT>_07_merged_2cm_treeid.laz   treeid per point (0 = no tree)
  07_trees/<PLOT>_07_trees.csv, <PLOT>_07_transfer.json                         per-tree counts and fit
Segmentation. The stems of stage 06 within the plot box plus 15 m grow into trees over the merge tiles, heights above the
stage-05 DTM, with sylva's tiled segmentation (sylva.als.tiles.segment_trees: merge_branches, segment_trees,
tree_heights, prune_trees, tile by tile with a buffer). Candidates that are branches of another stem are merged, those
under 3 m dropped.
  07_trees/<PLOT>_07_merged_2cm_treeid_tiles/, <PLOT>_07_merged_2cm_treeid.laz   treeid per point (0 = no tree)
  07_trees/<PLOT>_07_trees.csv, <PLOT>_07_segmentation.json                     per-tree stem, height, points; settings
  (the stem DBH there is the stage-06 fit, not a measurement to use: tree DBH comes from the stage-08 QSM)
The stem finder over-counts and splits some trees; the segfix edit (07e) is where that is corrected.
Review. For each tree, the unlabelled points more than 1 m above the local floor inside its crown envelope (plan hull of
its points above 30 % of its height), from the current labels (project.tree_tiles: the segfix edit once imported):
  07_trees/<PLOT>_07_review.csv (most affected first), <PLOT>_07_review_top3.png
The review list says where to start in segfix (s07_segfix.py).
"""
import csv, time
from pathlib import Path
import numpy as np
import project
from project import ROOT, log

# Bin edges (m) of the per-tree fit histogram: distance from each manual tree point to the nearest merged point, in 5 mm
# steps to 10 cm. The histogram call adds a last edge at 1 m, and distances are clipped to 10 cm, so the last bin
# collects everything from 10 cm up. The first 10 bins are the points within 5 cm.
BINS = np.arange(0, 0.1005, 0.005)                      # fit histogram, 5 mm steps to 10 cm (+ overflow)


def load_trees(site):
    """The plot's manual trees as flat arrays: (xyz float32 (N, 3), tree id int32 (N,), {tree id: (xyz min, xyz max)}).

    The points are in the frame of the manual trees (manual_trees_frame, or the merge's matrices). float32 halves the
    memory and keeps sub-millimetre precision while the project coordinates stay within a few km of the origin.
    """
    from sylva import io
    xyz, ids, bounds = [], [], {}
    for tid, f in project.manual_tree_files(site):
        c = io.read(str(f))
        xyz.append(c.xyz.astype(np.float32)); ids.append(np.full(len(c), tid, np.int32)); bounds[tid] = (c.xyz.min(axis=0), c.xyz.max(axis=0))
    return np.concatenate(xyz), np.concatenate(ids), bounds


class Tiles:
    """The merged cloud's tiles, read on demand (xyz as float32), with their bounds."""
    def __init__(self, site):
        """Catalogue the stage-04 tiles of the plot (bounds from the file headers; no points are read yet)."""
        from sylva.als import catalogue
        self.cat = catalogue.catalog(project.merged_tiles(site)); self.cache = {}   # cache: tile path -> xyz float32

    def points_in(self, lo, hi):
        """The merged points inside the axis-aligned box lo..hi (each (3,), project frame), from every tile it overlaps.

        Tiles are read once and kept in the cache, so repeated queries (one per tree and block) do not re-read them; the
        caller clears the cache when done. Returns an (n, 3) float32 array, empty when no tile overlaps.
        """
        from sylva import io
        parts = []
        for t in self.cat.tiles:
            b = t.bounds                                # (xmin, ymin, zmin, xmax, ymax, zmax)
            if b[3] < lo[0] or b[0] > hi[0] or b[4] < lo[1] or b[1] > hi[1]: continue   # no plan overlap
            if t.path not in self.cache: self.cache[t.path] = io.read(str(t.path)).xyz.astype(np.float32)
            p = self.cache[t.path]; m = np.all((p >= lo) & (p <= hi), axis=1); parts.append(p[m])
        return np.concatenate(parts) if parts else np.zeros((0, 3), np.float32)


def needs_reframe(site):
    """True when the merge's matrices differ from the trees' for any scan (more than 1e-5)."""
    a, b = project.trees_frame_dir(site), project.matrices_dir(site)
    if a == b: return False                             # the trees are in the merge's own frame
    old = {f.stem: np.loadtxt(f) for f in a.glob("*.DAT")}; new = {f.stem: np.loadtxt(f) for f in b.glob("*.DAT")}
    return any(not np.allclose(old[n], new[n], atol=1e-5) for n in old if n in new)


def reframe_trees(site, xyz, ids, bounds, tiles):
    """Each tree moved by the block motion (old -> new matrices; scans sharing a motion form a block) under which a sample of
    its points fits the merged cloud best (median nearest-point distance). Choosing by the nearest scanner failed for
    trees near block boundaries.

    xyz, ids, bounds are load_trees' output; xyz and bounds are updated in place. tiles: a Tiles of the merge.
    Returns {tree id: record} with the chosen block (first scan and how many more), the shift of the tree's centroid (cm),
    and the fit (median distance in cm, share within 5 cm); a tree with no merged points near any candidate position is
    left where it was, with a note.
    """
    from sylva import filters
    new = {f.stem: np.loadtxt(f) for f in project.matrices_dir(site).glob("*.DAT")}; old = {f.stem: np.loadtxt(f) for f in project.trees_frame_dir(site).glob("*.DAT")}
    # Per scan, the motion that takes a point from the trees' frame to the merge's: world_new @ inverse(world_old).
    F = {n: new[n] @ np.linalg.inv(old[n]) for n in sorted(new) if n in old}
    # Group scans whose motions agree (translation within 5 cm, rotation within 1e-4) into blocks: a re-registration
    # typically moves groups of scans together. Each block is represented by the motion of its first scan.
    blocks = []
    for n, m in F.items():
        for blk in blocks:
            if np.linalg.norm(m[:3, 3] - F[blk[0]][:3, 3]) < 0.05 and np.allclose(m[:3, :3], F[blk[0]][:3, :3], atol=1e-4): blk.append(n); break
        else: blocks.append([n])
    # Largest blocks first, so a tie in fit goes to the larger block. Fixed seed: the sample, hence the result, is repeatable.
    blocks.sort(key=len, reverse=True); rng = np.random.default_rng(0); rec = {}
    for tid in sorted(bounds):
        # Try each block's motion on a sample of up to 20,000 of the tree's points; score it by the median distance to
        # the merged points around the moved sample (box + 30 cm). best = (median, block, share within 5 cm).
        m = ids == tid; p = xyz[m].astype(np.float64); sample = p[rng.choice(len(p), min(len(p), 20_000), replace=False)]; best = None
        for blk in blocks:
            Fm = F[blk[0]]; q = sample @ Fm[:3, :3].T + Fm[:3, 3]
            target = tiles.points_in(q.min(axis=0) - 0.3, q.max(axis=0) + 0.3)
            if len(target) < 200: continue              # too few merged points there to judge the fit
            d = filters.knn(target.astype(np.float64), q, 1)[0][:, 0]; med = float(np.median(d))
            if best is None or med < best[0]: best = (med, blk, float((d < 0.05).mean()))
        if best is None:
            rec[tid] = {"block_scan": "", "block_shift_cm": "", "fit_median_cm": "", "fit_within_5cm": "", "note": "no merged points near the tree"}; continue
        # Apply the winning motion to all of the tree's points (in place) and update its bounds.
        med, blk, within = best; Fm = F[blk[0]]; q = p @ Fm[:3, :3].T + Fm[:3, 3]
        xyz[m] = q.astype(np.float32); bounds[tid] = (q.min(axis=0), q.max(axis=0))
        rec[tid] = {"block_scan": f"{blk[0]} (+{len(blk) - 1})", "block_shift_cm": round(100 * float(np.linalg.norm(q.mean(axis=0) - p.mean(axis=0))), 1),
                    "fit_median_cm": round(100 * med, 2), "fit_within_5cm": round(within, 3)}
    return rec


# attach_clusters settings (m unless noted): see its docstring. voxel: cluster connectivity grid; min_h: height above the
# local floor below which unlabelled points are ignored; max_points: largest cluster considered; dominance,
# min_embedded and max_share are fractions.
CLUSTER = dict(gap=0.2, voxel=0.08, min_h=0.5, max_extent=4.0, max_points=150_000, passes=3, dominance=0.8,
               embed_radius=0.5, min_embedded=0.5, min_cluster_floor=1.0, max_share=0.3)


def attach_clusters(P, lab, h, gap, voxel, min_h, max_extent, max_points, passes, dominance, embed_radius, min_embedded, min_cluster_floor, max_share):
    """Join whole clusters of unlabelled points to the one tree they belong to.

    Unlabelled points above min_h are grouped into 26-connected voxel clusters. A cluster joins a tree when the labelled
    points within `gap` of it are at least `dominance` that tree's; at least `min_embedded` of its points lie within
    `embed_radius` of labelled points (a fragment sits inside a crown, a neighbouring plant only touches its edge); its
    lowest point is over `min_cluster_floor` above the local floor (a plant with its own base is not a fragment); and it
    holds at most `max_share` of the tree's labelled points. Clusters wider than max_extent or larger than max_points are
    left. Repeated so clusters attach through one another.

    P: (n, 3) points of one tile; lab: their tree labels (0 unlabelled), updated in place; h: height of each point above
    the local floor (ground_height). Returns the number of points labelled.
    """
    from scipy import ndimage
    from sylva import filters
    n_added = 0
    for _ in range(passes):
        cand = np.flatnonzero((lab == 0) & (h > min_h))           # indices into P of the unlabelled candidates
        if len(cand) == 0 or not (lab > 0).any(): break
        # Voxelise the candidates and label connected components of occupied voxels (26-connectivity: faces, edges and
        # corners). cid[j] is the cluster of candidate j; cluster ids run 1..n (0 is empty space and never occurs here).
        Q = P[cand]; ijk = np.floor((Q - Q.min(axis=0)) / voxel).astype(np.int64)
        occ = np.zeros(ijk.max(axis=0) + 1, bool); occ[tuple(ijk.T)] = True
        comp, n = ndimage.label(occ, structure=np.ones((3, 3, 3))); cid = comp[tuple(ijk.T)]; del occ, comp
        # Per-cluster size, bounding box and lowest height above the floor, indexed by cluster id.
        counts = np.bincount(cid, minlength=n + 1)
        mn = np.full((n + 1, 3), np.inf); mx = np.full((n + 1, 3), -np.inf); hmin = np.full(n + 1, np.inf)
        for a in range(3): np.minimum.at(mn[:, a], cid, Q[:, a]); np.maximum.at(mx[:, a], cid, Q[:, a])
        np.minimum.at(hmin, cid, h[cand])
        # Clusters eligible to attach: not too many points, not too wide, and not reaching down to the floor.
        small = (counts <= max_points) & (np.max(mx - mn, axis=1) <= max_extent) & (hmin >= min_cluster_floor); small[0] = False
        if not small.any(): break
        # For each point of an eligible cluster: distance d to the nearest labelled point, and that point's label.
        keep = small[cid]; Qk, ck = Q[keep], cid[keep]
        L = np.flatnonzero(lab > 0); d, i = filters.knn(P[L].astype(np.float64), Qk.astype(np.float64), 1); d = d[:, 0]
        touch = d <= gap; contact = lab[L][i[:, 0]]
        if not touch.any(): break
        # Count the touching points per (cluster, tree) pair; best[c] = (tree with the most contacts, its share of the
        # cluster's contacts), i.e. the dominant neighbouring tree and how dominant it is.
        up, pc = np.unique(np.stack([ck[touch], contact[touch]], axis=1), axis=0, return_counts=True)
        tot = np.bincount(up[:, 0], weights=pc, minlength=n + 1); order = np.argsort(-pc); best = {}
        for (c, t), k in zip(up[order], pc[order]):
            if c not in best: best[c] = (int(t), k / tot[c])
        # Share of each cluster's points within embed_radius of the labelled cloud (only eligible clusters are non-zero).
        embedded = np.bincount(ck, weights=(d <= embed_radius), minlength=n + 1) / np.maximum(counts, 1)
        # assign[c]: the tree cluster c joins (0 = none); per_tree[t]: tree t's labelled points in this tile.
        per_tree = np.bincount(lab[lab > 0]); assign = np.zeros(n + 1, np.int32)
        for c, (t, dom) in best.items():
            if dom >= dominance and embedded[c] >= min_embedded and counts[c] <= max_share * per_tree[t]: assign[c] = t
        new = assign[cid]; take = new > 0
        if not take.any(): break
        lab[cand[take]] = new[take]; n_added += int(take.sum())
    return n_added


def ground_height(P, cell=1.0):
    """Height of each point above the lowest 2 % of its 1 m cell (ground, or the understorey floor).

    A cheap local floor that needs no DTM, computed per tile; P is (n, 3) in m, `cell` the plan cell size in m.
    Returns a float64 array (n,) in the order of P.
    """
    # Same sort-by-cell-then-z bookkeeping as s05_ground.lowest_points: each cell is a contiguous run, lowest point first.
    ij = np.floor((P[:, :2] - P[:, :2].min(axis=0)) / cell).astype(np.int64); key = ij[:, 0] * 100000 + ij[:, 1]
    order = np.lexsort((P[:, 2], key)); k = key[order]; z = P[order, 2]
    starts = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]; counts = np.diff(np.r_[starts, len(k)])
    # z minus the cell's 2nd-percentile z, written back to the points' original positions through `order`.
    out = np.empty(len(P)); out[order] = z - np.repeat(z[starts + (0.02 * counts).astype(np.int64)], counts); return out


def transfer(site, r_label=0.05, r_grow=0.04, r_near=0.15, grow_passes=2, clusters=True):
    """Carry the manual tree labels onto the stage-04 merge, tile by tile (see the module docstring for the method).

    Radii in m: r_label, the nearest tree point a merged point must be within to take its id; r_grow, the distance to an
    already labelled merged point for a growth pass; r_near, the distance to any tree point beyond which no growth
    happens. grow_passes: number of growth passes; clusters: whether to run attach_clusters.
    Writes 07_trees/<PLOT>_07_merged_2cm_treeid_tiles/*.laz (the merge tiles with a uint32 treeid, 0 = no tree), the
    same as one <PLOT>_07_merged_2cm_treeid.laz, <PLOT>_07_trees.csv (per tree), <PLOT>_07_transfer.json (summary), and
    when the trees had to be moved into the merge frame, <PLOT>_07_tree_points_reframed.npz.
    """
    from sylva import io, filters
    t0 = time.time(); out = project.stage(site, "07_trees")
    txyz, tids, bounds = load_trees(site); tiles = Tiles(site); reframe = None
    log(f"{site}: {len(bounds)} trees, {len(txyz):,} tree points")
    # Trees digitised against other registration matrices are first moved into the merge's frame (txyz in place).
    # The reframed points are saved so they can be inspected without redoing the fit.
    if needs_reframe(site):
        reframe = reframe_trees(site, txyz, tids, bounds, tiles); tiles.cache.clear()
        np.savez(out / f"{site}_07_tree_points_reframed.npz", xyz=txyz, treeid=tids)
        fm = [v["fit_median_cm"] for v in reframe.values() if v.get("fit_median_cm") != ""]
        log(f"{site}: trees moved into the merge frame: fit median {np.median(fm):.1f} cm, worst {max(fm):.1f} cm; not placed: {[t for t, v in reframe.items() if 'note' in v]}")
    tdir = out / f"{site}_07_merged_2cm_treeid_tiles"; tdir.mkdir(exist_ok=True)
    # Per-tree tallies over all tiles: n_tree, manual tree points scored for fit; hist, their fit histogram (BINS);
    # n_lab, merged points that end up with the tree's label.
    n_tree = {t: 0 for t in bounds}; hist = {t: np.zeros(len(BINS), np.int64) for t in bounds}; n_lab = {t: 0 for t in bounds}
    n_grown = n_clustered = n_total = 0; outs = []
    for k, t in enumerate(sorted(tiles.cat.tiles, key=lambda t: t.path), 1):
        c = io.read(str(t.path)); P = c.xyz.astype(np.float32); n_total += len(P); b = t.bounds
        # Tree points in the tile's plan extent plus r_near: enough to label every merged point of the tile.
        m = (txyz[:, 0] >= b[0] - r_near) & (txyz[:, 0] <= b[3] + r_near) & (txyz[:, 1] >= b[1] - r_near) & (txyz[:, 1] <= b[4] + r_near)
        lab = np.zeros(len(P), np.int32)                # the tile's labels, in the tile's point order (0 = no tree)
        if m.any():
            Q, qi = txyz[m].astype(np.float64), tids[m]
            # Direct transfer: each merged point's nearest tree point; its id is taken within r_label. `near` marks the
            # points close enough to the trees to be candidates for growth.
            d, i = filters.knn(Q, P.astype(np.float64), 1); d = d[:, 0]; i = i[:, 0]
            near = d <= r_near; hit = d <= r_label; lab[hit] = qi[i[hit]]
            # Growth: an unlabelled near point within r_grow of a labelled merged point takes that point's label. Fills
            # gaps where the 2 cm merge has points the manual tree lacks (just beyond r_label of any tree point).
            for _ in range(grow_passes):
                L = lab > 0; U = (~L) & near
                if not U.any() or not L.any(): break
                d2, i2 = filters.knn(P[L].astype(np.float64), P[U].astype(np.float64), 1); g = d2[:, 0] <= r_grow
                idx = np.flatnonzero(U)[g]; lab[idx] = lab[L][i2[g, 0]]; n_grown += int(g.sum())
            # Clusters are found within this tile only, so a cluster cut by a tile edge is judged as two parts.
            if clusters and (lab > 0).any(): n_clustered += attach_clusters(P, lab, ground_height(P), **CLUSTER)
            # Fit, tree -> merge: distance from each tree point to the nearest merged point, counted only for tree points
            # inside this tile (half-open bounds), so a point in the r_near margin is not counted in two tiles.
            inside = (Q[:, 0] >= b[0]) & (Q[:, 0] < b[3]) & (Q[:, 1] >= b[1]) & (Q[:, 1] < b[4])   # tree -> cloud fit, no double counting at edges
            if inside.any():
                dq = filters.knn(P.astype(np.float64), Q[inside], 1)[0][:, 0]; qin = qi[inside]
                for tid in np.unique(qin):
                    sel = qin == tid; n_tree[int(tid)] += int(sel.sum()); hist[int(tid)] += np.histogram(np.minimum(dq[sel], 0.1), bins=np.r_[BINS, 1.0])[0]
        for tid, cnt in zip(*np.unique(lab[lab > 0], return_counts=True)): n_lab[int(tid)] += int(cnt)
        # The tile is written with every original field plus treeid, in its original point order (segfix relies on it).
        p = tdir / Path(t.path).name; io.write(c.with_attrs(treeid=lab.astype(np.uint32)), str(p)); outs.append(str(p))
        if k % 10 == 0 or k == len(tiles.cat.tiles): log(f"{site}: tile {k}/{len(tiles.cat.tiles)}: labelled {sum(n_lab.values()):,} of {n_total:,}")
    io.write_concat(outs, str(out / f"{site}_07_merged_2cm_treeid.laz"))
    # Median fit (cm) from a histogram: the lower edge of the bin holding the middle point; NaN for a tree with no points.
    def median_fit(h): return float(BINS[min(int(np.searchsorted(np.cumsum(h), h.sum() / 2)), len(BINS) - 1)]) * 100 if h.sum() else float("nan")
    # Per-tree table: points labelled in the merge, tree points scored, share of them within 5 cm (bins 0-9), median
    # fit, and the reframing record where the tree was moved.
    with open(out / f"{site}_07_trees.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["treeid", "n_points_new_cloud", "n_tree_points", "within_5cm_frac", "median_fit_cm", "block_scan", "block_shift_cm", "reframe_fit_median_cm", "reframe_within_5cm"])
        for tid in sorted(bounds):
            h = hist[tid]; n = h.sum(); rf = (reframe or {}).get(tid, {})
            w.writerow([tid, n_lab[tid], n_tree[tid], f"{h[:10].sum() / n:.3f}" if n else "nan", f"{median_fit(h):.1f}",
                        rf.get("block_scan", ""), rf.get("block_shift_cm", ""), rf.get("fit_median_cm", ""), rf.get("fit_within_5cm", "")])
    fits = [median_fit(h) for h in hist.values() if h.sum()]
    summary = {"site": site, "trees": len(bounds), "tree_points": int(len(txyz)), "merged_points": n_total, "labelled_points": int(sum(n_lab.values())),
               "of_which_grown": n_grown, "of_which_cluster_attached": n_clustered, "trees_with_no_points": [t for t, n in n_lab.items() if n == 0],
               "median_fit_cm": float(np.median(fits)) if fits else None,
               "tree_points_within_5cm": float(sum(h[:10].sum() for h in hist.values()) / max(sum(h.sum() for h in hist.values()), 1)),
               "settings": {"r_label": r_label, "r_grow": r_grow, "r_near": r_near, "grow_passes": grow_passes, "clusters": clusters, **{"cluster_" + k: v for k, v in CLUSTER.items()}},
               "frame": f"{project.matrices_dir(site).name}" + (f"; each tree moved from {project.trees_frame_dir(site).name} by its best-fitting block motion" if reframe else ""),
               "source": str(project.setting_path(site, "manual_trees")), "seconds": round(time.time() - t0)}
    # The transfer record is written last; done() checks for it.
    project.jsave(summary, out / f"{site}_07_transfer.json"); project.record_run(site, "07_transfer", labelled=summary["labelled_points"], trees=len(bounds))
    log(f"{site}: {summary['labelled_points']:,} of {n_total:,} points labelled ({n_grown:,} grown, {n_clustered:,} as clusters); median fit {summary['median_fit_cm']:.1f} cm")

# Automatic segmentation settings. margin: metres around the plot box segmented too, so trees leaning in or out of the box
# are whole; buffer: the band of neighbouring tiles each tile is segmented with (wider than the largest crown; sylva widens
# it up to max_buffer where crowns reach its edge); min_height: candidates lower than this are dropped (sylva's default);
# piece: side (m) of the pieces the merge tiles are cut into first, small enough that a densely scanned one (tens of
# thousands of points per m2 near the scanner) and its neighbours fit in memory, and sylva's estimate of what a piece
# and its buffer hold (whole files) close; voxels: the graph voxel sizes (m) tried in turn, the next one only when the
# graph of a piece and its buffer does not fit in memory (sylva's default is the first).
SEGMENT = dict(margin=15.0, buffer=20.0, max_buffer=60.0, min_height=3.0, piece=10.0, voxels=(0.03, 0.05, 0.08))


def _pieces(c, piece):
    """For each point of a cloud, the (column, row) of the piece-sized square it falls in (corners at multiples of piece)."""
    return np.floor(c.xyz[:, :2] / piece).astype(np.int64)


def read_stems(site):
    """The automatic stems of stage 06 (06_stems/<PLOT>_06_stems.csv) as sylva Tree objects, with their ids."""
    from sylva.trees import Tree
    out = []
    for r in csv.DictReader(open(ROOT / site / "06_stems" / f"{site}_06_stems.csv")):
        f = lambda k, d=float("nan"): float(r[k]) if r.get(k) not in (None, "", "nan") else d
        out.append(Tree(int(r["tree_id"]), f("x"), f("y"), dbh=f("dbh"), n_points=int(f("n_points", 0)), rmse=f("rmse"),
                        lean_deg=f("lean_deg"), quality=f("quality")))
    return out


def segment(site, margin, buffer, max_buffer, min_height, piece, voxels, workers=None):
    """Tree labels for a plot without manual trees: the stage-06 stems grown into trees over the 2 cm merge.

    sylva's tiled segmentation (sylva.als.tiles.segment_trees) on the merge tiles that touch the plot box plus margin,
    height-normalised with the stage-05 DTM: stem candidates that are branches of another are merged, each stem grows
    into its tree by least-cost paths through a 3 cm graph of the points, heights are measured and candidates under
    min_height or duplicated are dropped (prune). It gives the labels the whole-plot sequence would, tile by tile. The
    merge tiles are first cut into pieces of side piece, each with its height above the DTM (only coordinates and
    height are kept), since sylva holds a tile with its neighbours in memory and a merge tile can hold 200 M points. Only
    stems within the box plus margin seed trees. Every point then has a tree id or none (sylva's -1, written 0 here: -1
    means noise in segfix). The labels are written onto copies of the merge tiles exactly as transfer() writes them, so
    the review, segfix and stage 08 treat both alike. The normalised and segmented tiles are temporary (SCRATCH/segment).
    """
    import shutil, warnings
    from sylva import io
    from sylva.als import catalogue, tiles as stiles
    from sylva.ground import normalize_height
    from sylva.pointcloud import PointCloud
    t0 = time.time(); out = project.stage(site, "07_trees")
    x0, y0, x1, y1 = project.plot_bounds(site); lo = (x0 - margin, y0 - margin); hi = (x1 + margin, y1 + margin)
    every = sorted(Tiles(site).cat.tiles, key=lambda t: t.path)
    use = [t for t in every if not (t.bounds[3] < lo[0] or t.bounds[0] > hi[0] or t.bounds[4] < lo[1] or t.bounds[1] > hi[1])]
    stems = [s for s in read_stems(site) if lo[0] <= s.x <= hi[0] and lo[1] <= s.y <= hi[1]]
    tmp = project.SCRATCH / "segment" / site
    if tmp.exists(): shutil.rmtree(tmp)
    log(f"{site}: segmenting {len(use)} of {len(every)} tiles from {len(stems)} stem candidates within the plot box + {margin:g} m")
    # each tile cut into pieces with heights above the DTM of stage 05, so the trees stand on the same ground as the rest
    # of the products; a piece keeps its points in the tile's order, which is how the labels are put back below
    dtm = project.load_dtm(site); (tmp / "normalised").mkdir(parents=True); pieces = []
    for t in use:
        c = io.read(str(t.path), attrs=[]); cell = _pieces(c, piece)
        for cx, cy in np.unique(cell, axis=0):
            sub = PointCloud(c.xyz[(cell[:, 0] == cx) & (cell[:, 1] == cy)])
            p = tmp / "normalised" / f"{Path(t.path).stem}__{cx}_{cy}.laz"
            io.write(normalize_height(sub, dtm), str(p)); pieces.append(str(p))
        del c, cell
    log(f"{site}: {len(use)} tiles cut into {len(pieces)} pieces of {piece:g} m")
    norm = catalogue.catalog(pieces)
    # the finest graph that fits: densely scanned plots have too many 3 cm voxels within a piece and its buffer
    for i, voxel in enumerate(voxels):
        try:
            with warnings.catch_warnings(record=True) as warned:
                warnings.simplefilter("always")
                found, seg = stiles.segment_trees(norm, stems, tmp / "segmented", buffer=buffer, max_buffer=max_buffer,
                                                  merge={"voxel_size": voxel}, voxel_size=voxel,
                                                  prune={"min_height": min_height}, attribute="tree_id", workers=workers)
            break
        except ValueError as e:
            if "GB" not in str(e) or i == len(voxels) - 1: raise
            log(f"{site}: {voxel * 100:g} cm graph does not fit in memory ({e}); trying {voxels[i + 1] * 100:g} cm")
            shutil.rmtree(tmp / "segmented", ignore_errors=True)
    for w in warned: log(f"{site}: sylva: {w.message}")
    log(f"{site}: {len(found)} trees ({time.time() - t0:.0f} s)")
    # the labels onto copies of the merge tiles: each piece holds its tile's points in order, checked
    by_name = {Path(t.path).stem: t.path for t in seg.tiles}
    tdir = out / f"{site}_07_merged_2cm_treeid_tiles"
    if tdir.exists(): shutil.rmtree(tdir)
    tdir.mkdir(); outs = []; n_total = n_lab = 0
    for t in every:
        c = io.read(str(t.path)); lab = np.zeros(len(c), np.int32); n_total += len(c); name = Path(t.path).name
        if t in use:
            cell = _pieces(c, piece)
            for cx, cy in np.unique(cell, axis=0):
                m = (cell[:, 0] == cx) & (cell[:, 1] == cy); sc = io.read(str(by_name[f"{Path(name).stem}__{cx}_{cy}"]), attrs=["tree_id"])
                if len(sc) != int(m.sum()) or not np.allclose(sc.xyz, c.xyz[m], atol=1e-3):
                    raise RuntimeError(f"{site}: segmented piece {cx}_{cy} of {name} does not hold the merge tile's points in order")
                lab[m] = np.maximum(sc.attrs["tree_id"].astype(np.int32), 0)
            del cell
        n_lab += int((lab > 0).sum())
        p = tdir / name; io.write(c.with_attrs(treeid=lab.astype(np.uint32)), str(p)); outs.append(str(p))
    io.write_concat(outs, str(out / f"{site}_07_merged_2cm_treeid.laz"))
    shutil.rmtree(tmp)
    with open(out / f"{site}_07_trees.csv", "w", newline="") as f:
        # stem_dbh_06 is the stage-06 circle fit, inflated on multi-stem clumps and shrubs; the tree DBH is the stage-08 QSM's
        w = csv.writer(f); w.writerow(["treeid", "x", "y", "stem_dbh_06", "height", "n_points", "stem_quality", "crown_at_edge"])
        for t in found:
            w.writerow([t.tree_id, f"{t.x:.3f}", f"{t.y:.3f}", f"{t.dbh:.3f}", f"{t.height:.2f}", t.n_points, f"{t.quality:.3f}",
                        int(bool(t.extra.get("crown_at_edge")))])
    summary = {"site": site, "method": "sylva.als.tiles.segment_trees (merge_branches, segment_trees, tree_heights, prune_trees) from the stage-06 stems",
               "stem_candidates": len(stems), "trees": len(found), "merged_points": n_total, "labelled_points": n_lab,
               "tiles_segmented": len(use), "area": {"xmin": lo[0], "ymin": lo[1], "xmax": hi[0], "ymax": hi[1]},
               "settings": {"margin": margin, "buffer": buffer, "max_buffer": max_buffer, "min_height": min_height, "piece": piece,
                            "voxel": voxel},
               "warnings": [str(w.message) for w in warned], "sylva": project.sylva_version(), "seconds": round(time.time() - t0)}
    project.jsave(summary, out / f"{site}_07_segmentation.json"); project.record_run(site, "07_segmentation", trees=len(found), labelled=n_lab)
    log(f"{site}: {n_lab:,} of {n_total:,} points labelled into {len(found)} trees ({summary['seconds']} s)")


def review(site):
    """Rank the trees by how much unlabelled canopy lies inside their crown envelope: where to start editing in segfix.

    Reads the current labels (project.tree_tiles: the segfix-edited tiles once imported, else the transfer's; treeid > 0
    tree, 0 unassigned, -1 noise). Two passes over the tiles: the first collects each tree's points (thinned to 10 cm)
    to build its envelope, the plan convex hull of its points above 30 % of its height range, between that level and its
    top; the second counts the unlabelled points more than 1 m above the local floor inside each envelope.
    Writes 07_trees/<PLOT>_07_review.csv (most unlabelled points first) and <PLOT>_07_review_top3.png (side and plan
    views of the three worst trees).
    """
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath
    from scipy.spatial import ConvexHull
    from sylva import io
    from sylva.als import catalogue
    d = project.stage(site, "07_trees"); tiles = sorted(catalogue.catalog(project.tree_tiles(site)).tiles, key=lambda t: t.path)
    pts = {}                                           # pass 1: each tree's points, thinned to 10 cm
    for t in tiles:
        c = io.read(str(t.path)); lab = c.attrs["treeid"].astype(np.int64); m = lab > 0
        if not m.any(): continue
        # One point per (10 cm voxel, tree): np.unique over voxel index and label, keeping the first of each.
        xyz = c.xyz[m]; lb = lab[m]; _, first = np.unique(np.c_[np.floor(xyz / 0.1).astype(np.int64), lb], axis=0, return_index=True)
        for tid in np.unique(lb[first]): pts.setdefault(int(tid), []).append(xyz[first][lb[first] == tid])
    # hulls[tid] = (plan hull path, plan bbox min, plan bbox max, crown base z, top z). The bbox is a fast pre-test before
    # the point-in-polygon test. Trees with too few crown points or a degenerate hull (e.g. collinear) are skipped.
    hulls = {}
    for tid, parts in pts.items():
        P = np.concatenate(parts); z0, z1 = P[:, 2].min(), P[:, 2].max(); crown = P[P[:, 2] >= z0 + 0.3 * (z1 - z0)]
        if len(crown) < 10: continue
        try: h = ConvexHull(crown[:, :2]); hulls[tid] = (MplPath(crown[h.vertices, :2]), crown[:, :2].min(axis=0), crown[:, :2].max(axis=0), z0 + 0.3 * (z1 - z0), z1)
        except Exception: continue
    # inside[tid]: unlabelled canopy points in the envelope; labelled[tid]: the tree's own points (all, not thinned);
    # sample[tid]: every 5th of the unlabelled points in the envelope, for the figure.
    inside = {t: 0 for t in hulls}; labelled = {t: 0 for t in hulls}; sample = {}    # pass 2: unlabelled canopy inside each envelope
    for t in tiles:
        c = io.read(str(t.path)); lab = c.attrs["treeid"].astype(np.int64); P = c.xyz; h = ground_height(P)
        for tid, cnt in zip(*np.unique(lab[lab > 0], return_counts=True)):
            if int(tid) in labelled: labelled[int(tid)] += int(cnt)
        # Unassigned points (0; noise, -1, is left out) more than 1 m above the local floor, so ground and low
        # understorey under a crown do not count.
        Pu = P[(lab == 0) & (h > 1.0)]; b = t.bounds
        for tid, (path, lo, hi, zlo, zhi) in hulls.items():
            if hi[0] < b[0] or lo[0] > b[3] or hi[1] < b[1] or lo[1] > b[4]: continue   # envelope does not reach this tile
            m = (Pu[:, 0] >= lo[0]) & (Pu[:, 0] <= hi[0]) & (Pu[:, 1] >= lo[1]) & (Pu[:, 1] <= hi[1]) & (Pu[:, 2] >= zlo) & (Pu[:, 2] <= zhi)
            if not m.any(): continue
            k = path.contains_points(Pu[m][:, :2]); inside[tid] += int(k.sum())
            if k.any(): sample.setdefault(tid, []).append(Pu[m][k][::5])
    # Most unlabelled canopy first (ties broken by labelled points, then id). ratio: unlabelled in the envelope per own point.
    rows = sorted(((inside[t], labelled[t], t) for t in hulls), reverse=True)
    with open(d / f"{site}_07_review.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["treeid", "unlabelled_canopy_in_envelope", "labelled_points", "ratio", "crown_base_z", "top_z"])
        for u, l, t in rows: w.writerow([t, u, l, f"{u / max(l, 1):.3f}", f"{hulls[t][3]:.2f}", f"{hulls[t][4]:.2f}"])
    log(f"{site}: {len(hulls)} trees reviewed; over 20 % of their own points unlabelled in the envelope: {sum(1 for u, l, t in rows if u > 0.2 * max(l, 1))}")
    # Figure: the (up to) three worst trees, side view (x, z) above and plan view (x, y) below; the tree's own points
    # (10 cm thinned) in green, the unlabelled points in its envelope in orange.
    top = [t for u, l, t in rows[:3] if u > 0]
    if top:
        fig, axes = plt.subplots(2, len(top), figsize=(6 * len(top), 11), layout="constrained", squeeze=False)
        for k, tid in enumerate(top):
            L = np.concatenate(pts[tid]); U = np.concatenate(sample[tid])
            for row, (i, j, name) in enumerate(((0, 2, "side"), (0, 1, "plan"))):   # (horizontal axis, vertical axis) columns of xyz
                ax = axes[row, k]; ax.scatter(L[:, i], L[:, j], s=0.6, c="#009E73", rasterized=True, label=f"tree {tid}")
                ax.scatter(U[:, i], U[:, j], s=0.8, c="#D55E00", rasterized=True, label="unlabelled in its envelope"); ax.set_aspect("equal"); ax.set_title(f"tree {tid}: {name}", fontsize=10)
                if row == 0 and k == 0: ax.legend(markerscale=10, fontsize=8)
        fig.suptitle(f"{site}: trees with the most unlabelled canopy inside their crown envelope", fontsize=10)
        fig.savefig(d / f"{site}_07_review_top3.png", dpi=70); plt.close(fig)


def run(site, review_only=False, **kw):
    """Stage 07 for one plot: the transfer (unless review_only) and then the review. kw are transfer()'s settings.

    Plots without manual trees are skipped. With review_only, the review is redone on the current labels (e.g. after a
    segfix import) without touching the transferred ones.
    """
    if not project.has_trees(site): log(f"{site}: no manual trees and segment_trees = false; no tree labels"); return
    if not review_only:
        if project.has_manual_trees(site): transfer(site, **kw)
        else: segment(site, **SEGMENT)
    review(site)


def done(site):
    """Whether stage 07 has completed for the plot (transfer or segmentation record, and the review list, exist); always true
    for a plot that gets no tree labels."""
    d = ROOT / site / "07_trees"
    record = d / (f"{site}_07_transfer.json" if project.has_manual_trees(site) else f"{site}_07_segmentation.json")
    return not project.has_trees(site) or (record.exists() and (d / f"{site}_07_review.csv").exists())


if __name__ == "__main__":
    p = project.parser(__doc__); p.add_argument("--review-only", action="store_true"); p.add_argument("--no-clusters", action="store_true")
    for k, v in (("r-label", 0.05), ("r-grow", 0.04), ("r-near", 0.15)): p.add_argument(f"--{k}", type=float, default=v)
    p.add_argument("--grow-passes", type=int, default=2); a = p.parse_args()
    project.for_sites(run, a.sites, "07", review_only=a.review_only, r_label=a.r_label, r_grow=a.r_grow, r_near=a.r_near,
                   grow_passes=a.grow_passes, clusters=not a.no_clusters)
