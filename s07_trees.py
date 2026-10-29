"""Stage 07: manually segmented trees carried onto the 2 cm merge, and a review list for editing.

usage: s07_trees.py [PLOT ...] [--review-only] [--r-label=0.05] [--r-grow=0.04] [--r-near=0.15] [--grow-passes=2] [--no-clusters]

Transfer. The trees are one point cloud per tree (project.manual_tree_files: manual_trees, optionally listed in
manual_trees_table), in the frame of the matrices in manual_trees_frame (default: the merge's own). Where the merge uses
other matrices (e.g. a coreg set), each tree is moved by the rigid block motion under which it fits the merged cloud best.
Then, tile by tile, a merged point takes the id of the nearest tree point within r_label; for a few passes an unlabelled
point within r_grow of a labelled one and within r_near of any tree point takes that label; and small clusters of
unlabelled canopy that sit inside one tree's crown are joined to it (attach_clusters).
  07_trees/<PLOT>_07_merged_2cm_treeid_tiles/, <PLOT>_07_merged_2cm_treeid.laz   treeid per point (0 = no tree)
  07_trees/<PLOT>_07_trees.csv, <PLOT>_07_transfer.json                         per-tree counts and fit
Review. For each tree, the unlabelled points more than 1 m above the local floor inside its crown envelope (plan hull of
its points above 30 % of its height), from the current labels (project.tree_tiles: the segfix edit once imported):
  07_trees/<PLOT>_07_review.csv (most affected first), <PLOT>_07_review_top3.png
The review list says where to start in segfix (s07_segfix.py).
"""
import csv, json, time
from pathlib import Path
import numpy as np
import project
from project import ROOT, log

BINS = np.arange(0, 0.1005, 0.005)                      # fit histogram, 5 mm steps to 10 cm (+ overflow)


def load_trees(site):
    from sylva import io
    xyz, ids, bounds = [], [], {}
    for tid, f in project.manual_tree_files(site):
        c = io.read(str(f))
        xyz.append(c.xyz.astype(np.float32)); ids.append(np.full(len(c), tid, np.int32)); bounds[tid] = (c.xyz.min(axis=0), c.xyz.max(axis=0))
    return np.concatenate(xyz), np.concatenate(ids), bounds


class Tiles:
    """The merged cloud's tiles, read on demand (xyz as float32), with their bounds."""
    def __init__(self, site):
        from sylva.als import catalogue
        self.cat = catalogue.catalog(project.merged_tiles(site)); self.cache = {}

    def points_in(self, lo, hi):
        from sylva import io
        parts = []
        for t in self.cat.tiles:
            b = t.bounds
            if b[3] < lo[0] or b[0] > hi[0] or b[4] < lo[1] or b[1] > hi[1]: continue
            if t.path not in self.cache: self.cache[t.path] = io.read(str(t.path)).xyz.astype(np.float32)
            p = self.cache[t.path]; m = np.all((p >= lo) & (p <= hi), axis=1); parts.append(p[m])
        return np.concatenate(parts) if parts else np.zeros((0, 3), np.float32)


def needs_reframe(site):
    """True when the merge's matrices differ from the trees' for any scan (more than 1e-5)."""
    a, b = project.trees_frame_dir(site), project.matrices_dir(site)
    if a == b: return False
    old = {f.stem: np.loadtxt(f) for f in a.glob("*.DAT")}; new = {f.stem: np.loadtxt(f) for f in b.glob("*.DAT")}
    return any(not np.allclose(old[n], new[n], atol=1e-5) for n in old if n in new)


def reframe_trees(site, xyz, ids, bounds, tiles):
    """Each tree moved by the block motion (old -> new matrices; scans sharing a motion form a block) under which a sample of
    its points fits the merged cloud best (median nearest-point distance). Choosing by the nearest scanner failed for
    trees near block boundaries."""
    from sylva import filters
    new = {f.stem: np.loadtxt(f) for f in project.matrices_dir(site).glob("*.DAT")}; old = {f.stem: np.loadtxt(f) for f in project.trees_frame_dir(site).glob("*.DAT")}
    F = {n: new[n] @ np.linalg.inv(old[n]) for n in sorted(new) if n in old}
    blocks = []
    for n, m in F.items():
        for blk in blocks:
            if np.linalg.norm(m[:3, 3] - F[blk[0]][:3, 3]) < 0.05 and np.allclose(m[:3, :3], F[blk[0]][:3, :3], atol=1e-4): blk.append(n); break
        else: blocks.append([n])
    blocks.sort(key=len, reverse=True); rng = np.random.default_rng(0); rec = {}
    for tid in sorted(bounds):
        m = ids == tid; p = xyz[m].astype(np.float64); sample = p[rng.choice(len(p), min(len(p), 20_000), replace=False)]; best = None
        for blk in blocks:
            Fm = F[blk[0]]; q = sample @ Fm[:3, :3].T + Fm[:3, 3]
            target = tiles.points_in(q.min(axis=0) - 0.3, q.max(axis=0) + 0.3)
            if len(target) < 200: continue
            d = filters.knn(target.astype(np.float64), q, 1)[0][:, 0]; med = float(np.median(d))
            if best is None or med < best[0]: best = (med, blk, float((d < 0.05).mean()))
        if best is None:
            rec[tid] = {"block_scan": "", "block_shift_cm": "", "fit_median_cm": "", "fit_within_5cm": "", "note": "no merged points near the tree"}; continue
        med, blk, within = best; Fm = F[blk[0]]; q = p @ Fm[:3, :3].T + Fm[:3, 3]
        xyz[m] = q.astype(np.float32); bounds[tid] = (q.min(axis=0), q.max(axis=0))
        rec[tid] = {"block_scan": f"{blk[0]} (+{len(blk) - 1})", "block_shift_cm": round(100 * float(np.linalg.norm(q.mean(axis=0) - p.mean(axis=0))), 1),
                    "fit_median_cm": round(100 * med, 2), "fit_within_5cm": round(within, 3)}
    return rec


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
    """
    from scipy import ndimage
    from sylva import filters
    n_added = 0
    for _ in range(passes):
        cand = np.flatnonzero((lab == 0) & (h > min_h))
        if len(cand) == 0 or not (lab > 0).any(): break
        Q = P[cand]; ijk = np.floor((Q - Q.min(axis=0)) / voxel).astype(np.int64)
        occ = np.zeros(ijk.max(axis=0) + 1, bool); occ[tuple(ijk.T)] = True
        comp, n = ndimage.label(occ, structure=np.ones((3, 3, 3))); cid = comp[tuple(ijk.T)]; del occ, comp
        counts = np.bincount(cid, minlength=n + 1)
        mn = np.full((n + 1, 3), np.inf); mx = np.full((n + 1, 3), -np.inf); hmin = np.full(n + 1, np.inf)
        for a in range(3): np.minimum.at(mn[:, a], cid, Q[:, a]); np.maximum.at(mx[:, a], cid, Q[:, a])
        np.minimum.at(hmin, cid, h[cand])
        small = (counts <= max_points) & (np.max(mx - mn, axis=1) <= max_extent) & (hmin >= min_cluster_floor); small[0] = False
        if not small.any(): break
        keep = small[cid]; Qk, ck = Q[keep], cid[keep]
        L = np.flatnonzero(lab > 0); d, i = filters.knn(P[L].astype(np.float64), Qk.astype(np.float64), 1); d = d[:, 0]
        touch = d <= gap; contact = lab[L][i[:, 0]]
        if not touch.any(): break
        up, pc = np.unique(np.stack([ck[touch], contact[touch]], axis=1), axis=0, return_counts=True)
        tot = np.bincount(up[:, 0], weights=pc, minlength=n + 1); order = np.argsort(-pc); best = {}
        for (c, t), k in zip(up[order], pc[order]):
            if c not in best: best[c] = (int(t), k / tot[c])
        embedded = np.bincount(ck, weights=(d <= embed_radius), minlength=n + 1) / np.maximum(counts, 1)
        per_tree = np.bincount(lab[lab > 0]); assign = np.zeros(n + 1, np.int32)
        for c, (t, dom) in best.items():
            if dom >= dominance and embedded[c] >= min_embedded and counts[c] <= max_share * per_tree[t]: assign[c] = t
        new = assign[cid]; take = new > 0
        if not take.any(): break
        lab[cand[take]] = new[take]; n_added += int(take.sum())
    return n_added


def ground_height(P, cell=1.0):
    """Height of each point above the lowest 2 % of its 1 m cell (ground, or the understorey floor)."""
    ij = np.floor((P[:, :2] - P[:, :2].min(axis=0)) / cell).astype(np.int64); key = ij[:, 0] * 100000 + ij[:, 1]
    order = np.lexsort((P[:, 2], key)); k = key[order]; z = P[order, 2]
    starts = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]; counts = np.diff(np.r_[starts, len(k)])
    out = np.empty(len(P)); out[order] = z - np.repeat(z[starts + (0.02 * counts).astype(np.int64)], counts); return out


def transfer(site, r_label=0.05, r_grow=0.04, r_near=0.15, grow_passes=2, clusters=True):
    from sylva import io, filters
    t0 = time.time(); out = project.stage(site, "07_trees")
    txyz, tids, bounds = load_trees(site); tiles = Tiles(site); reframe = None
    log(f"{site}: {len(bounds)} trees, {len(txyz):,} tree points")
    if needs_reframe(site):
        reframe = reframe_trees(site, txyz, tids, bounds, tiles); tiles.cache.clear()
        np.savez(out / f"{site}_07_tree_points_reframed.npz", xyz=txyz, treeid=tids)
        fm = [v["fit_median_cm"] for v in reframe.values() if v.get("fit_median_cm") != ""]
        log(f"{site}: trees moved into the merge frame: fit median {np.median(fm):.1f} cm, worst {max(fm):.1f} cm; not placed: {[t for t, v in reframe.items() if 'note' in v]}")
    tdir = out / f"{site}_07_merged_2cm_treeid_tiles"; tdir.mkdir(exist_ok=True)
    n_tree = {t: 0 for t in bounds}; hist = {t: np.zeros(len(BINS), np.int64) for t in bounds}; n_lab = {t: 0 for t in bounds}
    n_grown = n_clustered = n_total = 0; outs = []
    for k, t in enumerate(sorted(tiles.cat.tiles, key=lambda t: t.path), 1):
        c = io.read(str(t.path)); P = c.xyz.astype(np.float32); n_total += len(P); b = t.bounds
        m = (txyz[:, 0] >= b[0] - r_near) & (txyz[:, 0] <= b[3] + r_near) & (txyz[:, 1] >= b[1] - r_near) & (txyz[:, 1] <= b[4] + r_near)
        lab = np.zeros(len(P), np.int32)
        if m.any():
            Q, qi = txyz[m].astype(np.float64), tids[m]
            d, i = filters.knn(Q, P.astype(np.float64), 1); d = d[:, 0]; i = i[:, 0]
            near = d <= r_near; hit = d <= r_label; lab[hit] = qi[i[hit]]
            for _ in range(grow_passes):
                L = lab > 0; U = (~L) & near
                if not U.any() or not L.any(): break
                d2, i2 = filters.knn(P[L].astype(np.float64), P[U].astype(np.float64), 1); g = d2[:, 0] <= r_grow
                idx = np.flatnonzero(U)[g]; lab[idx] = lab[L][i2[g, 0]]; n_grown += int(g.sum())
            if clusters and (lab > 0).any(): n_clustered += attach_clusters(P, lab, ground_height(P), **CLUSTER)
            inside = (Q[:, 0] >= b[0]) & (Q[:, 0] < b[3]) & (Q[:, 1] >= b[1]) & (Q[:, 1] < b[4])   # tree -> cloud fit, no double counting at edges
            if inside.any():
                dq = filters.knn(P.astype(np.float64), Q[inside], 1)[0][:, 0]; qin = qi[inside]
                for tid in np.unique(qin):
                    sel = qin == tid; n_tree[int(tid)] += int(sel.sum()); hist[int(tid)] += np.histogram(np.minimum(dq[sel], 0.1), bins=np.r_[BINS, 1.0])[0]
        for tid, cnt in zip(*np.unique(lab[lab > 0], return_counts=True)): n_lab[int(tid)] += int(cnt)
        p = tdir / Path(t.path).name; io.write(c.with_attrs(treeid=lab.astype(np.uint32)), str(p)); outs.append(str(p))
        if k % 10 == 0 or k == len(tiles.cat.tiles): log(f"{site}: tile {k}/{len(tiles.cat.tiles)}: labelled {sum(n_lab.values()):,} of {n_total:,}")
    io.write_concat(outs, str(out / f"{site}_07_merged_2cm_treeid.laz"))
    def median_fit(h): return float(BINS[min(int(np.searchsorted(np.cumsum(h), h.sum() / 2)), len(BINS) - 1)]) * 100 if h.sum() else float("nan")
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
    project.jsave(summary, out / f"{site}_07_transfer.json"); project.record_run(site, "07_transfer", labelled=summary["labelled_points"], trees=len(bounds))
    log(f"{site}: {summary['labelled_points']:,} of {n_total:,} points labelled ({n_grown:,} grown, {n_clustered:,} as clusters); median fit {summary['median_fit_cm']:.1f} cm")


def review(site):
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
        xyz = c.xyz[m]; lb = lab[m]; _, first = np.unique(np.c_[np.floor(xyz / 0.1).astype(np.int64), lb], axis=0, return_index=True)
        for tid in np.unique(lb[first]): pts.setdefault(int(tid), []).append(xyz[first][lb[first] == tid])
    hulls = {}
    for tid, parts in pts.items():
        P = np.concatenate(parts); z0, z1 = P[:, 2].min(), P[:, 2].max(); crown = P[P[:, 2] >= z0 + 0.3 * (z1 - z0)]
        if len(crown) < 10: continue
        try: h = ConvexHull(crown[:, :2]); hulls[tid] = (MplPath(crown[h.vertices, :2]), crown[:, :2].min(axis=0), crown[:, :2].max(axis=0), z0 + 0.3 * (z1 - z0), z1)
        except Exception: continue
    inside = {t: 0 for t in hulls}; labelled = {t: 0 for t in hulls}; sample = {}    # pass 2: unlabelled canopy inside each envelope
    for t in tiles:
        c = io.read(str(t.path)); lab = c.attrs["treeid"].astype(np.int64); P = c.xyz; h = ground_height(P)
        for tid, cnt in zip(*np.unique(lab[lab > 0], return_counts=True)):
            if int(tid) in labelled: labelled[int(tid)] += int(cnt)
        Pu = P[(lab == 0) & (h > 1.0)]; b = t.bounds
        for tid, (path, lo, hi, zlo, zhi) in hulls.items():
            if hi[0] < b[0] or lo[0] > b[3] or hi[1] < b[1] or lo[1] > b[4]: continue
            m = (Pu[:, 0] >= lo[0]) & (Pu[:, 0] <= hi[0]) & (Pu[:, 1] >= lo[1]) & (Pu[:, 1] <= hi[1]) & (Pu[:, 2] >= zlo) & (Pu[:, 2] <= zhi)
            if not m.any(): continue
            k = path.contains_points(Pu[m][:, :2]); inside[tid] += int(k.sum())
            if k.any(): sample.setdefault(tid, []).append(Pu[m][k][::5])
    rows = sorted(((inside[t], labelled[t], t) for t in hulls), reverse=True)
    with open(d / f"{site}_07_review.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["treeid", "unlabelled_canopy_in_envelope", "labelled_points", "ratio", "crown_base_z", "top_z"])
        for u, l, t in rows: w.writerow([t, u, l, f"{u / max(l, 1):.3f}", f"{hulls[t][3]:.2f}", f"{hulls[t][4]:.2f}"])
    log(f"{site}: {len(hulls)} trees reviewed; over 20 % of their own points unlabelled in the envelope: {sum(1 for u, l, t in rows if u > 0.2 * max(l, 1))}")
    top = [t for u, l, t in rows[:3] if u > 0]
    if top:
        fig, axes = plt.subplots(2, len(top), figsize=(6 * len(top), 11), layout="constrained", squeeze=False)
        for k, tid in enumerate(top):
            L = np.concatenate(pts[tid]); U = np.concatenate(sample[tid])
            for row, (i, j, name) in enumerate(((0, 2, "side"), (0, 1, "plan"))):
                ax = axes[row, k]; ax.scatter(L[:, i], L[:, j], s=0.6, c="#009E73", rasterized=True, label=f"tree {tid}")
                ax.scatter(U[:, i], U[:, j], s=0.8, c="#D55E00", rasterized=True, label="unlabelled in its envelope"); ax.set_aspect("equal"); ax.set_title(f"tree {tid}: {name}", fontsize=10)
                if row == 0 and k == 0: ax.legend(markerscale=10, fontsize=8)
        fig.suptitle(f"{site}: trees with the most unlabelled canopy inside their crown envelope", fontsize=10)
        fig.savefig(d / f"{site}_07_review_top3.png", dpi=70); plt.close(fig)


def run(site, review_only=False, **kw):
    if not project.has_trees(site): log(f"{site}: no manual trees; nothing to transfer"); return
    if not review_only: transfer(site, **kw)
    review(site)


def done(site):
    return not project.has_trees(site) or ((ROOT / site / "07_trees" / f"{site}_07_transfer.json").exists() and (ROOT / site / "07_trees" / f"{site}_07_review.csv").exists())


if __name__ == "__main__":
    p = project.parser(__doc__); p.add_argument("--review-only", action="store_true"); p.add_argument("--no-clusters", action="store_true")
    for k, v in (("r-label", 0.05), ("r-grow", 0.04), ("r-near", 0.15)): p.add_argument(f"--{k}", type=float, default=v)
    p.add_argument("--grow-passes", type=int, default=2); a = p.parse_args()
    project.for_sites(run, a.sites, "07", review_only=a.review_only, r_label=a.r_label, r_grow=a.r_grow, r_near=a.r_near,
                   grow_passes=a.grow_passes, clusters=not a.no_clusters)
