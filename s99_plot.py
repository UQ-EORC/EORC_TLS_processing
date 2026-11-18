"""Plot box and plot-level numbers: the box from the field inventory, every product clipped to it, and stand
density and basal area from the segmented trees (manual or automatic, as edited in segfix).

usage: s99_plot.py [PLOT ...] [--bounds] [--no-laz]          (--bounds with no PLOT also draws the check figure)

--bounds writes the plot boxes (99_meta/<PLOT>_99_plot_bounds.json and .geojson, and <project>_plot_bounds_check.png):
  a square of side size_m ([plot_box] in project.toml; defaults in brackets), so every plot covers the same area (100 m:
  1 ha). At every rotation from -max_rotation_deg (10) to +max_rotation_deg in 1 degree steps, the best square is found in
  three steps: (1) inventory: keep the squares holding as many field inventory stems as any square can, less stem_slack
  (1 %) of them, so a small cluster of stems only has to sit inside the square and does not pull it onto itself (plots
  without an inventory skip this); (2) scan coverage: each cell scores 1 - exp(-n / 2), n the scan positions within
  cover_r_m (15) of it, so a cell seen from two positions counts almost as much as one seen from ten; keep the squares
  within cover_slack (0.5 %) of the best total, which follows the scans however irregular their layout and ignores stray
  positions; (3) the square whose centre is nearest the mean scan position. The rotation is the smallest within the
  stem slack of the most stems any rotation holds (no inventory: within the cover slack of the best coverage), so a
  square only turns when that clearly fits more. Searched on a 1 m grid, the chosen rotation solved again on 0.5 m.
  Plots whose scan positions form a regular grid can set scan_grid_rings = N: the grid's rotation and spacing are fitted
  to the positions, the outer N rings left out, and the square centred on the remaining positions with its sides along
  the grid. Scan positions within 1 m (upright and tilted scans on one tripod) count once. The inventory (a CSV with
  columns inventory_x, inventory_y) is in the frame of the matrices in inventory_frame (default: the merge's) and is
  moved into the merge's frame with the median change; stems more than isolated_stem_m (15) from any other are dropped.
Clip (default) to the box (cells and points inside it; rasters cut to the cells around it, those outside set to nodata),
next to the full-extent products, which are kept:
  04_registered/<PLOT>_04_merged_2cm_plot.laz   05_ground/<PLOT>_05_dtm_1m_plot.tif, _05_chm_0.5m_plot.tif
  06_stems/<PLOT>_06_stems_plot.csv            automatic stems inside the box (they over-count and inflate DBH)
  07_trees/<PLOT>_07_trees_vs_plot.csv         per tree (09_leafwood): centroid, share of points outside the box, status
  99_meta/<PLOT>_99_plot_clip.json             areas, CHM cover, counts, and stems/ha, basal area and median DBH from the
                                               trees whose centroid is inside (DBH from the sylva QSM, 08)
Voxel profiles are clipped by s10_canopy.py (--reprofile after the box changes).
"""
import csv, glob, json, shutil, time
from pathlib import Path
import numpy as np
import project
from project import ROOT, PLOT_BOX, log


def old_to_current(site):
    """4x4 from the inventory's frame (inventory_frame) to the merge's frame: the median change of the scan matrices.

    For every scan with a .DAT matrix in both folders, new @ inv(old) maps a point from the old project frame to the new
    one; the element-wise median over the scans is a robust estimate of the frame change (close to rigid when the scans
    agree). The identity when inventory_frame is not set or is the merge's own folder.
    """
    a = project.setting_path(site, "inventory_frame"); b = project.matrices_dir(site)
    if a is None or a == b: return np.eye(4)
    old = {f.stem: np.loadtxt(f) for f in a.glob("*.DAT")}; new = {f.stem: np.loadtxt(f) for f in b.glob("*.DAT")}
    return np.median(np.array([new[n] @ np.linalg.inv(old[n]) for n in old if n in new]), 0)


def inventory(site):
    """(n, 2) xy of the plot's field inventory stems in the merge's project frame, or None when the plot has no inventory.

    Reads the inventory CSV (setting inventory, columns inventory_x and inventory_y; rows with an empty or NA coordinate
    skipped) and moves it with old_to_current. With more than 20 stems, a stem whose nearest neighbour is further than
    PLOT_BOX["isolated_stem_m"] is dropped (a stray or mistyped coordinate would otherwise stretch the box).
    """
    path = project.setting_path(site, "inventory")
    if not path: return None
    kx, ky = project.setting(site, "inventory_x"), project.setting(site, "inventory_y")
    r = list(csv.DictReader(open(path, encoding="utf-8-sig")))       # utf-8-sig: spreadsheet exports often start with a BOM
    p = np.array([(float(x[kx]), float(x[ky])) for x in r if x.get(kx) not in (None, "", "NA") and x.get(ky) not in (None, "", "NA")])
    q = (np.c_[p, np.zeros(len(p)), np.ones(len(p))] @ old_to_current(site).T)[:, :2]   # homogeneous (x, y, 0, 1); z is not used
    if len(q) > 20:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(q).query(q, k=2); q = q[d[:, 1] <= PLOT_BOX["isolated_stem_m"]]   # k=2: column 0 is the stem itself
    return q


def scan_positions(site):
    """(n, 2) xy of the scan positions, project frame, one per tripod: positions within 1 m of one already kept are dropped."""
    keep = []
    for p in np.array([m[:2, 3] for m in project.matrices(site)[0].values()]):
        if not keep or np.min(np.hypot(*(np.array(keep) - p).T)) > 1.0: keep.append(p)
    return np.array(keep)


def _rot(xy, deg):
    """xy in a frame turned deg anticlockwise: (x cos + y sin, -x sin + y cos)."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.c_[xy[:, 0] * c + xy[:, 1] * s, -xy[:, 0] * s + xy[:, 1] * c]


def _box_sums(a, n):
    """Sum of every n x n window of a (whole windows only), from the integral image."""
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1)); s[1:, 1:] = a.cumsum(0).cumsum(1)
    return s[n:, n:] - s[:-n, n:] - s[n:, :-n] + s[:-n, :-n]


def square_corners(centre, deg, size):
    """The (4, 2) corners of a square of side size centred on centre, turned deg anticlockwise; anticlockwise from the
    corner at the square's own (0, 0)."""
    local = np.array([[0, 0], [size, 0], [size, size], [0, size]], float) - size / 2
    return _rot(local, -deg) + centre


class Coverage:
    """The number of scan positions within cover_r_m of every point of a fixed world grid (cell 0.25 m), sampled at any xy."""
    def __init__(self, scans, pts, size, r):
        from scipy.spatial import cKDTree
        self.x0, self.y0 = np.floor(pts.min(0) - size); x1, y1 = np.ceil(pts.max(0) + size); self.c = 0.25
        nx, ny = int((x1 - self.x0) / self.c), int((y1 - self.y0) / self.c)
        gx, gy = np.meshgrid(self.x0 + (np.arange(nx) + 0.5) * self.c, self.y0 + (np.arange(ny) + 0.5) * self.c, indexing="ij")
        self.n = np.array([len(v) for v in cKDTree(scans).query_ball_point(np.c_[gx.ravel(), gy.ravel()], r)]).reshape(nx, ny)

    def sample(self, xy):
        i = np.clip(((xy[:, 0] - self.x0) / self.c).astype(int), 0, self.n.shape[0] - 1)
        j = np.clip(((xy[:, 1] - self.y0) / self.c).astype(int), 0, self.n.shape[1] - 1)
        return self.n[i, j]


def best_at(deg, scans, stems, cov, cell, box):
    """The best square at one rotation (steps 1-3 of the module docstring). box: PLOT_BOX. Returns (the most stems a square
    holds, its coverage (m2, saturated), its centre in world xy)."""
    size = box["size_m"]; pts = scans if stems is None else np.vstack([scans, stems]); r = _rot(pts, deg)
    u0, v0 = np.floor(r.min(0) - size * 0.6); u1, v1 = np.ceil(r.max(0) + size * 0.6)
    nu, nv = int((u1 - u0) / cell), int((v1 - v0) / cell)
    gu, gv = np.meshgrid(u0 + (np.arange(nu) + 0.5) * cell, v0 + (np.arange(nv) + 0.5) * cell, indexing="ij")
    n = cov.sample(_rot(np.c_[gu.ravel(), gv.ravel()], -deg)).reshape(nu, nv); w = int(round(size / cell))
    cover = _box_sums(1 - np.exp(-n / 2.0), w) * cell * cell; ok = np.ones_like(cover, bool); most = 0
    if stems is not None:                                            # 1: as many stems as any square holds, less the slack
        rs = _rot(stems, deg); h, _, _ = np.histogram2d(rs[:, 0], rs[:, 1], bins=(nu, nv), range=((u0, u0 + nu * cell), (v0, v0 + nv * cell)))
        n_in = _box_sums(h, w); most = int(n_in.max()); ok &= n_in >= most - np.floor(box["stem_slack"] * len(stems))
    best = cover[ok].max(); ok &= cover >= (1 - box["cover_slack"]) * best   # 2: coverage within the slack of the best
    i, j = np.nonzero(ok); centres = np.c_[u0 + (i + w / 2) * cell, v0 + (j + w / 2) * cell]
    k = int(np.argmin(np.hypot(*(centres - _rot(scans.mean(0)[None], deg)[0]).T)))   # 3: nearest the mean scan position
    return most, float(best), _rot(centres[k:k + 1], -deg)[0]


def grid_square(scans, rings, box):
    """For scan positions on a regular grid: (rotation, centre, text) of the square along the grid, centred on the
    positions left after dropping the outer `rings` rings. The grid's rotation (0.05 degree steps) and spacing are those
    that put the positions nearest to grid nodes."""
    from scipy.spatial import cKDTree
    nn = np.median(cKDTree(scans).query(scans, k=2)[0][:, 1]); best = None; mr = box["max_rotation_deg"]
    for deg in np.arange(-mr, mr + 0.001, 0.05):
        r = _rot(scans, deg)
        for sp in np.arange(0.8 * nn, 1.6 * nn, 0.05):              # above half the grid step, which would fit too
            res = 0.0
            for k in (0, 1):                                         # mean distance to the nearest grid line, per axis
                q = r[:, k] / sp; off = np.angle(np.exp(2j * np.pi * q).mean()) / (2 * np.pi); res += np.abs((q - off + 0.5) % 1 - 0.5).mean() * sp
            if best is None or res < best[2]: best = (float(deg), float(sp), res / 2)
    deg, sp, res = best; r = _rot(scans, deg)
    iu = np.round((r[:, 0] - r[:, 0].min()) / sp).astype(int); iv = np.round((r[:, 1] - r[:, 1].min()) / sp).astype(int)
    inner = (iu >= rings) & (iu <= iu.max() - rings) & (iv >= rings) & (iv <= iv.max() - rings); ri = r[inner]
    how = (f"{box['size_m']:g} m square on the scan grid ({sp:.2f} m spacing, {deg:+.2f} degrees, mean residual {res:.2f} m), "
           f"outer {rings} ring(s) left out; the inner {int(inner.sum())} positions span {np.ptp(ri[:, 0]):.1f} x {np.ptp(ri[:, 1]):.1f} m")
    return deg, _rot(((ri.min(0) + ri.max(0)) / 2)[None], -deg)[0], how


def propose(site):
    """The plot box of a plot, by the rules of the module docstring.

    Returns (record for 99_meta/<PLOT>_99_plot_bounds.json, scan positions, inventory xy or None, Coverage), project frame.
    """
    box = PLOT_BOX; size = float(box["size_m"]); scans = scan_positions(site); stems = inventory(site)
    cov = Coverage(scans, scans if stems is None else np.vstack([scans, stems]), size, box["cover_r_m"])
    rings = project.setting(site, "scan_grid_rings")
    if rings: deg, centre, how = grid_square(scans, int(rings), box)
    else:
        mr = box["max_rotation_deg"]; trials = [(d, *best_at(d, scans, stems, cov, 1.0, box)[:2]) for d in np.arange(-mr, mr + 0.5, 1.0)]
        if stems is not None:                                        # the smallest rotation within the slack of the most stems
            most = max(t[1] for t in trials); trials = [t for t in trials if t[1] >= most - np.floor(box["stem_slack"] * len(stems))]
        else:                                                        # no inventory: within the slack of the best coverage
            top = max(t[2] for t in trials); trials = [t for t in trials if t[2] >= (1 - box["cover_slack"]) * top]
        deg = float(min(trials, key=lambda t: abs(t[0]))[0]); centre = best_at(deg, scans, stems, cov, 0.5, box)[2]
        how = (f"{size:g} m square: " + ("most field inventory stems, then " if stems is not None else "no field inventory; ")
               + f"best scan coverage, then nearest the mean scan position; turned {deg:+g} degrees")
    cs = square_corners(centre, deg, size)
    g = (np.arange(int(size / 0.5)) + 0.5) * 0.5 - size / 2; gu, gv = np.meshgrid(g, g, indexing="ij")
    seen = cov.sample(_rot(np.c_[gu.ravel(), gv.ravel()], -deg) + centre) >= 1
    rec = {"site": site, "definition": how, "centre_x": float(centre[0]), "centre_y": float(centre[1]), "angle_deg": deg, "size_m": size,
           "corners": [[round(float(x), 3), round(float(y), 3)] for x, y in cs],
           "xmin": float(cs[:, 0].min()), "ymin": float(cs[:, 1].min()), "xmax": float(cs[:, 0].max()), "ymax": float(cs[:, 1].max()),
           "area_ha": size * size / 1e4, "scan_positions": len(scans), "scan_positions_inside": int(project.in_box(cs, scans[:, 0], scans[:, 1]).sum()),
           "inventory_stems": None if stems is None else len(stems),
           "inventory_stems_inside": None if stems is None else int(project.in_box(cs, stems[:, 0], stems[:, 1]).sum()),
           f"share_within_{box['cover_r_m']:g}m_of_a_scan": round(float(seen.mean()), 4), "settings": dict(box), "frame": "project frame of stages 04-11"}
    return rec, scans, stems, cov


def write_bounds(sites, figure=True):
    """Write the plot box of every plot in sites (99_meta/<PLOT>_99_plot_bounds.json and .geojson, project frame).

    figure: also save <project root>/<project>_plot_bounds_check.png, one panel per plot with the scan coverage, the scan
    positions, the inventory stems and the box (drawn either way, saved only then).
    """
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(sites); cols = min(4, n); rows = -(-n // cols)            # up to 4 panels a row; rows rounded up
    fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.9 * rows), facecolor="#fcfcfb", squeeze=False)
    for ax in axs.ravel()[n:]: ax.set_visible(False)                  # hide the unused panels of the last row
    for ax, site in zip(axs.ravel(), sites):
        rec, scans, stems, cov = propose(site); ring = rec["corners"] + rec["corners"][:1]
        ext = (cov.x0, cov.x0 + cov.n.shape[0] * cov.c, cov.y0, cov.y0 + cov.n.shape[1] * cov.c)
        ax.imshow((1 - np.exp(-cov.n / 2.0)).T, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=2.5, interpolation="nearest")
        ax.scatter(scans[:, 0], scans[:, 1], s=9, marker="^", color="#1f1f1f", label="scan positions", zorder=3)
        if stems is not None: ax.scatter(stems[:, 0], stems[:, 1], s=3, color="#2a78d6", label="field inventory stems", zorder=2)
        ax.plot(*np.array(ring).T, color="#e34948", lw=2, label="plot box", zorder=4)
        pad = 15; ax.set_xlim(min(scans[:, 0].min(), rec["xmin"]) - pad, max(scans[:, 0].max(), rec["xmax"]) + pad)
        ax.set_ylim(min(scans[:, 1].min(), rec["ymin"]) - pad, max(scans[:, 1].max(), rec["ymax"]) + pad); ax.set_aspect("equal")
        inv = "no inventory" if stems is None else f"{rec['inventory_stems_inside']} of {rec['inventory_stems']} stems"
        ax.set_title(f"{site}: turned {rec['angle_deg']:+.1f} deg\n{inv}, {rec['scan_positions_inside']} of {rec['scan_positions']} positions", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        # the box as JSON (read by project.plot_box) and as a GeoJSON polygon (project frame, not a geographic CRS)
        meta = project.stage(site, "99_meta"); project.jsave(rec, meta / f"{site}_99_plot_bounds.json")
        (meta / f"{site}_99_plot_bounds.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"site": site, "frame": "project", "definition": rec["definition"]}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}))
        log(f"{site}: {rec['definition']}; {inv}, {rec['scan_positions_inside']} of {rec['scan_positions']} scan positions inside")
    if figure: axs[0, 0].legend(loc="lower left", fontsize=7, framealpha=0.9); fig.tight_layout(); fig.savefig(ROOT / f"{ROOT.name}_plot_bounds_check.png", dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def clip_raster(src, dst, corners):
    """Clip the GeoTIFF src to the box with these corners and write it to dst with the same profile.

    The window is the cells around the box (snapped to whole cells); cells whose centre is outside the box are set to
    nodata (NaN when src has none). Returns the clipped band as float with nodata as NaN (row 0 north, as rasterio reads it).
    """
    import rasterio
    from rasterio.windows import from_bounds
    c = np.asarray(corners, float); xy = (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max())
    with rasterio.open(src) as r:
        w = from_bounds(*xy, transform=r.transform).round_offsets().round_lengths(); a = r.read(1, window=w).astype(float); prof = r.profile.copy()
        tr = r.window_transform(w); nod = r.nodata
    if nod is not None: a[a == nod] = np.nan
    cx = tr.c + (np.arange(a.shape[1]) + 0.5) * tr.a; cy = tr.f + (np.arange(a.shape[0]) + 0.5) * tr.e   # cell centres
    X, Y = np.meshgrid(cx, cy); a[~project.in_box(c, X, Y)] = np.nan
    prof.update(width=a.shape[1], height=a.shape[0], transform=tr, dtype="float32", nodata=np.nan)
    with rasterio.open(dst, "w", **prof) as o: o.write(a.astype(np.float32), 1)
    return a


def manual_trees(site, area_ha):
    """Manual trees whose centroid is inside the box, with basal area from their sylva DBH (08). The automatic stems of 06
    over-count and inflate DBH (basal area 3-15x the field inventory); these agree with the inventory within about 20 %.

    Reads 08_qsm/<PLOT>_08_sylva_trees.csv and 07_trees/<PLOT>_07_trees_vs_plot.csv (written by clip just before). Trees
    without a sylva DBH count towards the density but not the basal area. area_ha: the box area. Returns the dict of
    manual_* values merged into the clip record (trees/ha, m2/ha, median DBH in cm).
    """
    q = {int(r["treeid"]): r["sylva_dbh_m"] for r in csv.DictReader(open(ROOT / site / "08_qsm" / f"{site}_08_sylva_trees.csv"))}; d = []   # tree id -> DBH (m) as text
    for r in csv.DictReader(open(ROOT / site / "07_trees" / f"{site}_07_trees_vs_plot.csv")):
        if project.in_plot(site, float(r["centroid_x"]), float(r["centroid_y"])):
            try: d.append(float(q.get(int(r["treeid"]), "nan")))
            except ValueError: d.append(np.nan)                          # empty DBH: no sylva QSM for the tree
    d = np.array(d); ok = d[np.isfinite(d)]
    return {"manual_trees_plot": len(d), "manual_trees_per_ha": len(d) / area_ha, "manual_trees_with_dbh": len(ok),
            "manual_basal_area_m2_ha": float(np.sum(np.pi * (ok / 2) ** 2) / area_ha), "manual_dbh_median_cm": float(100 * np.median(ok)) if len(ok) else None}


def clip(site, laz=True):
    """Clip the plot's products to its plot box (project.plot_bounds) and write the plot-level numbers.

    Writes the clipped DTM and CHM, the stage-06 stems inside the box, the per-tree position table of the manual trees
    (from the stage-09 tree files) and, with laz, the merged 2 cm cloud inside the box; the summary goes to
    99_meta/<PLOT>_99_plot_clip.json. The full-extent products are left as they are.
    """
    from sylva import io
    t0 = time.time(); r_ = ROOT / site; box = project.plot_box(site); inside = lambda x, y: project.in_box(box, x, y)
    rec = {"site": site, "plot_bounds": {"corners": box.round(3).tolist()}, "area_ha": project.box_area_ha(box)}

    # rasters: canopy cover is the share of CHM cells at or above 2 m inside the box
    dtm = clip_raster(r_ / "05_ground" / f"{site}_05_dtm_1m.tif", r_ / "05_ground" / f"{site}_05_dtm_1m_plot.tif", box)
    chm = clip_raster(r_ / "05_ground" / f"{site}_05_chm_0.5m.tif", r_ / "05_ground" / f"{site}_05_chm_0.5m_plot.tif", box)
    if np.isfinite(chm).any(): rec.update(chm_cover_2m_pct=float(100 * np.nanmean(chm >= 2.0)), chm_p95_m=float(np.nanpercentile(chm, 95)))
    rec["dtm_range_m"] = [float(np.nanmin(dtm)), float(np.nanmax(dtm))]

    # automatic stems (06) inside the box, with their density and basal area (kept for comparison; they over-count)
    rows = list(csv.DictReader(open(r_ / "06_stems" / f"{site}_06_stems.csv"))); keep = [r for r in rows if inside(float(r["x"]), float(r["y"]))]
    with open(r_ / "06_stems" / f"{site}_06_stems_plot.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(keep)
    dbh = np.array([float(r["dbh"]) for r in keep]); dbh = dbh[np.isfinite(dbh)]
    rec.update(stems_all=len(rows), stems_plot=len(keep), stems_per_ha=len(keep) / rec["area_ha"], basal_area_m2_ha=float(np.sum(np.pi * (dbh / 2) ** 2) / rec["area_ha"]),
               dbh_median_cm=float(100 * np.median(dbh)) if len(dbh) else None)

    # manual trees: centroid and the share of each tree's points outside the box, from its stage-09 file; status inside
    # (no point outside), outside (every point outside) or partly_outside. Density and basal area count by centroid.
    files = sorted(glob.glob(str(r_ / "09_leafwood" / f"{site}_09_tree_*.laz")))
    if files:
        out = []
        for p in files:
            c = io.read(p); x, y = c.xyz[:, 0], c.xyz[:, 1]; share = 1 - inside(x, y).mean()
            out.append({"treeid": int(Path(p).stem.split("_")[-1]), "centroid_x": round(float(x.mean()), 2), "centroid_y": round(float(y.mean()), 2), "n_points": len(x),
                        "share_outside": round(float(share), 4), "status": "inside" if share == 0 else ("outside" if share == 1 else "partly_outside")})
        out.sort(key=lambda d: d["treeid"])
        with open(r_ / "07_trees" / f"{site}_07_trees_vs_plot.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
        st = [d["status"] for d in out]; rec["trees"] = {k: st.count(k) for k in ("inside", "partly_outside", "outside")}
        rec.update(manual_trees(site, rec["area_ha"]))

    # merged cloud: each 2 cm tile clipped to the box into scratch, then the pieces joined into one LAZ
    if laz:
        tmp = project.SCRATCH / "clip" / site; tmp.mkdir(parents=True, exist_ok=True); parts = []; n_in = n_all = 0
        for t in project.merged_tiles(site):
            c = io.read(t); x, y = c.xyz[:, 0], c.xyz[:, 1]; n_all += len(x); m = inside(x, y)
            if m.any(): p = tmp / Path(t).name; io.write(c[m], p); parts.append(str(p)); n_in += int(m.sum())
        io.write_concat(parts, r_ / "04_registered" / f"{site}_04_merged_2cm_plot.laz"); shutil.rmtree(tmp)
        rec.update(merged_points_all=n_all, merged_points_plot=n_in)
    rec["seconds"] = round(time.time() - t0); project.jsave(rec, r_ / "99_meta" / f"{site}_99_plot_clip.json"); project.record_run(site, "99_plot_clip")
    log(f"{site}: {rec['area_ha']:.2f} ha, trees {rec.get('trees')}, basal area {rec.get('manual_basal_area_m2_ha', float('nan')):.1f} m2/ha")


def done(site):
    """Whether the clip is finished for the plot: its record exists, was made with the current plot box, and is no older
    than the stage-08 manifest (new tree models mean new plot totals)."""
    j = ROOT / site / "99_meta" / f"{site}_99_plot_clip.json"; m = ROOT / site / "08_qsm" / f"{site}_08_models_manifest.json"
    return (j.exists() and project.same_bounds(json.load(open(j)).get("plot_bounds"), site)       # made with the current plot box
            and (not m.exists() or j.stat().st_mtime >= m.stat().st_mtime))


if __name__ == "__main__":
    p = project.parser(__doc__); p.add_argument("--bounds", action="store_true"); p.add_argument("--no-laz", action="store_true"); a = p.parse_args()
    if a.bounds: write_bounds(a.sites or list(project.plots()), figure=not a.sites)
    else: project.for_sites(clip, a.sites, "99", laz=not a.no_laz)
