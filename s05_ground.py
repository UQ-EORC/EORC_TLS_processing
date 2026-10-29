"""Stages 05 (ground, DTM, CHM) and 06 (automatic stems), with QC figures (90) and a processing record (99).

usage: s05_ground.py [PLOT ...]

The 2 cm merge (04) is thinned to 5 cm (8 cm above 260 M points). Ground: the lowest points of each 2 m cell (within
-0.3..+0.4 m of the cell's 1st percentile, 10 cm thinned) through sylva.ground.classify_ground_pmf (2 m cells, 40 m
window); TIN DTM at 1 m with a support raster (plan distance to the nearest ground point); height above the DTM; CHM
at 0.5 m. Stems: sylva.trees.detect_stems on the 0.5-3 m slab, with per-scan stem noise (sylva.quality.stem_noise).
The automatic stems over-count and inflate DBH; plot basal area and density come from the manual trees (s99_plot.py).
  05_ground/<PLOT>_05_dtm_1m.tif, _dtm_support_1m.tif, _chm_0.5m.tif
  06_stems/<PLOT>_06_stems.csv
  90_qc/<PLOT>_90_qc_ground_dtm_chm.png, _ground_transect.png, _stems_slab.png, _qc_stem_noise.json
  99_meta/<PLOT>_99_processing.json        counts, checks and flags (status PASS or CHECK)
"""
import csv, json, time
import numpy as np
import project
from project import ROOT, log


def thinned_merge(site, voxel):
    """The stage-04 merge of a plot (project frame) read tile by tile and thinned to `voxel` m, as one PointCloud.

    Only xyz and a "scan_id" attribute (the merge's point_source_id, the scan position index) are kept, to hold memory
    down; scan_id is what the per-scan stem-noise check needs. Thinning per tile on the origin-anchored grid gives the
    same result as thinning the whole cloud, since the tile edges lie on voxel boundaries. Returns (cloud, number of tiles).
    """
    from sylva import io, filters, PointCloud
    tiles = project.merged_tiles(site); parts = []
    for t in tiles:
        c = io.read(t)
        parts.append(filters.voxel_downsample(PointCloud(c.xyz, {"scan_id": c.attrs["point_source_id"].astype(np.int32)}), voxel, origin=(0.0, 0.0, 0.0)))
    return PointCloud.concatenate(parts), len(tiles)


def lowest_points(cloud, cell=2.0):
    """Points near the bottom of each cell: within -0.3..+0.4 m of the cell's 1st percentile, thinned to 10 cm.

    A pre-selection of ground candidates for the morphological filter, which then needs to handle far fewer points and
    sees little vegetation. The 1st percentile rather than the minimum keeps a few low outliers (multipath, points below
    the surface) from setting the level. `cell` is the plan cell size in m.
    """
    from sylva import filters
    xyz = cloud.xyz; x0, y0 = float(xyz[:, 0].min()), float(xyz[:, 1].min())
    # One integer key per plan cell (column * 100000 + row; fine while a plot spans fewer than 100000 cells per axis).
    key = np.floor((xyz[:, 0] - x0) / cell).astype(np.int64) * 100000 + np.floor((xyz[:, 1] - y0) / cell).astype(np.int64)
    # Sort by cell, then by z within a cell: each cell is then a contiguous run, lowest point first.
    order = np.lexsort((xyz[:, 2], key)); k, zs = key[order], xyz[order, 2]
    starts = np.r_[0, np.flatnonzero(k[1:] != k[:-1]) + 1]; counts = np.diff(np.r_[starts, len(k)])   # each run's start and length
    # z of each cell's 1st-percentile point, repeated over the cell's points (still in sorted order).
    zlow = np.repeat(zs[starts + (0.01 * counts).astype(np.int64)], counts)
    return filters.voxel_downsample(cloud[order[(zs < zlow + 0.4) & (zs > zlow - 0.3)]], 0.1)


def run(site):
    """Stages 05 and 06 for one plot: DTM, support raster and CHM, automatic stems, QC figures and the processing record.

    Reads the stage-04 tiles and 04_registered/<PLOT>_04_merge.json; everything is in the project frame. Writes the files
    listed in the module docstring; 99_meta/<PLOT>_99_processing.json is written last (done() checks for it).
    """
    import sylva
    from sylva import filters, ground, quality, trees
    t0 = time.time(); g4, s5, qc, meta = (project.stage(site, d) for d in ("05_ground", "06_stems", "90_qc", "99_meta"))
    merge = json.load(open(ROOT / site / "04_registered" / f"{site}_04_merge.json"))
    # Coarser thinning for very dense plots, to stay within memory.
    voxel = 0.05 if merge["points_merged"] < 260_000_000 else 0.08
    cloud, n_tiles = thinned_merge(site, voxel)
    rec = {"site": site, "site_code": project.setting(site, "code"), "sylva_version": sylva.__version__, "run": time.strftime("%Y-%m-%d") + ", s05_ground.py",
           "source": f"04_registered/{site}_04_merged_2cm_tiles ({n_tiles} tiles, {merge['points_merged']:,} points), thinned to {voxel * 100:g} cm",
           "matrices": merge.get("matrices"), "scans_used": len(merge["scans_used"]), "frame": "project coordinates", "points_used": len(cloud), "voxel_m": voxel}
    log(f"{site}: {len(cloud):,} points at {voxel * 100:g} cm from {n_tiles} tiles")
    # Ground: the morphological filter on the low-point candidates; gp are the points it classes as ground (LAS class 2).
    # make_dtm triangulates the ground points of g (TIN) onto a 1 m grid.
    g = ground.classify_ground_pmf(lowest_points(cloud), cell_size=2.0, max_window=40.0); gp = g[g.attrs["classification"] == 2]
    dtm = ground.make_dtm(g, resolution=1.0, method="tin"); dtm.to_geotiff(g4 / f"{site}_05_dtm_1m.tif")
    # Support raster: for each DTM cell centre, the plan (2D) distance in m to the nearest ground point. z is set to 0 on
    # both sides so the 3D nearest-neighbour search measures plan distance. Large values mark interpolated terrain.
    cx, cy = dtm.cell_centers(); gxy = gp.xyz.copy(); gxy[:, 2] = 0.0                     # plan distance
    dist = filters.knn(gxy, np.column_stack([cx.ravel(), cy.ravel(), np.zeros(cx.size)]), 1)[0][:, 0]
    sup = sylva.Raster(dist.reshape(cx.shape).astype(np.float32), dtm.xmin, dtm.ymin, dtm.resolution); sup.to_geotiff(g4 / f"{site}_05_dtm_support_1m.tif")
    # Height above the DTM for every point (attribute "height"), then the CHM (highest point per 0.5 m cell).
    cloud = ground.normalize_height(cloud, dtm); h = cloud.attrs["height"]
    chm = ground.make_chm(cloud, resolution=0.5); chm.to_geotiff(g4 / f"{site}_05_chm_0.5m.tif")
    # Plausibility numbers for the record: the share of cells with ground within 3 m, extreme heights, points well
    # below the terrain (a DTM that is too high, or noise).
    rec.update(ground_points=len(gp), dtm_z_min=float(np.nanmin(dtm.data)), dtm_z_max=float(np.nanmax(dtm.data)),
               dtm_cells_with_ground_within_3m_pct=float(100 * (dist <= 3).mean()), height_p999_m=float(np.nanpercentile(h, 99.9)),
               height_max_m=float(np.nanmax(h)), points_below_dtm_pct=float(100 * (h < -0.3).mean()))
    log(f"{site}: DTM {rec['dtm_z_min']:.1f}..{rec['dtm_z_max']:.1f} m, ground within 3 m in {rec['dtm_cells_with_ground_within_3m_pct']:.0f}% of cells, "
        f"height p99.9 {rec['height_p999_m']:.1f} m, below DTM {rec['points_below_dtm_pct']:.1f}%")
    # Stage 06: automatic stems from the 0.5-3 m slab above the DTM (circle fits; dbh in m, x/y in the project frame).
    lowslice = cloud[(h > 0.5) & (h < 3.0)]; stems = trees.detect_stems(lowslice)
    fields = ["tree_id", "x", "y", "dbh", "height", "n_points", "rmse", "lean_deg", "quality"]
    with open(s5 / f"{site}_06_stems.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(fields)
        for s in stems: w.writerow([getattr(s, k) for k in fields])
    dbh = np.array([s.dbh for s in stems]) if stems else np.zeros(1)                    # a dummy 0 so median() works without stems
    rec.update(stems=len(stems), stems_dbh_ge_10cm=int((dbh >= 0.1).sum()), dbh_median_cm=float(100 * np.median(dbh)))
    log(f"{site}: {len(stems)} stems, DBH median {rec['dbh_median_cm']:.0f} cm")
    # Per-scan stem noise: how far each scan's stem points sit from the fitted stem circles, a check on registration and
    # ranging noise. Optional: a failure is recorded instead of stopping the stage.
    try: rec["scan_qc"] = quality.stem_noise(lowslice, scan_id=lowslice.attrs["scan_id"].astype(int), stems=stems).summary()
    except Exception as exc: rec["scan_qc"] = f"not computed: {exc}"
    (qc / f"{site}_90_qc_stem_noise.json").write_text(json.dumps(rec["scan_qc"], indent=2, default=float))
    make_figs(site, cloud, dtm, sup, chm, stems, qc)
    # Automatic flags; any flag sets the status to CHECK. PASS only means none fired: the figures still need a look.
    flags = []
    if rec["points_below_dtm_pct"] > 2: flags.append("many points below the DTM")
    if rec["height_p999_m"] > 80: flags.append("implausible canopy height")
    if rec["dtm_cells_with_ground_within_3m_pct"] < 20: flags.append("little observed ground")
    if not stems: flags.append("no stems detected")
    rec["flags"] = flags or ["none"]; rec["status"] = "CHECK" if flags else "PASS (automatic checks only; look at the 90_qc figures)"
    rec["seconds"] = round(time.time() - t0); project.jsave(rec, meta / f"{site}_99_processing.json")
    project.record_run(site, "05_ground", status=rec["status"], stems=len(stems)); log(f"{site}: {rec['status']} ({rec['seconds']} s)")


def make_figs(site, cloud, dtm, sup, chm, stems, qc):
    """The stage-05 QC figures, written to the 90_qc folder `qc`.

    cloud: the thinned plot cloud with a "height" attribute (project frame); dtm, sup, chm: sylva Rasters (row 0 south);
    stems: sylva.trees.Tree objects. Writes <PLOT>_90_qc_ground_dtm_chm.png (DTM, interpolated areas, CHM),
    _ground_transect.png (a side view of a strip with the DTM drawn on it) and, when stems were found,
    _stems_slab.png (stem circles over the breast-height slab).
    """
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # imshow extents (left, right, bottom, top) in project coordinates; origin="lower" below matches sylva's row 0 south.
    ext = (dtm.xmin, dtm.xmin + dtm.resolution * dtm.data.shape[1], dtm.ymin, dtm.ymin + dtm.resolution * dtm.data.shape[0])
    cext = (chm.xmin, chm.xmin + chm.resolution * chm.data.shape[1], chm.ymin, chm.ymin + chm.resolution * chm.data.shape[0])
    # Figure 1: DTM, support (only cells more than 3 m from any ground point, i.e. where the DTM is interpolated), CHM.
    fig, ax = plt.subplots(1, 3, figsize=(17, 6.2), layout="constrained")
    im = ax[0].imshow(dtm.data, extent=ext, origin="lower", cmap="terrain"); fig.colorbar(im, ax=ax[0], label="DTM z (m)", shrink=0.8); ax[0].set_title("DTM (tin, 1 m)")
    im = ax[1].imshow(np.where(sup.data <= 3, np.nan, sup.data), extent=ext, origin="lower", cmap="magma_r", vmax=15)
    fig.colorbar(im, ax=ax[1], label="m to nearest ground point (cells > 3 m only)", shrink=0.8); ax[1].set_title("Where the DTM is interpolated")
    im = ax[2].imshow(chm.data, extent=cext, origin="lower", cmap="viridis", vmax=np.nanpercentile(chm.data, 99.5))
    fig.colorbar(im, ax=ax[2], label="canopy height (m)", shrink=0.8); ax[2].set_title("CHM (0.5 m)")
    for a in ax: a.set_aspect("equal")
    fig.suptitle(site); fig.savefig(qc / f"{site}_90_qc_ground_dtm_chm.png", dpi=70); plt.close(fig)
    # Figure 2: side view (x against z) of a 4 m strip at the median y, every 5th point, with the DTM row under it.
    xyz = cloud.xyz; h = cloud.attrs["height"]
    y0 = float(np.median(xyz[:, 1])); m = np.abs(xyz[:, 1] - y0) < 2.0                    # a 4 m strip through the middle
    px, pz = xyz[m, 0][::5], xyz[m, 2][::5]; xs = np.arange(ext[0] + dtm.resolution / 2, ext[1], dtm.resolution)   # xs: DTM column centres
    row = min(max(int((y0 - dtm.ymin) / dtm.resolution), 0), dtm.data.shape[0] - 1)       # the DTM row holding y0, clamped to the grid
    fig, ax = plt.subplots(figsize=(16, 5.5), layout="constrained")
    ax.scatter(px, pz, s=0.15, c="0.45", rasterized=True); ax.plot(xs, dtm.data[row, :len(xs)], c="#D55E00", lw=2, label="DTM")
    ax.set_title(f"{site}: side view, strip y = {y0:.0f} m +- 2 m; the DTM should run along the tree bases"); ax.legend()
    # z range: from just below the terrain to just above the canopy, capped at 60 m above the lowest DTM cell.
    ax.set_aspect("equal"); ax.set_ylim(np.nanmin(dtm.data) - 3, min(np.nanpercentile(pz, 99.9) + 2, np.nanmin(dtm.data) + 60))
    fig.savefig(qc / f"{site}_90_qc_ground_transect.png", dpi=70); plt.close(fig)
    # Figure 3: plan view of the 1.2-1.4 m slab in a 40 m window around the median stem, with each stem drawn at its DBH.
    sx = np.array([[s.x, s.y, s.dbh] for s in stems]) if stems else np.zeros((0, 3))
    if len(sx):                                                                           # stems over the breast-height slab, 40 m window
        cx0, cy0 = np.median(sx[:, 0]), np.median(sx[:, 1])
        sel = (np.abs(xyz[:, 0] - cx0) < 20) & (np.abs(xyz[:, 1] - cy0) < 20) & (h > 1.2) & (h < 1.4)
        fig, ax = plt.subplots(figsize=(9, 9), layout="constrained"); ax.scatter(xyz[sel, 0], xyz[sel, 1], s=0.5, c="0.35", rasterized=True)
        inw = (np.abs(sx[:, 0] - cx0) < 20) & (np.abs(sx[:, 1] - cy0) < 20)              # stems inside the window
        for x, y, d in sx[inw]: ax.add_patch(plt.Circle((x, y), d / 2, fill=False, ec="#D55E00", lw=1))
        ax.set_aspect("equal"); ax.set_xlim(cx0 - 20, cx0 + 20); ax.set_ylim(cy0 - 20, cy0 + 20)
        ax.set_title(f"{site}: {int(inw.sum())} stems (circles at DBH) over the 1.2-1.4 m slab")
        fig.savefig(qc / f"{site}_90_qc_stems_slab.png", dpi=70); plt.close(fig)


def done(site):
    """Whether stages 05/06 have completed for the plot: the processing record and the stem table both exist."""
    return (ROOT / site / "99_meta" / f"{site}_99_processing.json").exists() and (ROOT / site / "06_stems" / f"{site}_06_stems.csv").exists()


if __name__ == "__main__":
    a = project.parser(__doc__).parse_args()
    project.for_sites(run, a.sites, "05")
