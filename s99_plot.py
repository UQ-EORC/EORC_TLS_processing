"""Plot box and plot-level numbers: the box from the field inventory, every product clipped to it, and stand
density and basal area from the segmented trees (manual or automatic, as edited in segfix).

usage: s99_plot.py [PLOT ...] [--bounds] [--no-laz]          (--bounds with no PLOT also draws the check figure)

--bounds writes the plot boxes (99_meta/<PLOT>_99_plot_bounds.json and .geojson, and <project>_plot_bounds_check.png),
  with the sizes of [plot_box] in project.toml (defaults in brackets): where the plot's field inventory spans size_m
  (100) +- tolerance_m (4) in x and y, a square of size_m centred on it; otherwise the inventory's extent rounded out to
  whole metres (stems more than isolated_stem_m (15) from any other dropped); plots without an inventory get a square of
  size_m centred on the scan positions. The inventory (a CSV with columns inventory_x, inventory_y) is in the frame of
  the matrices in inventory_frame (default: the merge's) and is moved into the merge's frame with the median change.
Clip (default), next to the full-extent products, which are kept:
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


def propose(site):
    """The plot box of a plot, by the rules of the module docstring.

    Returns (box (xmin, ymin, xmax, ymax), scan box (the extent of the scan positions), how the box was made (text),
    inventory xy or None), all in the project frame (m).
    """
    o = np.array([m[:3, 3] for m in project.matrices(site)[0].values()]); scan = (o[:, 0].min(), o[:, 1].min(), o[:, 0].max(), o[:, 1].max())
    p = inventory(site); size = PLOT_BOX["size_m"]; h = size / 2
    if p is None:
        c = ((scan[0] + scan[2]) / 2, (scan[1] + scan[3]) / 2)
        return (c[0] - h, c[1] - h, c[0] + h, c[1] + h), scan, f"{size:g} m square centred on the scan positions (no field inventory)", p
    lo, hi = p.min(0), p.max(0)
    # the stems span the nominal size in both x and y: the plot is that square (stems sit a little inside its edges);
    # otherwise the stems' own extent, rounded out to whole metres
    if np.all(np.abs(hi - lo - size) <= PLOT_BOX["tolerance_m"]): c = (lo + hi) / 2; b = (c[0] - h, c[1] - h, c[0] + h, c[1] + h); how = f"{size:g} m square on the field inventory"
    else: b = (np.floor(lo[0]), np.floor(lo[1]), np.ceil(hi[0]), np.ceil(hi[1])); how = f"extent of the field inventory ({len(p)} stems)"
    return tuple(float(v) for v in b), scan, how, p


def write_bounds(sites, figure=True):
    """Write the plot box of every plot in sites (99_meta/<PLOT>_99_plot_bounds.json and .geojson, project frame).

    figure: also save <project root>/<project>_plot_bounds_check.png, one panel per plot with the stage-06 stems, the
    inventory stems, the extent of the scan positions and the box (drawn either way, saved only then).
    """
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(sites); cols = min(4, n); rows = -(-n // cols)            # up to 4 panels a row; rows rounded up
    fig, axs = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.7 * rows), facecolor="#fcfcfb", squeeze=False)
    for ax in axs.ravel()[n:]: ax.set_visible(False)                  # hide the unused panels of the last row
    for ax, site in zip(axs.ravel(), sites):
        b, scan, how, p = propose(site); x0, y0, x1, y1 = b
        st = list(csv.DictReader(open(ROOT / site / "06_stems" / f"{site}_06_stems.csv")))
        ax.scatter([float(r["x"]) for r in st], [float(r["y"]) for r in st], s=1, color="#b3b2ab", label="detected stems (06)")
        if p is not None: ax.scatter(p[:, 0], p[:, 1], s=4, color="#2a78d6", label="field inventory stems")
        sx0, sy0, sx1, sy1 = scan; ax.plot([sx0, sx1, sx1, sx0, sx0], [sy0, sy0, sy1, sy1, sy0], color="#52514e", lw=1, ls="--", label="scan positions")
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#e34948", lw=2, label="plot box")
        ax.set_aspect("equal"); ax.set_title(f"{site}: {x1 - x0:.0f} x {y1 - y0:.0f} m\n{how}", fontsize=10, loc="left"); ax.tick_params(labelsize=8)
        # the box as JSON (read by project.plot_bounds) and as a GeoJSON polygon (project frame, not a geographic CRS)
        rec = {"site": site, "definition": how, "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1, "width_m": x1 - x0, "height_m": y1 - y0,
               "area_ha": (x1 - x0) * (y1 - y0) / 1e4, "frame": "project frame of stages 04-11"}
        meta = project.stage(site, "99_meta"); project.jsave(rec, meta / f"{site}_99_plot_bounds.json")
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        (meta / f"{site}_99_plot_bounds.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"site": site, "frame": "project", "definition": how}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}))
        log(f"{site}: {x1 - x0:.1f} x {y1 - y0:.1f} m, {how}")
    if figure: axs[0, 0].legend(loc="lower left", fontsize=8); fig.tight_layout(); fig.savefig(ROOT / f"{ROOT.name}_plot_bounds_check.png", dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def clip_raster(src, dst, xy):
    """Clip the GeoTIFF src to the box xy = (xmin, ymin, xmax, ymax) and write it to dst with the same profile.

    The window is snapped to whole cells, so the clip may reach up to half a cell beyond the box. Returns the clipped band
    as float with nodata as NaN (row 0 north, as rasterio reads it).
    """
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(src) as r:
        w = from_bounds(*xy, transform=r.transform).round_offsets().round_lengths(); a = r.read(1, window=w); prof = r.profile.copy()
        prof.update(width=a.shape[1], height=a.shape[0], transform=r.window_transform(w))
        with rasterio.open(dst, "w", **prof) as o: o.write(a, 1)
        nod = r.nodata
    a = a.astype(float)
    if nod is not None: a[a == nod] = np.nan
    return a


def manual_trees(site, area_ha):
    """Manual trees whose centroid is inside the box, with basal area from their sylva DBH (08). The automatic stems of 06
    over-count and inflate DBH (basal area 3-15x the field inventory); these agree with the inventory within about 20 %.

    Reads 08_qsm/<PLOT>_08_sylva_trees.csv and 07_trees/<PLOT>_07_trees_vs_plot.csv (written by clip just before). Trees
    without a sylva DBH count towards the density but not the basal area. area_ha: the box area. Returns the dict of
    manual_* values merged into the clip record (trees/ha, m2/ha, median DBH in cm).
    """
    x0, y0, x1, y1 = project.plot_bounds(site)
    q = {int(r["treeid"]): r["sylva_dbh_m"] for r in csv.DictReader(open(ROOT / site / "08_qsm" / f"{site}_08_sylva_trees.csv"))}; d = []   # tree id -> DBH (m) as text
    for r in csv.DictReader(open(ROOT / site / "07_trees" / f"{site}_07_trees_vs_plot.csv")):
        if x0 <= float(r["centroid_x"]) <= x1 and y0 <= float(r["centroid_y"]) <= y1:
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
    t0 = time.time(); r_ = ROOT / site; xy = project.plot_bounds(site); x0, y0, x1, y1 = xy
    rec = {"site": site, "plot_bounds": {"xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1}, "area_ha": (x1 - x0) * (y1 - y0) / 1e4}

    # rasters: canopy cover is the share of CHM cells at or above 2 m inside the box
    dtm = clip_raster(r_ / "05_ground" / f"{site}_05_dtm_1m.tif", r_ / "05_ground" / f"{site}_05_dtm_1m_plot.tif", xy)
    chm = clip_raster(r_ / "05_ground" / f"{site}_05_chm_0.5m.tif", r_ / "05_ground" / f"{site}_05_chm_0.5m_plot.tif", xy)
    if np.isfinite(chm).any(): rec.update(chm_cover_2m_pct=float(100 * np.nanmean(chm >= 2.0)), chm_p95_m=float(np.nanpercentile(chm, 95)))
    rec["dtm_range_m"] = [float(np.nanmin(dtm)), float(np.nanmax(dtm))]

    # automatic stems (06) inside the box, with their density and basal area (kept for comparison; they over-count)
    rows = list(csv.DictReader(open(r_ / "06_stems" / f"{site}_06_stems.csv"))); keep = [r for r in rows if x0 <= float(r["x"]) <= x1 and y0 <= float(r["y"]) <= y1]
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
            c = io.read(p); x, y = c.xyz[:, 0], c.xyz[:, 1]; share = 1 - ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)).mean()
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
            c = io.read(t); x, y = c.xyz[:, 0], c.xyz[:, 1]; n_all += len(x); m = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
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
