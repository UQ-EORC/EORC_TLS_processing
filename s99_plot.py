"""Plot box and plot-level numbers: the box from the field inventory, every product clipped to it, and stand
density and basal area from the manual trees.

usage: s99_plot.py [SITE ...] [--bounds] [--no-laz]          (--bounds with no SITE also draws the check figure)

--bounds writes the plot boxes (99_meta/<SITE>_99_plot_bounds.json and .geojson, and TERN_TLS_plot_bounds_check.png):
  where the registered field inventory spans 100 +- 4 m in x and y, a 100 m square centred on it; otherwise the
  inventory's extent rounded out to whole metres (isolated stems more than 15 m from any other dropped); plots without
  an inventory get a 100 m square centred on the scan positions. Inventories are in the TERN processing frame and are
  moved into the merge's frame with the median scan-matrix change (Litchfield).
Clip (default), next to the full-extent products, which are kept:
  04_registered/<SITE>_04_merged_2cm_plot.laz   05_ground/<SITE>_05_dtm_1m_plot.tif, _05_chm_0.5m_plot.tif
  06_stems/<SITE>_06_stems_plot.csv            automatic stems inside the box (they over-count and inflate DBH)
  07_trees/<SITE>_07_trees_vs_plot.csv         per manual tree (09_leafwood): centroid, share of points outside the box, status
  99_meta/<SITE>_99_plot_clip.json             areas, CHM cover, counts, and stems/ha, basal area and median DBH from the
                                               manual trees whose centroid is inside (DBH from the sylva QSM, 08)
Voxel profiles are clipped by s10_canopy.py (--reprofile after the box changes).
"""
import csv, glob, json, shutil, time
from pathlib import Path
import numpy as np
import tern
from tern import ROOT, TREES, log


def old_to_current(site):
    """4x4 from the TERN processing frame to the merge's frame (identity unless the coreg set is used)."""
    reg = ROOT / site / "03_registration"; cd = reg / f"{site}_03_coreg_dat"
    if not cd.exists(): return np.eye(4)
    old = {f.stem: np.loadtxt(f) for f in tern.trees_frame_dir(site).glob("*.DAT")}; new = {f.stem: np.loadtxt(f) for f in cd.glob("*.DAT")}
    return np.median(np.array([new[n] @ np.linalg.inv(old[n]) for n in old if n in new]), 0)


def inventory(site):
    rel = tern.setting(site, "inventory")
    if not rel: return None
    r = list(csv.DictReader(open(TREES / rel, encoding="utf-8-sig")))
    p = np.array([(float(x["local_x"]), float(x["local_y"])) for x in r if x.get("local_x") not in (None, "", "NA") and x.get("local_y") not in (None, "", "NA")])
    q = (np.c_[p, np.zeros(len(p)), np.ones(len(p))] @ old_to_current(site).T)[:, :2]
    if len(q) > 20:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(q).query(q, k=2); q = q[d[:, 1] <= 15.0]
    return q


def propose(site):
    o = np.array([m[:3, 3] for m in tern.matrices(site)[0].values()]); scan = (o[:, 0].min(), o[:, 1].min(), o[:, 0].max(), o[:, 1].max())
    p = inventory(site)
    if p is None:
        c = ((scan[0] + scan[2]) / 2, (scan[1] + scan[3]) / 2)
        return (c[0] - 50, c[1] - 50, c[0] + 50, c[1] + 50), scan, "100 m square centred on the scan positions (no field inventory)", p
    lo, hi = p.min(0), p.max(0)
    if np.all(np.abs(hi - lo - 100) <= 4): c = (lo + hi) / 2; b = (c[0] - 50, c[1] - 50, c[0] + 50, c[1] + 50); how = "100 m square on the field inventory"
    else: b = (np.floor(lo[0]), np.floor(lo[1]), np.ceil(hi[0]), np.ceil(hi[1])); how = f"extent of the field inventory ({len(p)} stems)"
    return tuple(float(v) for v in b), scan, how, p


def write_bounds(sites, figure=True):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(sites); cols = min(4, n); rows = -(-n // cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.7 * rows), facecolor="#fcfcfb", squeeze=False)
    for ax in axs.ravel()[n:]: ax.set_visible(False)
    for ax, site in zip(axs.ravel(), sites):
        b, scan, how, p = propose(site); x0, y0, x1, y1 = b
        st = list(csv.DictReader(open(ROOT / site / "06_stems" / f"{site}_06_stems.csv")))
        ax.scatter([float(r["x"]) for r in st], [float(r["y"]) for r in st], s=1, color="#b3b2ab", label="detected stems (06)")
        if p is not None: ax.scatter(p[:, 0], p[:, 1], s=4, color="#2a78d6", label="field inventory stems")
        sx0, sy0, sx1, sy1 = scan; ax.plot([sx0, sx1, sx1, sx0, sx0], [sy0, sy0, sy1, sy1, sy0], color="#52514e", lw=1, ls="--", label="scan positions")
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#e34948", lw=2, label="plot box")
        ax.set_aspect("equal"); ax.set_title(f"{site}: {x1 - x0:.0f} x {y1 - y0:.0f} m\n{how}", fontsize=10, loc="left"); ax.tick_params(labelsize=8)
        rec = {"site": site, "definition": how, "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1, "width_m": x1 - x0, "height_m": y1 - y0,
               "area_ha": (x1 - x0) * (y1 - y0) / 1e4, "frame": "project frame of stages 04-11"}
        meta = tern.stage(site, "99_meta"); tern.jsave(rec, meta / f"{site}_99_plot_bounds.json")
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        (meta / f"{site}_99_plot_bounds.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"site": site, "frame": "project", "definition": how}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}))
        log(f"{site}: {x1 - x0:.1f} x {y1 - y0:.1f} m, {how}")
    if figure: axs[0, 0].legend(loc="lower left", fontsize=8); fig.tight_layout(); fig.savefig(ROOT / "TERN_TLS_plot_bounds_check.png", dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def clip_raster(src, dst, xy):
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
    over-count and inflate DBH (basal area 3-15x the field inventory); these agree with the inventory within about 20 %."""
    x0, y0, x1, y1 = tern.plot_bounds(site)
    q = {int(r["treeid"]): r["sylva_dbh_m"] for r in csv.DictReader(open(ROOT / site / "08_qsm" / f"{site}_08_sylva_trees.csv"))}; d = []
    for r in csv.DictReader(open(ROOT / site / "07_trees" / f"{site}_07_trees_vs_plot.csv")):
        if x0 <= float(r["centroid_x"]) <= x1 and y0 <= float(r["centroid_y"]) <= y1:
            try: d.append(float(q.get(int(r["treeid"]), "nan")))
            except ValueError: d.append(np.nan)
    d = np.array(d); ok = d[np.isfinite(d)]
    return {"manual_trees_plot": len(d), "manual_trees_per_ha": len(d) / area_ha, "manual_trees_with_dbh": len(ok),
            "manual_basal_area_m2_ha": float(np.sum(np.pi * (ok / 2) ** 2) / area_ha), "manual_dbh_median_cm": float(100 * np.median(ok)) if len(ok) else None}


def clip(site, laz=True):
    from sylva import io
    t0 = time.time(); r_ = ROOT / site; xy = tern.plot_bounds(site); x0, y0, x1, y1 = xy
    rec = {"site": site, "plot_bounds": {"xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1}, "area_ha": (x1 - x0) * (y1 - y0) / 1e4}
    dtm = clip_raster(r_ / "05_ground" / f"{site}_05_dtm_1m.tif", r_ / "05_ground" / f"{site}_05_dtm_1m_plot.tif", xy)
    chm = clip_raster(r_ / "05_ground" / f"{site}_05_chm_0.5m.tif", r_ / "05_ground" / f"{site}_05_chm_0.5m_plot.tif", xy)
    if np.isfinite(chm).any(): rec.update(chm_cover_2m_pct=float(100 * np.nanmean(chm >= 2.0)), chm_p95_m=float(np.nanpercentile(chm, 95)))
    rec["dtm_range_m"] = [float(np.nanmin(dtm)), float(np.nanmax(dtm))]
    rows = list(csv.DictReader(open(r_ / "06_stems" / f"{site}_06_stems.csv"))); keep = [r for r in rows if x0 <= float(r["x"]) <= x1 and y0 <= float(r["y"]) <= y1]
    with open(r_ / "06_stems" / f"{site}_06_stems_plot.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(keep)
    dbh = np.array([float(r["dbh"]) for r in keep]); dbh = dbh[np.isfinite(dbh)]
    rec.update(stems_all=len(rows), stems_plot=len(keep), stems_per_ha=len(keep) / rec["area_ha"], basal_area_m2_ha=float(np.sum(np.pi * (dbh / 2) ** 2) / rec["area_ha"]),
               dbh_median_cm=float(100 * np.median(dbh)) if len(dbh) else None)
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
    if laz:
        tmp = tern.SCRATCH / "clip" / site; tmp.mkdir(parents=True, exist_ok=True); parts = []; n_in = n_all = 0
        for t in tern.merged_tiles(site):
            c = io.read(t); x, y = c.xyz[:, 0], c.xyz[:, 1]; n_all += len(x); m = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            if m.any(): p = tmp / Path(t).name; io.write(c[m], p); parts.append(str(p)); n_in += int(m.sum())
        io.write_concat(parts, r_ / "04_registered" / f"{site}_04_merged_2cm_plot.laz"); shutil.rmtree(tmp)
        rec.update(merged_points_all=n_all, merged_points_plot=n_in)
    rec["seconds"] = round(time.time() - t0); tern.jsave(rec, r_ / "99_meta" / f"{site}_99_plot_clip.json"); tern.record_run(site, "99_plot_clip")
    log(f"{site}: {rec['area_ha']:.2f} ha, trees {rec.get('trees')}, basal area {rec.get('manual_basal_area_m2_ha', float('nan')):.1f} m2/ha")


def done(site):
    j = ROOT / site / "99_meta" / f"{site}_99_plot_clip.json"; m = ROOT / site / "08_qsm" / f"{site}_08_models_manifest.json"
    return j.exists() and (not m.exists() or j.stat().st_mtime >= m.stat().st_mtime)


if __name__ == "__main__":
    p = tern.parser(__doc__); p.add_argument("--bounds", action="store_true"); p.add_argument("--no-laz", action="store_true"); a = p.parse_args()
    if a.bounds: write_bounds(a.sites or list(tern.SITES), figure=not a.sites)
    else: tern.for_sites(clip, a.sites, "99", laz=not a.no_laz)
