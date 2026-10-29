"""Stages 10 (ray-traced voxels) and 11 (canopy gap profiles) per plot, from the unfiltered pulses of 02_shots.

usage: s10_canopy.py [PLOT ...] [--reprofile] [--voxel=0.5] [--buffer=20] [--workers=8] [--keep-shots]

Each scan's pulses go into the project frame with the merge's matrices (project.matrices), so voxels and canopy share the
frame of stages 04-09. Scans with a matrix but no pulses, or the reverse, are left out and listed in the manifest.
  10_voxels/<PLOT>_10_voxels_<v>m/             sylva.voxels.ray_voxelize, blocked (sylva.voxels.blocks.open_blocked):
                                               every pulse, DTM ground, occlusion, FPL attenuation, spherical G
  10_voxels/<PLOT>_10_voxel_profile_<v>m.csv   by height above the DTM, voxels inside the plot box: observed / occluded
                                               share, mean PAD of voxels crossed by at least MIN_BEAMS beams, cumulative PAI
  11_canopy/<PLOT>_11_gap_profile.json          sylva.canopy.GapProfile over the upright scans (Jupp et al. 2009): PAI
                                               hinge, linear, weighted, clumping, canopy height, cover, saturation flag
  11_canopy/<PLOT>_11_pavd_profile.csv          gap-profile PAI and PAVD by height
  11_canopy/<PLOT>_11_canopy_profiles.png       gap-profile and voxel PAVD side by side
  10_voxels/<PLOT>_10_11_manifest.json          inputs, scans, settings, timings, results; written last (resumable per plot)
The grid covers the scanner positions plus --buffer m, from the lowest DTM cell to 2 m above the tallest CHM cell.
The gap-profile PAI is not meaningful for low, sparse canopy (scanners standing in the gaps) and saturates in dense
canopy: use the voxel PAI there. --reprofile redoes the voxel profile of a finished plot (e.g. after the plot box changed).
"""
import json, os, shutil, subprocess, time, traceback
from pathlib import Path
import numpy as np
import project
from project import ROOT, log
from s02_shots import shots_file

MIN_BEAMS = 5


def run(site, voxel=0.5, buffer=20.0, workers=8, keep_shots=False):
    from sylva import Shots, canopy, voxels
    t0 = time.time(); out10 = project.stage(site, "10_voxels"); out11 = project.stage(site, "11_canopy")
    mf = out10 / f"{site}_10_11_manifest.json"
    M, msrc = project.matrices(site)
    have = {n: shots_file(site, n) for n in M if shots_file(site, n).exists()}
    allshots = {p.name[len(site) + 4:-len("_shots.parquet")] for p in (ROOT / site / "02_shots").glob(f"{site}_02_*_shots.parquet")}
    names = sorted(have)
    if not names: raise RuntimeError(f"{site}: no scan has both a matrix and a shots file")
    dtm = project.load_dtm(site).fill_nearest(); chm = project.load_raster(ROOT / site / "05_ground" / f"{site}_05_chm_0.5m.tif").data
    tmp = project.SCRATCH / "voxels" / site; tmp.mkdir(parents=True, exist_ok=True)
    prof = canopy.GapProfile.empty(); upright = []; parts = []; n_shots = n_echoes = 0
    for i, n in enumerate(names):
        part = tmp / f"{i:04d}.parquet"; parts.append(str(part))
        s = Shots.load(have[n]); s = Shots(s.origin, s.direction, s.echo_start, s.echo_count, s.echo_range).transform(np.asarray(M[n]))
        n_shots += s.n_shots; n_echoes += s.n_echoes
        if "tilt" not in n.lower():                                  # the gap profile assumes an upright scanner
            xyz = s.echo_xyz(); gx = np.clip(xyz[:, 0], dtm.xmin, dtm.xmax - 1e-6); gy = np.clip(xyz[:, 1], dtm.ymin, dtm.ymax - 1e-6)
            prof.add_scan(s, xyz[:, 2] - dtm.sample(gx, gy)); upright.append(n)
        if not part.exists(): s.save(part)
        if (i + 1) % 10 == 0: log(f"{site}: {i + 1}/{len(names)} scans in the project frame")
        del s
    t1 = time.time()

    rep = prof.report()                                              # 11: gap profile
    scal = {k: (v.item() if hasattr(v, "item") else v) for k, v in rep.items() if np.ndim(v) == 0}
    scal["canopy_cover_chm_2m"] = float(canopy.canopy_cover(chm, 2.0)) if np.isfinite(chm).any() else None
    project.jsave({"site": site, "upright_scans": upright, **scal}, out11 / f"{site}_11_gap_profile.json")
    cols = ["height", "pai_hinge_profile", "pavd_hinge", "pai_linear_profile", "pavd_linear"]
    np.savetxt(out11 / f"{site}_11_pavd_profile.csv", np.column_stack([np.asarray(rep[c], float) for c in cols]), delimiter=",", header=",".join(cols), comments="", fmt="%.6g")
    log(f"{site}: gap profile from {len(upright)} upright scans: PAI hinge {scal['pai_hinge']:.2f}, linear {scal['pai_linear']:.2f}, saturated {scal['saturated']}")

    plot = tmp / f"{site}_plot_shots.parquet"                        # 10: one plot shots file, traced block by block
    if not plot.exists():
        subprocess.run([project.PYARROW_PYTHON, str(project.CODE / "merge_shots.py"), str(plot) + ".part", *parts], check=True); os.replace(str(plot) + ".part", plot)
    org = np.array([np.asarray(M[n])[:3, 3] for n in names])
    lo = [org[:, 0].min() - buffer, org[:, 1].min() - buffer, float(np.nanmin(dtm.data)) - 1.0]
    hi = [org[:, 0].max() + buffer, org[:, 1].max() + buffer, float(np.nanmax(dtm.data)) + float(np.nanmax(chm)) + 2.0]
    vdir = out10 / f"{site}_10_voxels_{voxel:g}m"
    if vdir.exists(): shutil.rmtree(vdir)
    log(f"{site}: tracing {n_shots:,} pulses into {voxel} m voxels over {hi[0] - lo[0]:.0f} x {hi[1] - lo[1]:.0f} x {hi[2] - lo[2]:.0f} m")
    g = voxels.ray_voxelize(str(plot), voxel, (lo, hi), dtm=dtm, ground_distance=0.2, occlusion=True, attenuation=["fpl"], block_size=64, out=vdir, max_memory=20, workers=workers)
    t2 = time.time(); xy = project.plot_bounds(site)
    pai_vox = write_profile(site, voxel, voxel_profile(g, voxel, hi[2] - lo[2], xy), rep, scal)
    rec = {"site": site, "run": f"{time.strftime('%Y-%m-%d')} from 02_shots", "shots": "02_shots", "matrices": msrc, "scans": names, "upright_scans": upright,
           "matrices_without_shots": sorted(set(M) - set(have)), "shots_without_matrix": sorted(allshots - set(M)), "pulses": n_shots, "echoes": n_echoes,
           "voxel": {"size_m": voxel, "bounds": [lo, hi], "dtm": "05_ground dtm_1m, nearest-filled", "ground_distance_m": 0.2, "occlusion": True, "attenuation": "fpl",
                     "lad": "spherical", "beam": None, "min_beams_profile": MIN_BEAMS, "block_stats": dict(g.block_stats), "pai_voxel": pai_vox,
                     "profile_region": {"plot_bounds": xy, "note": "voxel profile and PAI over voxels inside the plot box"}},
           "gap_profile": scal, "seconds": {"frame": round(t1 - t0), "trace": round(t2 - t1), "total": round(time.time() - t0)}}
    project.jsave(rec, mf)
    if not keep_shots: shutil.rmtree(tmp)
    project.record_run(site, "10_canopy", pai_voxel=pai_vox, pai_hinge=scal["pai_hinge"]); log(f"{site}: done in {rec['seconds']['total'] / 60:.0f} min")


def voxel_profile(g, voxel, z_extent, xy):
    """Profile by height above the DTM over the voxels whose centre lies in the plot box xy = (xmin, ymin, xmax, ymax)."""
    nb = int(np.ceil(z_extent / voxel)) + 1; edges = np.arange(nb + 1) * voxel
    cnt, obs, occ, pad_sum, pad_n = (np.zeros(nb) for _ in range(5))
    for _, b in g.blocks():
        X, Y, _ = b.centers(); h = b["distance_from_ground"].ravel()
        ok = np.isfinite(h) & (h >= 0) & (X.ravel() >= xy[0]) & (X.ravel() <= xy[2]) & (Y.ravel() >= xy[1]) & (Y.ravel() <= xy[3])
        if not ok.any(): continue
        k = np.minimum((h[ok] / voxel).astype(int), nb - 1)
        st = b["state"].ravel()[ok]; pad = b["pad_g_corrected"].ravel()[ok]; beams = b["num_beams"].ravel()[ok]
        cnt += np.bincount(k, minlength=nb); obs += np.bincount(k, weights=(st >= 2), minlength=nb); occ += np.bincount(k, weights=(st == 1), minlength=nb)
        use = (beams >= MIN_BEAMS) & np.isfinite(pad); pad_sum += np.bincount(k[use], weights=pad[use], minlength=nb); pad_n += np.bincount(k[use], minlength=nb)
    return dict(edges=edges, cnt=cnt, obs=obs, occ=occ, pad_n=pad_n, pavd=np.where(pad_n > 0, pad_sum / np.maximum(pad_n, 1), np.nan))


def write_profile(site, voxel, pr, rep, scal):
    """Voxel profile CSV and the comparison figure; returns the voxel PAI."""
    edges, cnt, obs, occ, pad_n, pavd = (pr[k] for k in ("edges", "cnt", "obs", "occ", "pad_n", "pavd"))
    keep = cnt > 0; out10 = ROOT / site / "10_voxels"; out11 = ROOT / site / "11_canopy"
    np.savetxt(out10 / f"{site}_10_voxel_profile_{voxel:g}m.csv",
               np.column_stack([edges[:-1], edges[1:], cnt, obs / np.maximum(cnt, 1), occ / np.maximum(cnt, 1), pad_n, pavd, np.nancumsum(pavd) * voxel])[keep],
               delimiter=",", header="height_from_m,height_to_m,voxels,observed_share,occluded_share,voxels_with_pad,pavd_m2_m3,pai_cumulative", comments="", fmt="%.6g")
    pai_vox = float(np.nansum(pavd) * voxel); log(f"{site}: voxel PAI {pai_vox:.2f} inside the plot box")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        zc = (edges[:-1] + voxel / 2)[keep]; fig, ax = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
        ax[0].plot(rep["pavd_hinge"], rep["height"], label="gap profile, hinge"); ax[0].plot(rep["pavd_linear"], rep["height"], label="gap profile, linear")
        ax[0].plot(pavd[keep], zc, label=f"voxels {voxel:g} m, FPL, plot box"); ax[0].set_xlabel("PAVD (m2/m3)"); ax[0].set_ylabel("height above ground (m)"); ax[0].legend()
        ax[1].plot(obs[keep] / cnt[keep], zc, label="observed"); ax[1].plot(occ[keep] / cnt[keep], zc, label="occluded"); ax[1].set_xlabel("share of voxels in the plot box"); ax[1].legend()
        filled = np.r_[np.asarray(rep["height"])[np.asarray(rep["pavd_hinge"]) > 1e-3], (edges[:-1] + voxel)[keep & (np.nan_to_num(pavd) > 1e-3)]]
        ax[0].set_ylim(0, (filled.max() if len(filled) else 10.0) + 5.0)
        fig.suptitle(f"{site}: PAI hinge {scal['pai_hinge']:.2f}, linear {scal['pai_linear']:.2f}, voxel {pai_vox:.2f}"); fig.tight_layout()
        fig.savefig(out11 / f"{site}_11_canopy_profiles.png", dpi=120); plt.close(fig)
    except Exception: traceback.print_exc()
    return pai_vox


def reprofile(site, voxel=0.5, **_):
    from sylva.voxels.blocks import open_blocked
    mf = ROOT / site / "10_voxels" / f"{site}_10_11_manifest.json"; rec = json.load(open(mf)); lo, hi = rec["voxel"]["bounds"]
    a = np.genfromtxt(ROOT / site / "11_canopy" / f"{site}_11_pavd_profile.csv", delimiter=",", names=True); rep = {k: a[k] for k in a.dtype.names}
    g = open_blocked(ROOT / site / "10_voxels" / f"{site}_10_voxels_{voxel:g}m"); xy = project.plot_bounds(site)
    rec["voxel"]["pai_voxel"] = write_profile(site, voxel, voxel_profile(g, voxel, hi[2] - lo[2], xy), rep, rec["gap_profile"])
    rec["voxel"]["profile_region"] = {"plot_bounds": xy, "note": "voxel profile and PAI over voxels inside the plot box"}; project.jsave(rec, mf)


def done(site): return (ROOT / site / "10_voxels" / f"{site}_10_11_manifest.json").exists()


if __name__ == "__main__":
    p = project.parser(__doc__); p.add_argument("--reprofile", action="store_true"); p.add_argument("--voxel", type=float, default=0.5)
    p.add_argument("--buffer", type=float, default=20.0); p.add_argument("--workers", type=int, default=8); p.add_argument("--keep-shots", action="store_true"); a = p.parse_args()
    if a.reprofile: project.for_sites(reprofile, a.sites, "10", voxel=a.voxel)
    else: project.for_sites(run, a.sites, "10", voxel=a.voxel, buffer=a.buffer, workers=a.workers, keep_shots=a.keep_shots)
