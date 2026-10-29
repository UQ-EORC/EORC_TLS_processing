"""Stage 04: one registered, 2 cm voxel-thinned cloud per plot from the stage-01 scans and the stage-03 matrices.

usage: s04_merge.py [PLOT ...] [--voxel=0.02] [--tile=50] [--workers=3]

Each scan goes into the project frame (project.matrices: the coreg set where present, else 03_transforms), is rounded to
the millimetre the file stores and voxel-thinned on a grid anchored at the project origin; the thinned scans are retiled
(50 m) and each tile thinned again so one point per voxel survives across scans (the first in file order). The tiles are
streamed into one LAZ (sylva.io.write_concat), so plot size is not limited by memory.
  04_registered/<PLOT>_04_merged_2cm_tiles/*.laz, <PLOT>_04_merged_2cm.laz, <PLOT>_04_merge.json
point_source_id is the scan position's index (as in 01_scans).
"""
import glob, shutil, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import project
from project import ROOT, log


def thin_one(args):
    site, name, M, voxel, out = args
    from sylva import io, filters, PointCloud
    c = io.read(str(project.scan_laz(site, name))); n = len(c); c = c.transform(np.asarray(M))
    c = filters.voxel_downsample(PointCloud(np.round(c.xyz, 3), c.attrs), voxel, origin=(0.0, 0.0, 0.0))
    io.write(c, out); return name, n, len(c)


def run(site, voxel=0.02, tile=50.0, workers=3):
    from sylva import io, filters, PointCloud
    from sylva.als import catalogue, ops
    t0 = time.time(); M, src = project.matrices(site); out = project.stage(site, "04_registered")
    names = sorted(n for n in M if project.scan_laz(site, n).exists()); missing = sorted(n for n in M if n not in names)
    sc = project.SCRATCH / "merge" / site; shutil.rmtree(sc, ignore_errors=True); (sc / "thin").mkdir(parents=True); (sc / "tiles").mkdir()
    log(f"{site}: {len(names)} scans with a matrix and a stage-01 file ({len(missing)} matrices without a scan file); matrices: {src}")
    counts = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, (name, n, m) in enumerate(pool.map(thin_one, [(site, n, M[n].tolist(), voxel, str(sc / "thin" / f"{n}.laz")) for n in names]), 1):
            counts[name] = (n, m)
            if k % 10 == 0 or k == len(names): log(f"{site}: thinned {k}/{len(names)} scans")
    cat = ops.retile(catalogue.catalog(sorted(glob.glob(str(sc / "thin" / "*.laz")))), str(sc / "tiles"), tile, workers=2, format="laz")
    tag = f"{voxel * 100:g}cm"; tdir = out / f"{site}_04_merged_{tag}_tiles"; shutil.rmtree(tdir, ignore_errors=True); tdir.mkdir()
    n_out = 0; tiles = []
    for t in sorted(cat.tiles, key=lambda t: t.path):
        c = io.read(str(t.path)); c = filters.voxel_downsample(PointCloud(np.round(c.xyz, 3), c.attrs), voxel, origin=(0.0, 0.0, 0.0))
        p = tdir / Path(t.path).name; io.write(c, str(p)); n_out += len(c); tiles.append(str(p))
    single = out / f"{site}_04_merged_{tag}.laz"; io.write_concat(tiles, str(single))
    log(f"{site}: merged {n_out:,} points at {tag} in {len(tiles)} tiles, {single.stat().st_size / 1e9:.2f} GB ({time.time() - t0:.0f} s)")
    rec = {"site": site, "voxel_m": voxel, "method": "first point per voxel, grid anchored at the project origin; per scan, then per tile across scans",
           "tile_m": tile, "matrices": src, "scans_used": names, "matrices_without_scan_file": missing,
           "points_in_scans": sum(v[0] for v in counts.values()), "points_after_per_scan_thinning": sum(v[1] for v in counts.values()),
           "points_merged": n_out, "tiles": len(tiles), "single_file": single.name, "point_source_id": "index of the scan position (as in 01_scans)",
           "per_scan": {k: {"in": v[0], "thinned": v[1]} for k, v in counts.items()}, "seconds": round(time.time() - t0)}
    project.jsave(rec, out / f"{site}_04_merge.json"); shutil.rmtree(sc, ignore_errors=True)
    project.record_run(site, "04_merge", points=n_out, scans=len(names))


def done(site): return (ROOT / site / "04_registered" / f"{site}_04_merge.json").exists()


if __name__ == "__main__":
    p = project.parser(__doc__); p.add_argument("--voxel", type=float, default=0.02); p.add_argument("--tile", type=float, default=50.0)
    p.add_argument("--workers", type=int, default=3); a = p.parse_args()
    project.for_sites(run, a.sites, "04", voxel=a.voxel, tile=a.tile, workers=a.workers)
