"""Stage 02 pulses: every pulse of every registered scan, unfiltered, for the canopy stages (09 voxels, 10 gap profiles).

usage: s02_shots.py [PLOT ...] [--workers=4] [--dry-run]

The stage-01 scans hold echoes only, so the pulses are read from the raw .rxp of the RiSCAN project in 00_raw
(<project>/SCANS/<position>/SINGLESCANS/<scan>.rxp), with missed pulses rebuilt from the scan pattern in project.rsp
(sylva.io.read_rxp_shots, fill_missing). No point filter: every echo
is kept except those closer than 0.5 m (the scanner head). A point filter (reflectance floor, deviation cap) removes
every pulse whose echoes all fail it, which overstates gaps and understates plant area.
Each scan of 03_transforms.json is matched to its .rxp by position and by the time stamp in its source file name
(<scan>.rdbx, or the export's file name).

  02_shots/<PLOT>_02_<scan>_shots.parquet   scanner frame, all echoes >= 0.5 m, misses included;
                                            echo attributes reflectance, amplitude, deviation
  02_shots/<PLOT>_02_shots_manifest.json    per-scan counts. Resumable. A dense scan can take over 12 GB per worker.
"""
import json, os, re, time, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import project
from project import ROOT, log

MIN_RANGE = 0.5
STAMP = re.compile(r"\d{6}_\d{6}")


def shots_file(site, name): return ROOT / site / "02_shots" / f"{site}_02_{name}_shots.parquet"


def jobs_for(site):
    """(site, name, rxp path, theta limits) per registered scan of 03_transforms.json, and the scans that could not be matched."""
    proj = project.riscan_project(site)
    if proj is None: raise FileNotFoundError(f"{site}: no RiSCAN project folder in 00_raw (unzip it first)")
    pos = {p.name: p for p in proj.with_scans(require_sop=False) if p.rxp is not None and p.pattern is not None}
    tj = json.load(open(ROOT / site / "03_registration" / f"{site}_03_transforms.json"))
    jobs, bad = [], []
    for x in tj["scans"]:
        if not x.get("registered", True): continue
        name = x["name"]; p = pos.get(name.split("#")[0])
        if p is None: bad.append((name, "no position with an rxp")); continue
        m = STAMP.search(Path(x.get("source", "")).name)
        f = next((s for s in p.scans if m and Path(s.name).stem == m.group(0)), None) if m else None
        if f is None and len(p.scans) == 1 and "#" not in name: f = p.scans[0]
        if f is None: bad.append((name, f"no rxp for {Path(x.get('source', '')).name} among {[s.name for s in p.scans]}")); continue
        lim = (p.pattern["theta_start"], p.pattern["theta_start"] + p.pattern["theta_delta"] * p.pattern["theta_count"])
        jobs.append((site, name, str(f), lim))
    return jobs, bad


def convert_one(args):
    site, name, rxp, lim = args
    from sylva import io, Shots
    pq = shots_file(site, name); tmp = pq.with_name(pq.stem + ".part.parquet")
    try:
        t0 = time.time()
        shots, rep = io.read_rxp_shots(rxp, fill_missing=lim, return_report=True, min_range=0.0)
        n_pulses, n_echoes, n_miss = shots.n_shots, shots.n_echoes, int((shots.echo_count == 0).sum())
        keep = shots.echo_range >= MIN_RANGE
        so = shots.shot_of_echo(); cnt = np.bincount(so[keep], minlength=n_pulses); del so
        drop = (shots.echo_count > 0) & (cnt == 0); ks = ~drop                      # pulses whose only echoes were at the scanner head
        count = cnt[ks]; start = np.r_[0, np.cumsum(count)[:-1]]
        ea = {k: shots.echo_attrs[k][keep] for k in ("reflectance", "amplitude", "deviation") if k in shots.echo_attrs}
        f = Shots(origin=np.broadcast_to(np.zeros(3), (int(ks.sum()), 3)), direction=shots.direction[ks], echo_start=start,
                  echo_count=count, echo_range=shots.echo_range[keep], echo_attrs=ea)
        del shots
        f.save(tmp); os.replace(tmp, pq)
        return name, {"file": Path(rxp).name, "pulses": int(n_pulses), "echoes": int(n_echoes), "misses_reconstructed": n_miss,
                      "misses_gap_before_after": [int(rep["n_gap"]), int(rep["n_before"]), int(rep["n_after"])],
                      "echoes_dropped_min_range": int((~keep).sum()), "pulses_dropped_min_range": int(drop.sum()),
                      "echoes_written": int(f.n_echoes), "pulses_written": int(f.n_shots),
                      "parquet_mb": round(pq.stat().st_size / 1e6, 1), "seconds": round(time.time() - t0)}
    except Exception:
        if tmp.exists(): tmp.unlink()
        return name, {"error": traceback.format_exc().strip().splitlines()[-1]}


def run(site, workers=4, dry=False):
    jobs, bad = jobs_for(site)
    log(f"{site}: {len(jobs)} scans matched to an rxp, {len(bad)} not: {bad[:5]}")
    if dry: return jobs, bad
    out = project.stage(site, "02_shots"); mf = out / f"{site}_02_shots_manifest.json"
    man = project.jload(mf) or {"site": site, "frame": "scanner coordinates (SOCS); the matrices are in 03_registration",
                             "filter": f"none: every echo at range >= {MIN_RANGE} m", "scans": {}}
    man["unmatched"] = bad
    todo = [j for j in jobs if "error" in man["scans"].get(j[1], {"error": 1}) or not shots_file(site, j[1]).exists()]
    log(f"{site}: {len(todo)} scans to convert"); t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, (name, res) in enumerate(pool.map(convert_one, todo), 1):
            man["scans"][name] = res; project.jsave(man, mf)
            if "error" in res: log(f"{site}: {name}: ERROR {res['error'][:120]}")
            elif k % 10 == 0 or k == len(todo):
                log(f"{site}: {k}/{len(todo)} {name}: {res['pulses_written']:,} pulses, {res['echoes_written']:,} echoes ({(time.time() - t0) / 60:.0f} min)")
    ok = [v for v in man["scans"].values() if "error" not in v]
    man["totals"] = {"scans": len(ok), "pulses_written": sum(v["pulses_written"] for v in ok), "echoes_written": sum(v["echoes_written"] for v in ok),
                     "misses": sum(v["misses_reconstructed"] for v in ok), "gb": round(sum(v["parquet_mb"] for v in ok) / 1000, 1),
                     "errors": sum("error" in v for v in man["scans"].values())}
    project.jsave(man, mf); project.record_run(site, "02_shots", **man["totals"])
    if man["totals"]["errors"]: raise RuntimeError(f"{site}: {man['totals']['errors']} scans failed; see {mf.name}")


def done(site):
    t = (project.jload(ROOT / site / "02_shots" / f"{site}_02_shots_manifest.json") or {}).get("totals")
    return bool(t) and t.get("errors", 1) == 0


if __name__ == "__main__":
    p = project.parser(__doc__); p.add_argument("--workers", type=int, default=4); p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    project.for_sites(run, a.sites, "02_shots", workers=a.workers, dry=a.dry_run)
