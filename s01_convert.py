"""Stage 01: one filtered LAZ per scan from the RiSCAN PRO LAZ exports (scanner coordinates).

usage: s01_convert.py [SITE ...] [--workers=4] [--only=ScanPos001,...]

Input: RiSCAN_OUTPUTS/<SITE>/*.laz (one per scan, unfiltered) and its POINT_FILTERS.txt, or
01_scans/<SITE>_01_filter_override.txt where a plot deliberately departs from it (Tumbarumba: -30 dB). Filter keys:
  range lo hi            scanner range (the exports are in SOCS, so range = |xyz|)
  reflectance lo hi      dB (the export stores it x100 as int16)
  deviation lo hi        RIEGL pulse-shape deviation
  amplitude lo hi        dB, x100 in the export
  omit p1 p2 ...         scan positions left out
Output: 01_scans/<SITE>_01_<scan>.laz with reflectance, amplitude, deviation, range, return_number and
number_of_returns (recounted after filtering; a pulse = the points sharing a gps_time), echo_type (RIEGL type before
filtering: 0 single, 1 first, 2 interior, 3 last), gps_time, point_source_id (the position's index); plus
<SITE>_01_filter.txt and <SITE>_01_manifest.json. Resumable: a scan whose LAZ exists for the same export file is skipped.
A position with two export files gives ScanPosNNN and ScanPosNNN#2.
"""
import json, os, re, time, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import tern
from tern import EXPORTS, log

NAME = re.compile(r"^(ScanPos\d+|tilt ?\d+)", re.I)


def read_filters(path):
    st = {}
    for line in Path(path).read_text().splitlines():
        parts = line.replace(",", " ").split()
        if not parts or parts[0].startswith("#"): continue
        key = parts[0].lower()
        if key == "omit": st["omit"] = {int(p) for p in parts[1:]}
        elif len(parts) >= 3: st[key] = (float(parts[1]), float(parts[2]))
    return st


def scan_name(path):
    m = NAME.match(Path(path).name)
    if not m: raise ValueError(f"cannot name {path}")
    n = m.group(1)
    return "tilt " + re.sub(r"\D", "", n) if n.lower().startswith("tilt") else n      # 'tilt 001', as the SOP files are named


def filter_file(site):
    o = tern.ROOT / site / "01_scans" / f"{site}_01_filter_override.txt"
    return o if o.exists() else EXPORTS / site / "POINT_FILTERS.txt"


def jobs_for(site, only=None):
    """(filters, [(site, name, export file, filters, position index)], omitted positions). Positions are numbered in
    name order, upright before tilt; the index is the point_source_id every later stage uses."""
    st = read_filters(filter_file(site)); omit = st.get("omit", set())
    by_pos = {}
    for f in sorted(EXPORTS.joinpath(site).glob("*.laz"), key=lambda p: p.name.lower()): by_pos.setdefault(scan_name(f), []).append(f)
    names = sorted(by_pos, key=lambda n: (n.lower().startswith("tilt"), int(re.sub(r"\D", "", n))))
    jobs, skipped = [], []
    for i, n in enumerate(names, 1):
        if int(re.sub(r"\D", "", n)) in omit and not n.lower().startswith("tilt"): skipped.append(n); continue
        for k, f in enumerate(by_pos[n], 1):
            nm = n if k == 1 else f"{n}#{k}"
            if not only or nm in only: jobs.append((site, nm, str(f), st, i))
    return st, jobs, skipped


def convert_one(args):
    site, name, src, st, index = args
    from sylva import io, PointCloud
    laz = tern.scan_laz(site, name); tmp = laz.with_name(laz.stem + ".part.laz")
    try:
        t0 = time.time()
        c = io.read(src); n_in = len(c); xyz = c.xyz; a = c.attrs
        rng = np.linalg.norm(xyz, axis=1).astype(np.float32)
        refl = a["Reflectance"].astype(np.float32) / 100.0; amp = a["Amplitude"].astype(np.float32) / 100.0
        dev = a["Deviation"].astype(np.int32)
        rn0 = a["return_number"].astype(np.uint8); nr0 = a["number_of_returns"].astype(np.uint8)
        keep = np.ones(n_in, bool); dropped = {}
        for key, v in (("range", rng), ("reflectance", refl), ("deviation", dev), ("amplitude", amp)):
            if key in st:
                lo, hi = st[key]; m = (v >= lo) & (v <= hi); dropped[key] = int((keep & ~m).sum()); keep &= m
        et = np.where(nr0 == 1, 0, np.where(rn0 == 1, 1, np.where(rn0 == nr0, 3, 2))).astype(np.uint8)
        idx = np.flatnonzero(keep); del keep
        gps = a["gps_time"][idx]; rn = rn0[idx]
        # return numbering after filtering: a pulse is the points sharing a gps_time, ranked by the original order
        order = np.lexsort((rn, gps)); g = gps[order]
        start = np.flatnonzero(np.r_[True, g[1:] != g[:-1]]); size = np.diff(np.r_[start, len(g)])
        rank = np.arange(len(g)) - np.repeat(start, size); cnt = np.repeat(size, size)
        new_rn = np.empty(len(idx), np.uint8); new_nr = np.empty(len(idx), np.uint8)
        new_rn[order] = np.minimum(rank + 1, 15); new_nr[order] = np.minimum(cnt, 15); del order, g, start, size, rank, cnt
        attrs = {"reflectance": refl[idx], "amplitude": amp[idx], "deviation": np.clip(dev[idx], 0, 65535).astype(np.uint16),
                 "range": rng[idx], "return_number": new_rn, "number_of_returns": new_nr, "echo_type": et[idx],
                 "gps_time": gps.astype(np.float64), "point_source_id": np.full(len(idx), index, np.uint16)}
        del refl, amp, dev, rng, rn0, nr0, et, a, c
        io.write(PointCloud(xyz[idx], attrs), str(tmp)); os.replace(tmp, laz)
        return name, {"file": Path(src).name, "points": int(n_in), "dropped": dropped, "written": int(len(idx)),
                      "laz_mb": round(laz.stat().st_size / 1e6, 1), "seconds": round(time.time() - t0)}
    except Exception:
        if tmp.exists(): tmp.unlink()
        return name, {"file": Path(src).name, "error": traceback.format_exc().strip().splitlines()[-1]}


def run(site, workers=4, only=None):
    out = tern.stage(site, "01_scans")
    st, jobs, skipped = jobs_for(site, only); ff = filter_file(site)
    (out / f"{site}_01_filter.txt").write_text(f"# applied to every scan of {site} by s01_convert.py; source: {EXPORTS / site} (scanner coordinates)\n"
                                               f"# from {ff}\n" + ff.read_text().strip() + "\n" + (f"# positions omitted: {sorted(skipped)}\n" if skipped else ""))
    mf = out / f"{site}_01_manifest.json"
    man = tern.jload(mf) or {"site": site, "source": f"RiSCAN PRO LAZ export, {EXPORTS / site}", "frame": "scanner coordinates (SOCS); the matrices are in 03_registration",
                             "filters": {k: (sorted(v) if isinstance(v, set) else list(v)) for k, v in st.items()}, "filter_file": str(ff),
                             "omitted_positions": sorted(skipped), "scans": {}}
    todo = [j for j in jobs if not ("error" not in man["scans"].get(j[1], {"error": 1}) and man["scans"][j[1]].get("file") == Path(j[2]).name
                                    and tern.scan_laz(site, j[1]).exists())]
    log(f"{site}: {len(todo)} of {len(jobs)} scans to convert ({len(skipped)} positions omitted)")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, (name, res) in enumerate(pool.map(convert_one, todo), 1):
            man["scans"][name] = res; tern.jsave(man, mf)
            if "error" in res: log(f"{site}: {name}: ERROR {res['error'][:160]}")
            elif k % 10 == 0 or k == len(todo):
                log(f"{site}: {k}/{len(todo)} {name}: {res['written']:,} of {res['points']:,} kept, {(time.time() - t0) / 60:.0f} min")
    ok = [v for v in man["scans"].values() if "error" not in v]
    man["totals"] = {"scans": len(ok), "points_in": sum(v["points"] for v in ok), "points_written": sum(v["written"] for v in ok),
                     "gb": round(sum(v["laz_mb"] for v in ok) / 1000, 1), "errors": sum("error" in v for v in man["scans"].values())}
    tern.jsave(man, mf); tern.record_run(site, "01_convert", **man["totals"])
    if man["totals"]["errors"]: raise RuntimeError(f"{site}: {man['totals']['errors']} scans failed; see {mf.name}")


def done(site):
    m = tern.jload(tern.ROOT / site / "01_scans" / f"{site}_01_manifest.json") or {}; t = m.get("totals")
    return bool(t) and t.get("errors", 1) == 0 and t["scans"] == len(m["scans"])


if __name__ == "__main__":
    p = tern.parser(__doc__); p.add_argument("--workers", type=int, default=4); p.add_argument("--only", default="")
    a = p.parse_args()
    tern.for_sites(run, a.sites, "01", workers=a.workers, only=set(a.only.split(",")) - {""})
