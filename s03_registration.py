"""Stage 03: the scan matrices, optionally refined with sylva.coreg, and a registration report.

usage: s03_registration.py [PLOT ...] [--report-only] [--workers=N]

The matrix set per plot is chosen in project.toml ("registration", per plot or in [defaults]):
  sop     03_transforms.json and 03_transforms_dat/ from the SOP matrices of the RiSCAN project (project.rsp), or for
          source = "exports" the exported PRCS matrices (<exports>/<PLOT>/PRCS); refused when they mix frames, i.e.
          some are geocentric ("export" is the old name of this mode)
  coreg   as sop, then sylva.coreg from those priors (scans in "no_prior" get none, so a broken SOP cannot anchor them):
          03_coreg_dat/, _coreg_transforms.json, _coreg_report.txt, _coreg_before_after.json,
          _registration_report_coreg.html. Every later stage uses 03_coreg_dat where it exists.
  previous  matrices delivered with the data (previous_transforms: a transforms JSON with scans[].name and world_from_scan),
          renumbered to the stage-01 scan indices; for plots whose SOPs are unusable
One record per stage-01 scan; a position with two scans shares its matrix. Scans listed in "unregistered" are kept
with registered = false. The report: pair residuals of 5 cm / 30 m samples of the scans at the matrices the merge uses,
03_registration/<PLOT>_03_registration_report.html and _registration_residuals.json.
"""
import json, time
from pathlib import Path
import numpy as np
import project
from project import ROOT, log
from s01_convert import jobs_for                              # the stage-01 scan list and its position indices (point_source_id)


def write_transforms(site, scans, matrices_from, skipped, missing):
    """Write 03_registration/<PLOT>_03_transforms.json (the scan records) and _03_transforms_notes.txt (a short summary).

    scans: [{"index", "name", "source", "registered", "world_from_scan"}], one per stage-01 scan with a matrix, the matrix
    a 4x4 list mapping the scanner frame to the project frame. matrices_from: where the matrices came from (text).
    skipped: positions omitted by the filter file. missing: scans for which no matrix was found.
    """
    reg = project.stage(site, "03_registration")
    project.jsave({"reference": 0, "matrices_from": matrices_from, "frame": "project coordinates (PRCS)", "omitted_positions": sorted(skipped),
                "scans_without_matrix": missing, "scans": scans}, reg / f"{site}_03_transforms.json")
    unreg = [s["name"] for s in scans if not s["registered"]]
    (reg / f"{site}_03_transforms_notes.txt").write_text(
        f"{len(scans)} scans from {len({s['name'].split('#')[0] for s in scans})} positions; matrices: {matrices_from}\n"
        + (f"OMITTED (filter file): {sorted(skipped)}\n" if skipped else "") + (f"NO MATRIX: {missing}\n" if missing else "")
        + (f"NOT REGISTERED: {unreg}\n" if unreg else ""))
    log(f"{site}: {len(scans)} scans, omitted {sorted(skipped)}, no matrix {missing}, unregistered {unreg}")


def build_sop(site):
    """Registration "sop" (and the priors of "coreg"): the matrices as delivered (project.sop_matrices), one per position.

    Writes 03_transforms.json (every stage-01 scan takes its position's matrix) and 03_transforms_dat/<position>.DAT (every
    position with a matrix, replacing the folder's old .DAT files). Raises ValueError if any matrix is geocentric.
    """
    prcs, origin = project.sop_matrices(site)
    # a translation over 100 km means a geocentric (ECEF) matrix; mixed with local ones the merge would be meaningless
    far = sorted(n for n, m in prcs.items() if np.abs(m[:3, 3]).max() > 1e5)
    if far: raise ValueError(f"{site}: {len(far)} SOP matrices are geocentric ({far[:4]}...); use registration = 'previous' or 'coreg'")
    _, jobs, skipped, _ = jobs_for(site); unreg = set(project.setting(site, "unregistered"))
    scans, missing = [], []
    for _, name, src, _, index in jobs:
        pos = name.split("#")[0]                               # "<position>#2" (a second scan) uses its position's SOP
        if pos not in prcs: missing.append(name); continue
        scans.append({"index": index, "name": name, "source": src, "registered": name not in unreg, "world_from_scan": prcs[pos].tolist()})
    write_transforms(site, scans, origin, skipped, missing)
    dat = project.stage(site, "03_registration") / f"{site}_03_transforms_dat"; dat.mkdir(exist_ok=True)
    for f in dat.glob("*.DAT"): f.unlink()                     # no stale matrices from an earlier run
    for n, m in prcs.items(): np.savetxt(dat / f"{n}.DAT", m, fmt="%.16f")


def build_previous(site):
    """Registration "previous": matrices from an earlier registration (the previous_transforms JSON) for the stage-01 scans.

    Each stage-01 scan takes the record of the same name, else that of its position; it keeps the stage-01 index (so the
    point_source_id stays consistent) and is unregistered if either the previous set or "unregistered" says so. Writes
    03_transforms.json only (no 03_transforms_dat).
    """
    src = project.setting_path(site, "previous_transforms")
    if src is None: raise ValueError(f"{site}: registration = 'previous' needs previous_transforms in project.toml")
    rx = json.load(open(src))
    M = {x["name"]: x for x in rx["scans"]}; _, jobs, skipped, _ = jobs_for(site); unreg = set(project.setting(site, "unregistered"))
    scans, missing = [], []
    for _, name, src, _, index in jobs:
        x = M.get(name) or M.get(name.split("#")[0])
        if x is None: missing.append(name); continue
        scans.append({"index": index, "name": name, "source": x.get("source", src), "registered": bool(x.get("registered", True)) and name not in unreg,
                      "world_from_scan": x["world_from_scan"]})
    write_transforms(site, scans, f"previous registration ({src}), indices of the stage-01 scans", skipped, missing)


def thinned(site, name):
    """A 5 cm, 30 m scanner-frame sample of a stage-01 scan, cached in SCRATCH/thin.

    Points within 30 m of the scanner, voxel-thinned to 5 cm: enough overlap for the pair residuals at a fraction of the
    points. Returns an (n, 3) array; cached as SCRATCH/thin/<PLOT>/<scan>.npy (delete it if the stage-01 scan is redone).
    """
    p = project.SCRATCH / "thin" / site / f"{name}.npy"
    if p.exists(): return np.load(p)
    from sylva import io, filters, PointCloud
    c = io.read(str(project.scan_laz(site, name)))
    xyz = filters.voxel_downsample(PointCloud(c.xyz[np.linalg.norm(c.xyz, axis=1) < 30.0]), 0.05).xyz
    p.parent.mkdir(parents=True, exist_ok=True); np.save(p, xyz); return xyz


def residual_summary(clouds, T, names):
    """Pair residuals of scans at given matrices, summarised in cm.

    clouds: scanner-frame PointClouds; T: one 4x4 world_from_scan per cloud; names: one per cloud. Each scan is paired with
    its 4 nearest scanner positions; the residual of a pair is the median distance (m) to the other scan's nearest point
    within 1 m, after thinning to 5 cm. Returns the number of pairs, the median, 90th percentile and worst of the pair
    medians, and per scan the median over the pairs it is in. Pairs without overlap (NaN median) are left out.
    """
    from sylva.registration_report import pair_residuals
    res = pair_residuals(clouds, T, pairs=None, neighbours=4, search=1.0, voxel=0.05)
    m = np.array([r.median for r in res if np.isfinite(r.median)]); per = {}
    for r in res:
        if np.isfinite(r.median):
            for k in (r.i, r.j): per.setdefault(names[k], []).append(r.median)   # r.i, r.j: indices into clouds/names
    return {"pairs": len(res), "median_cm": round(100 * float(np.median(m)), 1), "p90_cm": round(100 * float(np.percentile(m, 90)), 1),
            "worst_cm": round(100 * float(m.max()), 1), "scan_medians_cm": {k: round(100 * float(np.median(v)), 1) for k, v in sorted(per.items())}}


def coreg(site, workers=None):
    """sylva.coreg on the stage-01 scans from the SOP matrices; the result is put back in the project frame.

    Needs build_sop first (03_transforms.json and 03_transforms_dat). Writes 03_coreg_dat/<scan>.DAT (project-frame
    world_from_scan of each scan coreg registered; every later stage prefers these), _03_coreg_transforms.json (coreg's
    own result, in its world frame), _03_coreg_transforms_project_frame.json, _03_coreg_report.txt,
    _03_coreg_before_after.json (pair residuals at the SOPs and after coreg) and _03_registration_report_coreg.html.
    workers: coreg's worker count (None: its default).
    """
    import sylva
    from sylva import coreg as sc, riscan
    from sylva.registration_report import write_report
    reg = ROOT / site / "03_registration"; t0 = time.time(); no_prior = set(project.setting(site, "no_prior"))
    tj = json.load(open(reg / f"{site}_03_transforms.json"))
    # the scans to register: those marked registered whose stage-01 LAZ exists; names[k] is scan k throughout
    names = [x["name"] for x in tj["scans"] if x.get("registered", True) and project.scan_laz(site, x["name"]).exists()]
    proj = project.riscan_project(site)                        # levelling, targets and GNSS where the project has them
    pos = {p.name: p for p in proj} if proj is not None else {}
    # per scan: priors (approximate world_from_levelled_scan or None), levelling (4x4 rotation that levels the scan),
    # reflectors (targets seen, scanner frame) and gnss (lat, lon, alt or None)
    pd = reg / f"{site}_03_transforms_dat"; priors, levelling, reflectors, gnss = [], [], [], []
    for n in names:
        base = n.split("#")[0]; p = pos.get(base)
        # the SOP of the scan's position, unless the scan or its position is listed in no_prior
        M = np.loadtxt(pd / f"{base}.DAT") if (pd / f"{base}.DAT").exists() and n not in no_prior and base not in no_prior else None
        # levelling: the scanner's own attitude if recorded, else the rotation part of the SOP (prior or not), else None
        lv = p.levelling if (p is not None and p.levelling is not None) else None
        if lv is None and M is not None: lv = np.eye(4); lv[:3, :3] = M[:3, :3]
        if lv is None and p is not None and p.sop is not None: lv = np.eye(4); lv[:3, :3] = np.asarray(p.sop)[:3, :3]
        # coreg takes priors as world_from_levelled_scan: since world_from_scan = prior @ levelling, prior = M @ inv(levelling)
        levelling.append(lv); priors.append(None if M is None else M @ np.linalg.inv(lv))
        reflectors.append(p.reflectors() if p is not None else []); gnss.append(p.gnss if p is not None else None)
    # GNSS fixes to local metres (east, north, up), NaN rows where a scan has none; only when at least one scan has a fix
    g = riscan.gnss_to_local([x if x is not None else np.full(3, np.nan) for x in gnss]) if any(x is not None for x in gnss) else None
    log(f"{site}: coreg on {len(names)} scans; no prior for {[n for n, p in zip(names, priors) if p is None]}")
    cfg = sc.CoregConfig(refine_multiview=True, verbose=True); cfg.riscan_filter = "none"   # the stage-01 scans are already filtered
    if workers: cfg.workers = workers
    # the world is the reference scan's levelled frame, so the reference must have a prior; mapped back to the project frame below
    # (raises StopIteration if no scan has a prior). project_from_world = the reference's prior; every prior is re-expressed
    # in the world frame, so the reference's own prior becomes the identity.
    ref = next(k for k, p in enumerate(priors) if p is not None); cfg.reference_scan = ref
    project_from_world = priors[ref].copy()
    priors = [None if p is None else np.linalg.inv(project_from_world) @ p for p in priors]
    files = [str(project.scan_laz(site, n)) for n in names]
    result = sc.coregister(files, cfg, names=names, levelling=levelling, reflectors=reflectors,
                           approximate_positions=g if (g is not None and np.isfinite(g).any()) else None, priors=priors)
    result.save(reg / f"{site}_03_coreg_transforms.json"); (reg / f"{site}_03_coreg_report.txt").write_text(result.report() + "\n")
    # idx: indices into names of the scans coreg registered; the rest are reported and get no matrix
    idx = [k for k, r in enumerate(result.registered) if r]; unreg = [names[k] for k, r in enumerate(result.registered) if not r]
    log(f"{site}: coreg registered {len(idx)} of {len(names)} ({time.time() - t0:.0f} s); unregistered {unreg}")
    cd = reg / f"{site}_03_coreg_dat"; cd.mkdir(exist_ok=True)
    for f in cd.glob("*.DAT"): f.unlink()                      # no stale matrices from an earlier run
    # transform_for(k) = world_from_scan for the raw (unlevelled) scan; project_from_world takes it to the project frame.
    # T is keyed by the index into names, one .DAT per scan (not per position: two scans of a position may now differ).
    T = {k: project_from_world @ result.transform_for(k) for k in idx}
    for k in idx: np.savetxt(cd / f"{names[k]}.DAT", T[k], fmt="%.16f")
    project.jsave({"frame": f"project frame of {pd.name}; world_from_scan = prior(reference) @ coreg pose", "reference": names[ref],
                "scans": [{"name": names[k], "world_from_scan": T[k].tolist()} for k in idx]}, reg / f"{site}_03_coreg_transforms_project_frame.json")
    # before/after: residuals of the registered scans at their SOPs (where they have one) and at the coreg matrices.
    # clouds, nm and before are indexed by position in idx, not by the index into names.
    clouds = [sylva.PointCloud(thinned(site, names[k])) for k in idx]; nm = [names[k] for k in idx]
    before = [np.loadtxt(pd / f"{n.split('#')[0]}.DAT") if (pd / f"{n.split('#')[0]}.DAT").exists() else None for n in nm]
    summary = {"site": site, "scans": len(names), "registered": len(idx), "unregistered": unreg, "priors": str(pd), "no_prior": sorted(no_prior)}
    for tag, Ts in (("before", before), ("after", [T[k] for k in idx])):
        ok = [i for i, t in enumerate(Ts) if t is not None]   # the scans that have a matrix in this set
        summary[tag] = residual_summary([clouds[i] for i in ok], [Ts[i] for i in ok], [nm[i] for i in ok])
        log(f"{site}: {tag}: pair medians median {summary[tag]['median_cm']} cm, p90 {summary[tag]['p90_cm']}, worst {summary[tag]['worst_cm']}")
    project.jsave(summary, reg / f"{site}_03_coreg_before_after.json")
    if len(idx) >= 2:                                          # a report needs at least one pair
        write_report(reg / f"{site}_03_registration_report_coreg.html", clouds, [T[k] for k in idx], nm, neighbours=4, search=1.0, voxel=0.05,
                     tolerance=0.10, worst=3, title=f"{site}: registration after sylva.coreg (priors: {pd.name})")


def report(site):
    """The registration report at the matrices the merge uses (project.matrices: the coreg set if any, else 03_transforms).

    Every scan with a matrix and a stage-01 LAZ is compared with its neighbours at 5 cm / 30 m samples (thinned).
    Writes 03_registration/<PLOT>_03_registration_report.html and _03_registration_residuals.json (the latter marks the
    stage done). Returns the residual summary (residual_summary plus the matrix source, scan count and seconds).
    """
    import sylva
    from sylva.registration_report import write_report
    t0 = time.time(); reg = ROOT / site / "03_registration"; M, src = project.matrices(site)
    names = sorted(n for n in M if project.scan_laz(site, n).exists())
    clouds = [sylva.PointCloud(thinned(site, n)) for n in names]; T = [M[n] for n in names]
    summary = {"site": site, "matrices": src, "scans": len(names), **residual_summary(clouds, T, names), "seconds": round(time.time() - t0)}
    write_report(reg / f"{site}_03_registration_report.html", clouds, T, names, neighbours=4, search=1.0, voxel=0.05, tolerance=0.10, worst=3,
                 title=f"{site}: registration at {src}")
    project.jsave(summary, reg / f"{site}_03_registration_residuals.json")
    log(f"{site}: {len(names)} scans, {summary['pairs']} pairs: pair medians median {summary['median_cm']} cm, p90 {summary['p90_cm']}, worst {summary['worst_cm']}")
    return summary


def run(site, report_only=False, workers=None):
    """Stage 03 for one plot: build the matrix set the plot's "registration" setting asks for, then the report.

    report_only: skip building the matrices and only redo the report at the existing ones. workers: passed to coreg.
    """
    mode = project.setting(site, "registration")
    if not report_only:
        if mode not in ("sop", "export", "coreg", "previous"): raise ValueError(f"{site}: unknown registration {mode!r}")
        # "previous" replaces the SOPs; "sop", "export" and "coreg" all start from them (coreg needs them as priors)
        build_previous(site) if mode == "previous" else build_sop(site)
        if mode == "coreg": coreg(site, workers)
    s = report(site); project.record_run(site, "03_registration", mode=mode, median_cm=s["median_cm"], worst_cm=s["worst_cm"])


# Stage 03 is done once the registration report's residuals are written (the last thing run() does).
def done(site): return (ROOT / site / "03_registration" / f"{site}_03_registration_residuals.json").exists()


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")                  # the reports draw figures; no display needed
    p = project.parser(__doc__); p.add_argument("--report-only", action="store_true"); p.add_argument("--workers", type=int, default=None)
    a = p.parse_args()
    project.for_sites(run, a.sites, "03", report_only=a.report_only, workers=a.workers)
