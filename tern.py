"""Shared settings and helpers for the TERN TLS processing scripts (built on sylva).

Paths can be moved with environment variables:
  TERN_TLS_ROOT    the TERN_TLS folder (default: the folder above this one)
  TERN_TREES_ROOT  the manual trees, TERN_TREES/<SITE>/final_trees_new_frame (default: beside TERN_TLS)
  TERN_SCRATCH     fast local disk for temporary files (default: ~/tern_scratch)
  RCT_SIF          raycloudtools apptainer image
  PYARROW_PYTHON   a Python with pyarrow (sylva's environment does not have it), for merge_shots.py
"""
import argparse, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("TERN_TLS_ROOT", Path(__file__).resolve().parent.parent))
EXPORTS = ROOT / "RiSCAN_OUTPUTS"                        # RiSCAN PRO LAZ exports and PRCS matrices, one folder per plot
TREES = Path(os.environ.get("TERN_TREES_ROOT", ROOT.parent / "TERN_TREES"))
SCRATCH = Path(os.environ.get("TERN_SCRATCH", Path.home() / "tern_scratch"))
RCT_SIF = os.environ.get("RCT_SIF", "/home/tim/containers/raycloudtools_latest.sif")
PYARROW_PYTHON = os.environ.get("PYARROW_PYTHON", "/home/tim/tumba_work/pa_venv/bin/python")
CODE = Path(__file__).resolve().parent

# registration: "export" = the PRCS matrices of the RiSCAN export (default)
#               "coreg"  = sylva.coreg from the export matrices; no_prior lists scans whose export matrix is wrong
#               "tern"   = the matrices of the TERN processing (03_transforms_rxp_run.json), where the export set is unusable
# trees: manual trees exist in TERN_TREES. inventory: field inventory (registered) under TERN_TREES, for the plot box.
# unregistered: scans in the project that were never registered.
SITES = {
    "ALICE_2021": dict(code="ALIC"),
    "BOYAGIN_2021": dict(code="BOYA", trees=True, inventory="BOYAGIN_2021/records/TERN_TLS_QA/default_outputs/individual_trees/Boyagin_20181007_inventory_registered_over5cm.csv"),
    "CALP_2021": dict(code="CLPM", trees=True, registration="tern", inventory="CALP_2021/records/TERN_TLS_QA/CALP_2018_20250506_registered.csv"),
    "CUP_2022": dict(code="CBLP"),
    "FLETCH_2022": dict(code=None, trees=True, inventory="FLETCH_2022/records/TERN_TLS_QA/default_output/individual_trees/Fletcherview_230510_inventory_registered.csv"),
    "GINGINSOUTH_2021": dict(code=None, trees=True, inventory="GINGINSOUTH_2021/records/TERN_TLS_QA/GINGINSOUTH_2019_inventory_registered.csv"),
    "GWW_2021": dict(code="GWWL", trees=True, inventory="GWW_2021/records/TERN_TLS_QA/GWW_20230405_inventory_registered.csv"),
    "LITCH_2021": dict(code="LFLD", trees=True, registration="coreg", no_prior=("ScanPos040", "ScanPos047"),
                       inventory="LITCH_2021/records/TERN_TLS_QA/default_outputs/individual_trees/LITCH_2019_20250506_registered.csv"),
    "ROBSON_2023": dict(code="FNQR", trees=True, unregistered=("tilt 042", "tilt 043", "tilt 044"),
                        inventory="ROBSON_2023/records/TERN_TLS_QA/Round_01/ROBSON_20230609_inventory_10cm.csv"),
    "TUMBA_2022": dict(code="TUMB", trees=True, inventory="TUMBA_2022/records/TERN_TLS_QA/Round_01/Tumba_2021_20250605_registered.csv"),
    "WARRA_2023": dict(code="WRRA"),
    "WOMBAT_2023": dict(code="WSBE", trees=True, inventory="WOMBAT_2023/records/TERN_TLS_QA/Round_01/WOMBAT_20180706_inventory_registered_10cm_oneStemPerMultistem_registered.csv"),
}


def setting(site, key, default=None): return SITES.get(site, {}).get(key, default)


def stage(site, folder):
    """ROOT/<SITE>/<folder>, created if missing."""
    d = ROOT / site / folder; d.mkdir(parents=True, exist_ok=True); return d


def log(*a): print(time.strftime("%H:%M"), *a, flush=True)


def jload(path, default=None):
    p = Path(path)
    return json.load(open(p)) if p.exists() else default


def jsave(obj, path):
    """Write JSON through a temporary file, so a crash never leaves half a record."""
    p = Path(path); tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))); os.replace(tmp, p)


def parser(doc, sites=True):
    """An argparse parser whose description is the script's docstring; SITE arguments (none = every plot)."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    if sites: p.add_argument("sites", nargs="*", metavar="SITE", help="plot folders (default: every plot)")
    return p


def for_sites(fn, sites, label, *args, **kw):
    """fn(site, ...) for each site; a failure is logged to 99_meta/<SITE>_99_error_<label>.txt and the next site runs."""
    ok = True
    for s in sites or list(SITES):
        try: fn(s, *args, **kw)
        except Exception:
            ok = False; err = traceback.format_exc(); print(err, flush=True)
            (stage(s, "99_meta") / f"{s}_99_error_{label}.txt").write_text(err)
    return ok


def matrices(site):
    """{scan name: 4x4 world_from_scan} the merge uses, and where they came from: the coreg set if present, else 03_transforms."""
    reg = ROOT / site / "03_registration"
    cd = reg / f"{site}_03_coreg_dat"
    if cd.exists():
        return {f.stem: np.loadtxt(f) for f in cd.glob("*.DAT")}, "03_coreg_dat (sylva.coreg)"
    j = json.load(open(reg / f"{site}_03_transforms.json"))
    return ({x["name"]: np.asarray(x["world_from_scan"], float) for x in j["scans"] if x.get("registered", True)},
            "03_transforms: " + j.get("matrices_from", ""))


def trees_frame_dir(site):
    """The matrices the manual trees (TERN_TREES) are in: those of the TERN processing, kept as *_transforms_dat_rxp_run."""
    reg = ROOT / site / "03_registration"; d = reg / f"{site}_03_transforms_dat_rxp_run"
    return d if d.exists() else reg / f"{site}_03_transforms_dat"


def scan_laz(site, name): return ROOT / site / "01_scans" / f"{site}_01_{name}.laz"


def merged_tiles(site):
    import glob
    return sorted(glob.glob(str(ROOT / site / "04_registered" / f"{site}_04_merged_2cm_tiles" / "*.laz")))


def load_raster(path):
    """A GeoTIFF as a sylva Raster (row 0 south), nodata as NaN."""
    import rasterio
    from sylva import Raster
    with rasterio.open(path) as r:
        a = r.read(1).astype(np.float64); tr = r.transform
        if r.nodata is not None and not np.isnan(r.nodata): a[a == r.nodata] = np.nan
        res = float(tr.a); xmin = float(tr.c); ymin = float(tr.f) + tr.e * a.shape[0]
        if tr.e < 0: a = a[::-1]                              # rasterio: row 0 north; sylva: row 0 south
    return Raster(a, xmin, ymin, res)


def load_dtm(site): return load_raster(ROOT / site / "05_ground" / f"{site}_05_dtm_1m.tif")


def plot_bounds(site):
    """(xmin, ymin, xmax, ymax) of the plot box (99_meta/<SITE>_99_plot_bounds.json, s99_plot.py --bounds), else the box
    around the scan positions."""
    j = jload(ROOT / site / "99_meta" / f"{site}_99_plot_bounds.json")
    if j: return j["xmin"], j["ymin"], j["xmax"], j["ymax"]
    o = np.array([m[:3, 3] for m in matrices(site)[0].values()])
    return float(o[:, 0].min()), float(o[:, 1].min()), float(o[:, 0].max()), float(o[:, 1].max())


def edited_tiles_dir(site): return ROOT / site / "07_trees" / f"{site}_07_trees_edited_tiles"


def tree_tiles(site):
    """The tiles with per-point tree labels (treeid > 0 a tree, 0 unassigned, -1 noise) that stages 08, 09 and the review
    read: the segfix-edited labels once imported (s07_segfix.py import), else the transferred ones (s07_trees.py)."""
    import glob
    d = edited_tiles_dir(site)
    if not d.exists(): d = ROOT / site / "07_trees" / f"{site}_07_merged_2cm_treeid_tiles"
    return sorted(glob.glob(str(d / "*.laz")))


def sylva_version():
    import sylva
    return getattr(sylva, "__version__", "?")


def record_run(site, step, **info):
    """Append one line per finished step to 99_meta/<SITE>_99_runs.jsonl: what ran, when, with which sylva."""
    line = {"step": step, "finished": time.strftime("%Y-%m-%d %H:%M"), "sylva": sylva_version(), "script": Path(sys.argv[0]).name, **info}
    with open(stage(site, "99_meta") / f"{site}_99_runs.jsonl", "a") as f: f.write(json.dumps(line, default=str) + "\n")
