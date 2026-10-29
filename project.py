"""The project: where the data are and how each plot is processed, from <root>/project.toml.

The project root is, in order: --project=PATH on the command line, the TLS_PROJECT environment variable, or the current
directory. Every plot has a folder <root>/<PLOT>/ with one sub-folder per stage. See project.example.toml for the keys.
"""
import argparse, json, os, re, sys, time, tomllib, traceback
from pathlib import Path
import numpy as np

CODE = Path(__file__).resolve().parent


def _root():
    for a in sys.argv[1:]:
        if a.startswith("--project="): return Path(a.split("=", 1)[1]).expanduser().resolve()
    if "--project" in sys.argv[:-1]: return Path(sys.argv[sys.argv.index("--project") + 1]).expanduser().resolve()
    return Path(os.environ.get("TLS_PROJECT", ".")).expanduser().resolve()


ROOT = _root()
CONFIG = tomllib.loads((ROOT / "project.toml").read_text()) if (ROOT / "project.toml").exists() else {}
_paths = CONFIG.get("paths", {})


def _path(value, plot=None):
    """A configured path: {plot} is filled in, ~ expanded, relative paths are relative to the project root."""
    if value is None: return None
    p = Path(str(value).format(plot=plot) if plot else str(value)).expanduser()
    return p if p.is_absolute() else ROOT / p


EXPORTS = _path(_paths.get("exports", "exports"))              # legacy input (source = "exports"): <EXPORTS>/<PLOT>/*.laz
SCRATCH = _path(os.environ.get("TLS_SCRATCH", _paths.get("scratch", "~/tls_scratch")))
RCT_SIF = str(_path(_paths["raycloudtools_sif"])) if "raycloudtools_sif" in _paths else os.environ.get("RCT_SIF")
PYARROW_PYTHON = str(_path(_paths["pyarrow_python"])) if "pyarrow_python" in _paths else sys.executable
DEFAULTS = {"source": "riscan", "riscan": None, "registration": "sop", "filter": None, "unregistered": [], "no_prior": [], "previous_transforms": None,
            "manual_trees": None, "manual_trees_table": None, "manual_trees_frame": None, "inventory": None, "inventory_frame": None,
            "inventory_x": "x", "inventory_y": "y", "code": None, "name": None}
DEFAULTS.update(CONFIG.get("defaults", {}))
PLOT_BOX = {"size_m": 100.0, "tolerance_m": 4.0, "isolated_stem_m": 15.0, **CONFIG.get("plot_box", {})}


def plots():
    """The plots of the project: [plots.<NAME>] tables in project.toml, else every folder holding a RiSCAN project in 00_raw."""
    if CONFIG.get("plots"): return list(CONFIG["plots"])
    return sorted(p.parent.parent.name for p in ROOT.glob("*/00_raw/*.RiSCAN") if p.is_dir())


def setting(plot, key):
    return CONFIG.get("plots", {}).get(plot, {}).get(key, DEFAULTS.get(key))


def setting_path(plot, key): return _path(setting(plot, key), plot)
def has_trees(plot): return bool(setting(plot, "manual_trees"))


def riscan_dir(plot):
    """The plot's RiSCAN PRO project: the "riscan" setting, else the one *.RiSCAN folder in <PLOT>/00_raw (None if there is none)."""
    p = setting_path(plot, "riscan")
    if p: return p
    found = sorted(d for d in (ROOT / plot / "00_raw").glob("*.RiSCAN") if d.is_dir())
    if len(found) > 1: raise ValueError(f"{plot}: {len(found)} RiSCAN projects in 00_raw; set riscan in project.toml")
    return found[0] if found else None


_riscan = {}


def riscan_project(plot):
    """The plot's RiSCAN project parsed by sylva (positions, SOP matrices, scan patterns, .rxp files); None without one."""
    if plot not in _riscan:
        d = riscan_dir(plot)
        if d is None: _riscan[plot] = None
        else:
            import sylva
            _riscan[plot] = sylva.read_riscan_project(str(d))
    return _riscan[plot]


def has_rxp(plot):
    """Whether the plot's RiSCAN project holds its .rxp scans (stages 02 and 10 need them)."""
    proj = riscan_project(plot)
    return proj is not None and any(p.rxp is not None for p in proj)


def scan_files(plot):
    """{position: [point files]} that stage 01 reads, in file-name order. source = "riscan": every position of
    project.rsp with each of its scans' .rdbx, the RiSCAN PRO database of the imported scan
    (<project>/project.rdb/SCANS/<position>/SINGLESCANS/<scan>/<scan>.rdbx), an empty list where there is none;
    source = "exports" (legacy): the per-scan RiSCAN PRO LAZ exports, <exports>/<PLOT>/<position> ... .laz."""
    out = {}
    if setting(plot, "source") == "exports":
        for f in sorted(EXPORTS.joinpath(plot).glob("*.laz"), key=lambda p: p.name.lower()):
            m = re.match(r"^(ScanPos\d+|tilt ?\d+)", f.name, re.I)
            if not m: raise ValueError(f"cannot name {f}")
            n = m.group(1)
            out.setdefault("tilt " + re.sub(r"\D", "", n) if n.lower().startswith("tilt") else n, []).append(f)  # 'tilt 001', as the SOPs are named
        return out
    d = riscan_dir(plot)
    if d is None: raise FileNotFoundError(f"{plot}: no RiSCAN project in 00_raw (a zipped one must be unpacked)")
    for pos in riscan_project(plot):
        out[pos.name] = sorted((d / "project.rdb" / "SCANS" / pos.name / "SINGLESCANS").glob("*/*.rdbx"))
    return out


def sop_matrices(plot):
    """{position: 4x4 scanner-to-project matrix} as the data came: the RiSCAN project's SOPs (project.rsp), or for
    source = "exports" the exported PRCS matrices (<exports>/<PLOT>/PRCS/*.DAT); and where they came from."""
    if setting(plot, "source") == "exports":
        d = EXPORTS / plot / "PRCS"
        return {f.stem: np.loadtxt(f) for f in d.glob("*.DAT")}, f"{d} (RiSCAN PRO export)"
    proj = riscan_project(plot)
    if proj is None: raise FileNotFoundError(f"{plot}: no RiSCAN project in 00_raw")
    return {p.name: np.asarray(p.sop, float) for p in proj if p.sop is not None}, f"{riscan_dir(plot) / 'project.rsp'} (SOP)"


def stage(plot, folder):
    """<root>/<PLOT>/<folder>, created if missing."""
    d = ROOT / plot / folder; d.mkdir(parents=True, exist_ok=True); return d


def log(*a): print(time.strftime("%H:%M"), *a, flush=True)


def jload(path, default=None):
    p = Path(path)
    return json.load(open(p)) if p.exists() else default


def jsave(obj, path):
    """Write JSON through a temporary file, so a crash never leaves half a record."""
    p = Path(path); tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))); os.replace(tmp, p)


def parser(doc, sites=True):
    """An argparse parser whose description is the script's docstring; PLOT arguments (none = every plot) and --project."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default=None, help="project root holding project.toml (default: $TLS_PROJECT, else the current directory)")
    if sites: p.add_argument("sites", nargs="*", metavar="PLOT", help="plot folders (default: every plot)")
    return p


def for_sites(fn, sites, label, *args, **kw):
    """fn(plot, ...) for each plot; a failure is logged to 99_meta/<PLOT>_99_error_<label>.txt and the next plot runs."""
    ok = True
    for s in sites or plots():
        try: fn(s, *args, **kw)
        except Exception:
            ok = False; err = traceback.format_exc(); print(err, flush=True)
            (stage(s, "99_meta") / f"{s}_99_error_{label}.txt").write_text(err)
    return ok


def matrices(plot):
    """{scan name: 4x4 world_from_scan} the merge uses, and where they came from: the coreg set if present, else 03_transforms."""
    reg = ROOT / plot / "03_registration"
    cd = reg / f"{plot}_03_coreg_dat"
    if cd.exists():
        return {f.stem: np.loadtxt(f) for f in cd.glob("*.DAT")}, "03_coreg_dat (sylva.coreg)"
    j = json.load(open(reg / f"{plot}_03_transforms.json"))
    return ({x["name"]: np.asarray(x["world_from_scan"], float) for x in j["scans"] if x.get("registered", True)},
            "03_transforms: " + j.get("matrices_from", ""))


def matrices_dir(plot):
    """The folder of .DAT matrices the merge uses."""
    reg = ROOT / plot / "03_registration"; d = reg / f"{plot}_03_coreg_dat"
    return d if d.exists() else reg / f"{plot}_03_transforms_dat"


def trees_frame_dir(plot):
    """The .DAT matrices the manual trees are in (manual_trees_frame), else the merge's own."""
    return setting_path(plot, "manual_trees_frame") or matrices_dir(plot)


def scan_laz(plot, name): return ROOT / plot / "01_scans" / f"{plot}_01_{name}.laz"


def merged_tiles(plot):
    import glob
    return sorted(glob.glob(str(ROOT / plot / "04_registered" / f"{plot}_04_merged_2cm_tiles" / "*.laz")))


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


def load_dtm(plot): return load_raster(ROOT / plot / "05_ground" / f"{plot}_05_dtm_1m.tif")


def plot_bounds(plot):
    """(xmin, ymin, xmax, ymax) of the plot box (99_meta/<PLOT>_99_plot_bounds.json, s99_plot.py --bounds), else the box
    around the scan positions."""
    j = jload(ROOT / plot / "99_meta" / f"{plot}_99_plot_bounds.json")
    if j: return j["xmin"], j["ymin"], j["xmax"], j["ymax"]
    o = np.array([m[:3, 3] for m in matrices(plot)[0].values()])
    return float(o[:, 0].min()), float(o[:, 1].min()), float(o[:, 0].max()), float(o[:, 1].max())


def edited_tiles_dir(plot): return ROOT / plot / "07_trees" / f"{plot}_07_trees_edited_tiles"


def tree_tiles(plot):
    """The tiles with per-point tree labels (treeid > 0 a tree, 0 unassigned, -1 noise) that stages 08, 09 and the review
    read: the segfix-edited labels once imported (s07_segfix.py import), else the transferred ones (s07_trees.py)."""
    import glob
    d = edited_tiles_dir(plot)
    if not d.exists(): d = ROOT / plot / "07_trees" / f"{plot}_07_merged_2cm_treeid_tiles"
    return sorted(glob.glob(str(d / "*.laz")))


def manual_tree_files(plot):
    """[(tree id, file)] of the plot's manual trees, one point cloud per tree: the rows of manual_trees_table (columns treeid
    and file, file relative to the manual_trees folder) when given, else every .ply/.las/.laz in manual_trees with the id
    taken from the last number in the file name."""
    folder = setting_path(plot, "manual_trees"); table = setting_path(plot, "manual_trees_table")
    if table:
        import csv
        return [(int(r["treeid"]), folder / r["file"]) for r in csv.DictReader(open(table))]
    out = []
    for f in sorted(folder.iterdir()):
        m = re.findall(r"\d+", f.stem)
        if f.suffix.lower() in (".ply", ".las", ".laz") and m: out.append((int(m[-1]), f))
    if len({t for t, _ in out}) != len(out): raise ValueError(f"{plot}: two manual tree files share a tree id; give manual_trees_table")
    return out


def sylva_version():
    import sylva
    return getattr(sylva, "__version__", "?")


def record_run(plot, step, **info):
    """Append one line per finished step to 99_meta/<PLOT>_99_runs.jsonl: what ran, when, with which sylva."""
    line = {"step": step, "finished": time.strftime("%Y-%m-%d %H:%M"), "sylva": sylva_version(), "script": Path(sys.argv[0]).name, **info}
    with open(stage(plot, "99_meta") / f"{plot}_99_runs.jsonl", "a") as f: f.write(json.dumps(line, default=str) + "\n")
