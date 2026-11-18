"""The project: where the data are and how each plot is processed, from <root>/project.toml.

The project root is, in order: --project=PATH on the command line, the TLS_PROJECT environment variable, or the current
directory. Every plot has a folder <root>/<PLOT>/ with one sub-folder per stage. See project.example.toml for the keys.

Importing this module reads the configuration once (ROOT, CONFIG and the constants below); every stage script imports it,
so they all agree on the root and the settings. Matrices are 4x4 homogeneous world_from_scan (a RiSCAN SOP is one):
applied to column vectors [x, y, z, 1] in the scanner frame (SOCS) they give the project frame (PRCS), and the
translation column m[:3, 3] is the scanner position in the project frame.
"""
import argparse, json, os, re, sys, time, tomllib, traceback
from pathlib import Path
import numpy as np

CODE = Path(__file__).resolve().parent                         # the folder holding the stage scripts (run.py starts them from here)


def _root():
    """The project root, resolved at import time from sys.argv and the environment (argparse has not run yet).

    Order: --project=PATH or --project PATH, then $TLS_PROJECT, then the parent of a plot given as a folder path (a
    folder with a 00_raw sub-folder), then the current directory. Returns an absolute Path.
    """
    for a in sys.argv[1:]:
        if a.startswith("--project="): return Path(a.split("=", 1)[1]).expanduser().resolve()
    # "--project PATH" as two arguments; [:-1] so that a trailing "--project" with no value is not taken
    if "--project" in sys.argv[:-1]: return Path(sys.argv[sys.argv.index("--project") + 1]).expanduser().resolve()
    if "TLS_PROJECT" in os.environ: return Path(os.environ["TLS_PROJECT"]).expanduser().resolve()
    for a in sys.argv[1:]:                                     # a plot given as a folder path: its parent is the project
        if os.sep in a and not a.startswith("-") and (Path(a).expanduser() / "00_raw").is_dir():
            return Path(a).expanduser().resolve().parent
    return Path(".").resolve()


ROOT = _root()
# project.toml as a dict; empty without one (every plot then runs on DEFAULTS and plots() finds the plots by their folders)
CONFIG = tomllib.loads((ROOT / "project.toml").read_text()) if (ROOT / "project.toml").exists() else {}
_paths = CONFIG.get("paths", {})                               # the [paths] table: exports, scratch, raycloudtools_sif, pyarrow_python


def _path(value, plot=None):
    """A configured path: {plot} is filled in, ~ expanded, relative paths are relative to the project root."""
    if value is None: return None
    p = Path(str(value).format(plot=plot) if plot else str(value)).expanduser()
    return p if p.is_absolute() else ROOT / p


EXPORTS = _path(_paths.get("exports", "exports"))              # legacy input (source = "exports"): <EXPORTS>/<PLOT>/*.laz
# working space for caches (e.g. the thinned scans of the registration report); $TLS_SCRATCH wins over project.toml
SCRATCH = _path(os.environ.get("TLS_SCRATCH", _paths.get("scratch", "~/tls_scratch")))
# apptainer image with raycloudtools for stage 08; None when neither project.toml nor $RCT_SIF gives one
RCT_SIF = str(_path(_paths["raycloudtools_sif"])) if "raycloudtools_sif" in _paths else os.environ.get("RCT_SIF")
# a Python interpreter that has pyarrow (merge_shots.py runs under it), else the running one
PYARROW_PYTHON = str(_path(_paths["pyarrow_python"])) if "pyarrow_python" in _paths else sys.executable
# per-plot settings and their defaults: [defaults] in project.toml overrides these, [plots.<NAME>] overrides both (setting())
DEFAULTS = {"source": "riscan", "riscan": None, "registration": "sop", "filter": None, "unregistered": [], "no_prior": [], "previous_transforms": None,
            "segment_trees": True, "manual_trees": None, "manual_trees_table": None, "manual_trees_frame": None, "inventory": None, "inventory_frame": None,
            "inventory_x": "x", "inventory_y": "y", "code": None, "name": None,
            "scan_grid_rings": None,      # scan positions on a regular grid: plot box along the grid, this many outer rings left out
            # stage-05 ground filter: method "pmf" (sylva.ground.classify_ground_pmf) or "csf" (classify_ground_csf), and its keywords
            "ground": {"method": "pmf", "cell_size": 2.0, "max_window": 40.0}}
DEFAULTS.update(CONFIG.get("defaults", {}))
# the plot box (m): nominal side, how far field stems may lie outside it, the distance beyond which a stem counts as isolated
# the plot box (s99_plot.py --bounds): a square of side size_m placed on the field inventory and the scan positions, turned
# by at most max_rotation_deg; the clipped products reach buffer_m beyond it (plot numbers are over the box itself); see s99_plot.py for cover_r_m, stem_slack and cover_slack, and isolated_stem_m (inventory
# stems farther than this from any other are ignored)
PLOT_BOX = {"size_m": 100.0, "buffer_m": 10.0, "max_rotation_deg": 10.0, "cover_r_m": 15.0, "stem_slack": 0.01, "cover_slack": 0.005, "isolated_stem_m": 15.0,
            **CONFIG.get("plot_box", {})}


def plots():
    """The plots of the project: [plots.<NAME>] tables in project.toml, else every folder holding a RiSCAN project in 00_raw."""
    if CONFIG.get("plots"): return list(CONFIG["plots"])
    # <root>/<PLOT>/00_raw/<name>.RiSCAN: the plot is two levels up from the .RiSCAN folder
    return sorted(p.parent.parent.name for p in ROOT.glob("*/00_raw/*.RiSCAN") if p.is_dir())


def setting(plot, key):
    """A per-plot setting: [plots.<PLOT>] in project.toml, else [defaults], else DEFAULTS; None for an unknown key."""
    return CONFIG.get("plots", {}).get(plot, {}).get(key, DEFAULTS.get(key))


# setting_path: a per-plot setting read as a path (see _path: {plot} filled in, relative to the root), None if unset.
def setting_path(plot, key): return _path(setting(plot, key), plot)


# has_manual_trees: whether the plot has manually segmented trees (the manual_trees setting), which stage 07 transfers.
def has_manual_trees(plot): return bool(setting(plot, "manual_trees"))


# has_trees: whether the plot gets tree labels and so stages 07, 07e and 08: its manual trees, else an automatic
# segmentation unless segment_trees = false.
def has_trees(plot): return has_manual_trees(plot) or bool(setting(plot, "segment_trees"))


def riscan_dir(plot):
    """The plot's RiSCAN PRO project: the "riscan" setting, else the one *.RiSCAN folder in <PLOT>/00_raw (None if there is none)."""
    p = setting_path(plot, "riscan")
    if p: return p
    found = sorted(d for d in (ROOT / plot / "00_raw").glob("*.RiSCAN") if d.is_dir())
    if len(found) > 1: raise ValueError(f"{plot}: {len(found)} RiSCAN projects in 00_raw; set riscan in project.toml")
    return found[0] if found else None


_riscan = {}                                                   # riscan_project() per plot, so project.rsp is parsed once per process


def riscan_project(plot):
    """The plot's RiSCAN project parsed by sylva (positions, SOP matrices, scan patterns, .rxp files); None without one.

    Cached per process. Positions whose .rxp RiSCAN PRO did not copy into SCANS get them from the scanner's linked project
    (linked_rxp), so pos.scans and pos.rxp (the first scan) are filled in for both layouts.
    """
    if plot not in _riscan:
        d = riscan_dir(plot)
        if d is None: _riscan[plot] = None
        else:
            import sylva
            proj = sylva.read_riscan_project(str(d))
            for pos in proj:
                if pos.rxp is None:                               # scans linked from the scanner's .PROJ (data source)
                    pos.scans = linked_rxp(d, pos.name); pos.rxp = pos.scans[0] if pos.scans else None
            _riscan[plot] = proj
    return _riscan[plot]


def linked_rxp(riscan, position):
    """The .rxp of a position that RiSCAN PRO links from the scanner's own project instead of copying into SCANS (newer
    scanners: <project>/<data source>.PROJ/<position>.SCNPOS/scans/<scan>.rxp); monitoring and residual files left out."""
    return sorted(f for f in Path(riscan).glob(f"*.PROJ/{position}.SCNPOS/scans/*.rxp")
                  if not f.name.endswith((".mon.rxp", ".residual.rxp")))


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
        # the position is the leading "ScanPosNNN", "tilt NNN" or "tiltNNN" of the export's file name
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
        return {f.stem: np.loadtxt(f) for f in d.glob("*.DAT")}, f"{d} (RiSCAN PRO export)"   # one <position>.DAT per position
    proj = riscan_project(plot)
    if proj is None: raise FileNotFoundError(f"{plot}: no RiSCAN project in 00_raw")
    return {p.name: np.asarray(p.sop, float) for p in proj if p.sop is not None}, f"{riscan_dir(plot) / 'project.rsp'} (SOP)"


def stage(plot, folder):
    """<root>/<PLOT>/<folder>, created if missing."""
    d = ROOT / plot / folder; d.mkdir(parents=True, exist_ok=True); return d


# A progress line prefixed with the time (HH:MM), flushed so it shows at once in the log of a background run.
def log(*a): print(time.strftime("%H:%M"), *a, flush=True)


def jload(path, default=None):
    """The JSON content of path, or default if the file does not exist (manifests and records start empty)."""
    p = Path(path)
    return json.load(open(p)) if p.exists() else default


def jsave(obj, path):
    """Write JSON through a temporary file, so a crash never leaves half a record."""
    p = Path(path); tmp = p.with_name(p.name + ".part")
    # numpy arrays and scalars are written through .tolist(), anything else JSON cannot hold (paths, sets...) as its str()
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))); os.replace(tmp, p)


def plot_name(arg):
    """A plot given by name or as its folder (a path inside the project root)."""
    if os.sep not in arg: return arg
    p = Path(arg).expanduser().resolve()
    if p.parent != ROOT: raise argparse.ArgumentTypeError(f"{arg} is not a plot folder of the project {ROOT}")
    return p.name


def parser(doc, sites=True):
    """An argparse parser whose description is the script's docstring; PLOT arguments (none = every plot) and --project.

    --project is declared so that argparse accepts it and lists it in --help; its value has already been read by _root()
    at import time. sites=False leaves out the PLOT arguments.
    """
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default=None, help="project root holding project.toml (default: $TLS_PROJECT, else the current directory)")
    if sites: p.add_argument("sites", nargs="*", metavar="PLOT", type=plot_name, help="plot names or folders (default: every plot)")
    return p


def check_plots(sites):
    """Refuse plots that have no folder in the project root (usually: the wrong root, e.g. run from the code folder)."""
    missing = [s for s in sites if not (ROOT / s).is_dir()]
    if missing: sys.exit(f"no plot folder {missing} in the project root {ROOT}; give --project=PATH, set TLS_PROJECT, "
                         "run from the project folder, or name the plot by its folder path")


def for_sites(fn, sites, label, *args, **kw):
    """fn(plot, ...) for each plot; a failure is logged to 99_meta/<PLOT>_99_error_<label>.txt and the next plot runs.

    sites empty or None means every plot of the project. Returns True if no plot failed. run.py does not rely on the exit
    status: it checks the step's done() after the script returns.
    """
    ok = True; check_plots(sites or plots())
    for s in sites or plots():
        try: fn(s, *args, **kw)
        except Exception:
            ok = False; err = traceback.format_exc(); print(err, flush=True)
            (stage(s, "99_meta") / f"{s}_99_error_{label}.txt").write_text(err)
    return ok


def matrices(plot):
    """{scan name: 4x4 world_from_scan} the merge uses, and where they came from: the coreg set while the plot's registration
    is "coreg" (a set left from an earlier coreg run is ignored once the setting changes), else 03_transforms."""
    reg = ROOT / plot / "03_registration"
    cd = coreg_dir(plot)
    if cd is not None:                                            # s03 coreg: one <scan>.DAT per scan coreg registered, project frame
        return {f.stem: np.loadtxt(f) for f in cd.glob("*.DAT")}, "03_coreg_dat (sylva.coreg)"
    # 03_transforms.json has one record per stage-01 scan; the ones marked registered = false are left out
    j = json.load(open(reg / f"{plot}_03_transforms.json"))
    return ({x["name"]: np.asarray(x["world_from_scan"], float) for x in j["scans"] if x.get("registered", True)},
            "03_transforms: " + j.get("matrices_from", ""))


def coreg_dir(plot):
    """03_registration/<PLOT>_03_coreg_dat when the plot's registration is "coreg" and stage 03 has written it, else None."""
    d = ROOT / plot / "03_registration" / f"{plot}_03_coreg_dat"
    return d if setting(plot, "registration") == "coreg" and d.exists() else None


def matrices_dir(plot):
    """The folder of .DAT matrices the merge uses: the coreg set (see coreg_dir), else 03_transforms_dat."""
    return coreg_dir(plot) or ROOT / plot / "03_registration" / f"{plot}_03_transforms_dat"


def trees_frame_dir(plot):
    """The .DAT matrices the manual trees are in (manual_trees_frame), else the merge's own."""
    return setting_path(plot, "manual_trees_frame") or matrices_dir(plot)


# The stage-01 LAZ of one scan, scanner frame: 01_scans/<PLOT>_01_<scan>.laz.
def scan_laz(plot, name): return ROOT / plot / "01_scans" / f"{plot}_01_{name}.laz"


def merged_tiles(plot):
    """The tile files (paths as str, sorted) of the stage-04 2 cm registered merge, project frame."""
    import glob
    return sorted(glob.glob(str(ROOT / plot / "04_registered" / f"{plot}_04_merged_2cm_tiles" / "*.laz")))


def load_raster(path):
    """A GeoTIFF as a sylva Raster (row 0 south), nodata as NaN."""
    import rasterio
    from sylva import Raster
    with rasterio.open(path) as r:
        a = r.read(1).astype(np.float64); tr = r.transform
        if r.nodata is not None and not np.isnan(r.nodata): a[a == r.nodata] = np.nan
        # affine: tr.a pixel width, tr.c left edge, tr.f top edge, tr.e pixel height (negative for north-up); square pixels
        # assumed. The bottom edge is the top edge plus every row's (negative) height.
        res = float(tr.a); xmin = float(tr.c); ymin = float(tr.f) + tr.e * a.shape[0]
        if tr.e < 0: a = a[::-1]                              # rasterio: row 0 north; sylva: row 0 south
    return Raster(a, xmin, ymin, res)


# The plot's 1 m DTM from stage 05 (05_ground/<PLOT>_05_dtm_1m.tif) as a sylva Raster, project frame.
def load_dtm(plot): return load_raster(ROOT / plot / "05_ground" / f"{plot}_05_dtm_1m.tif")


def _corners(box):
    """(4, 2) corners, anticlockwise, of a box given as corners, as a dict with corners or xmin..ymax, or as (xmin, ymin,
    xmax, ymax)."""
    if isinstance(box, dict): box = box["corners"] if "corners" in box else [box[k] for k in ("xmin", "ymin", "xmax", "ymax")]
    a = np.asarray(box, float)
    if a.shape == (4,): x0, y0, x1, y1 = a; a = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return a


def plot_box(plot):
    """The (4, 2) corners of the plot box, anticlockwise, project frame: 99_meta/<PLOT>_99_plot_bounds.json (s99_plot.py
    --bounds; a square that may be turned a few degrees), else the rectangle around the scan positions."""
    j = jload(ROOT / plot / "99_meta" / f"{plot}_99_plot_bounds.json")
    if j: return _corners(j)
    o = np.array([m[:3, 3] for m in matrices(plot)[0].values()])   # the scanner positions: translations of world_from_scan
    return _corners((o[:, 0].min(), o[:, 1].min(), o[:, 0].max(), o[:, 1].max()))


def plot_bounds(plot):
    """(xmin, ymin, xmax, ymax) around the plot box: the region to read or segment. Use in_plot for what is inside it."""
    c = plot_box(plot)
    return float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max())


def in_box(corners, x, y):
    """Mask of the points (x, y) inside the convex box with these corners (anticlockwise), edges included."""
    c = _corners(corners); x = np.asarray(x, float); y = np.asarray(y, float); inside = np.ones(np.broadcast(x, y).shape, bool)
    for (ax, ay), (bx, by) in zip(c, np.roll(c, -1, axis=0)):       # left of (or on) every edge
        inside &= (bx - ax) * (y - ay) - (by - ay) * (x - ax) >= -1e-9
    return inside


def in_plot(plot, x, y):
    """Mask of the points (x, y) inside the plot box."""
    return in_box(plot_box(plot), x, y)


def grow_box(corners, d):
    """The corners of a rectangular box (anticlockwise) moved out by d m on every side, along its own sides."""
    c = _corners(corners); e1 = (c[1] - c[0]) / np.linalg.norm(c[1] - c[0]); e2 = (c[3] - c[0]) / np.linalg.norm(c[3] - c[0])
    return np.array([c[0] - d * (e1 + e2), c[1] + d * (e1 - e2), c[2] + d * (e1 + e2), c[3] - d * (e1 - e2)])


def box_area_ha(corners):
    """Area (ha) of a box given by its corners (shoelace formula)."""
    c = _corners(corners); return float(abs(np.dot(c[:, 0], np.roll(c[:, 1], -1)) - np.dot(c[:, 1], np.roll(c[:, 0], -1))) / 2 / 1e4)


def dtm_signature(plot):
    """A short checksum of the plot's 1 m DTM, recorded by the stages that use it so they notice when stage 05 redid it."""
    import hashlib
    return hashlib.sha1(np.ascontiguousarray(load_dtm(plot).data).tobytes()).hexdigest()[:16]


def same_bounds(stored, plot):
    """Whether a plot box stored in a record (corners, a dict with corners or xmin..ymax, or a 4-sequence) is the plot's
    current one (to 1 mm)."""
    if stored is None: return False
    return bool(np.allclose(_corners(stored), plot_box(plot), atol=1e-3))


# Where s07_segfix.py import writes the tree labels edited in segfix (tree_tiles prefers them once they exist).
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
        m = re.findall(r"\d+", f.stem)                         # every number in the name; the last is taken as the tree id
        if f.suffix.lower() in (".ply", ".las", ".laz") and m: out.append((int(m[-1]), f))
    # names that differ only before the last number would give two trees one id: refuse rather than merge them
    if len({t for t, _ in out}) != len(out): raise ValueError(f"{plot}: two manual tree files share a tree id; give manual_trees_table")
    return out


def sylva_version():
    """The installed sylva's version string, "?" if it has none (recorded with every finished step)."""
    import sylva
    return getattr(sylva, "__version__", "?")


def record_run(plot, step, **info):
    """Append one line per finished step to 99_meta/<PLOT>_99_runs.jsonl: what ran, when, with which sylva."""
    line = {"step": step, "finished": time.strftime("%Y-%m-%d %H:%M"), "sylva": sylva_version(), "script": Path(sys.argv[0]).name, **info}
    with open(stage(plot, "99_meta") / f"{plot}_99_runs.jsonl", "a") as f: f.write(json.dumps(line, default=str) + "\n")
