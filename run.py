"""Run the processing for one or more plots, step by step, skipping the steps already done.

usage: run.py [PLOT ...] [--from=STEP] [--to=STEP] [--dry-run] [--skip-editing]

Steps, in order (each runs `python <script> PLOT` in its own process, so memory is freed between them):
  01  s01_convert.py        RiSCAN project scans (.rdbx) -> filtered scans
  03  s03_registration.py   matrices (export / coreg / previous set), registration report
  02  s02_shots.py          unfiltered pulses from the raw .rxp (needs the stage-03 scan list)
  04  s04_merge.py          2 cm registered merge
  05  s05_ground.py         ground, DTM, CHM (05), automatic stems (06), QC figures
  99b s99_plot.py --bounds  plot box from the field inventory (if not written yet)
  07  s07_trees.py          tree labels on the merge: manual trees transferred, else automatic segmentation; review list
  07e s07_segfix.py         manual editing in segfix: the project is prepared, then the run stops for that plot until the
                            edit is imported (s07_segfix.py import PLOT). --skip-editing goes on with the transferred labels.
  08  s08_tree_models.py    QSMs (08), leaf/wood (09), plot meshes; redoes the trees changed in segfix
  99  s99_plot.py           products clipped to the plot box, density and basal area from the segmented trees
  10  s10_canopy.py         voxels (10), gap profiles (11)
A step that fails stops that plot (see 99_meta/<PLOT>_99_error_<step>.txt); the other plots go on.
"""
import subprocess, sys, time
import project
from project import ROOT, log
# the stage modules are imported only for their done() checks (and s07_segfix's helpers); the steps themselves run as subprocesses
import s01_convert, s02_shots, s03_registration, s04_merge, s05_ground, s07_trees, s07_segfix, s08_tree_models, s10_canopy, s99_plot


# Predicates for STEPS' "applies" column: plots with tree labels (manual trees, or automatic segmentation unless segment_trees = false) get 07, 07e and 08; only plots with .rxp get 02 and 10.
def has_trees(site): return bool(project.has_trees(site))
def has_rxp(site): return project.has_rxp(site)   # no unpacked RiSCAN project with its .rxp scans: no pulses


# The pipeline in run order. done(site) tells whether the step's outputs are complete (it is also checked after the step
# ran, so a script that exits 0 without finishing counts as failed); applies(site) whether the step is meant for the plot.
# 02 comes after 03 because it takes its scan list from 03_transforms.json. 07e has no script: it is handled in run_site.
STEPS = [  # (id, script and extra arguments, done(site), applies(site))
    ("01", ["s01_convert.py"], s01_convert.done, lambda s: True),
    ("03", ["s03_registration.py"], s03_registration.done, lambda s: True),
    ("02", ["s02_shots.py"], s02_shots.done, has_rxp),
    ("04", ["s04_merge.py"], s04_merge.done, lambda s: True),
    ("05", ["s05_ground.py"], s05_ground.done, lambda s: True),
    ("99b", ["s99_plot.py", "--bounds"], lambda s: (ROOT / s / "99_meta" / f"{s}_99_plot_bounds.json").exists(), lambda s: True),
    ("07", ["s07_trees.py"], s07_trees.done, has_trees),
    ("07e", None, s07_segfix.done, has_trees),
    ("08", ["s08_tree_models.py"], s08_tree_models.done, has_trees),
    ("99", ["s99_plot.py"], s99_plot.done, lambda s: True),
    ("10", ["s10_canopy.py"], s10_canopy.done, has_rxp),
]
ORDER = [s[0] for s in STEPS]                                    # step ids in run order: the choices of --from and --to


def run_site(site, first, last, dry, skip_editing):
    """Run the steps first..last (ids, inclusive, in STEPS order) that apply to one plot and are not done yet.

    Each step runs as `python <script> --project=ROOT PLOT` in its own process. The plot stops at the first failure (non-zero
    exit or done() still false afterwards) and at the manual editing step 07e until its edit is imported; later steps of
    that plot are left for the next run. dry only logs what would run. skip_editing passes over 07e with the transferred
    tree labels. Returns None.
    """
    for sid, cmd, done, applies in STEPS[ORDER.index(first):ORDER.index(last) + 1]:
        if not applies(site): continue
        if done(site): log(f"{site} {sid}: done"); continue
        if sid == "07e":
            # manual editing: prepare the segfix project once, then stop this plot until `s07_segfix.py import` makes done() true
            if skip_editing: log(f"{site} 07e: not edited in segfix; going on with the transferred labels (--skip-editing)"); continue
            if not s07_segfix.cloud_path(site).exists():
                if dry: log(f"{site} 07e: would prepare the segfix project and stop"); return
                s07_segfix.prepare(site)
            log(f"{site} 07e: waiting for manual editing in segfix ({s07_segfix.project_dir(site)}); "
                f"then `python s07_segfix.py import {site}` and run again"); return
        if dry: log(f"{site} {sid}: would run {' '.join(cmd)}"); continue
        log(f"{site} {sid}: {' '.join(cmd)}"); t0 = time.time()
        # the same interpreter and an explicit --project, so the child resolves the same root whatever its cwd or environment
        r = subprocess.run([sys.executable, str(project.CODE / cmd[0]), *cmd[1:], f"--project={project.ROOT}", site])
        if r.returncode != 0 or not done(site):
            log(f"{site} {sid}: FAILED after {(time.time() - t0) / 60:.0f} min; see 99_meta/{site}_99_error_*.txt"); return
        log(f"{site} {sid}: finished in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    p = project.parser(__doc__); p.add_argument("--from", dest="first", default="01", choices=ORDER); p.add_argument("--to", dest="last", default="10", choices=ORDER)
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--skip-editing", action="store_true"); a = p.parse_args()
    project.check_plots(a.sites or list(project.plots()))
    # plots one after the other; a plot that stops (failure or editing) does not stop the next one
    for s in a.sites or list(project.plots()): run_site(s, a.first, a.last, a.dry_run, a.skip_editing)
