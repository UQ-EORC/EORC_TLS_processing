# TERN TLS processing code

Python scripts that make every stage folder of `TERN_TLS/<SITE>/` from the RiSCAN PRO exports, the raw `.rxp` scans
and the manual trees, using [sylva](https://github.com/UQ-EORC/Sylva). Each script is one step, takes plot names
(none = every plot), skips nothing it is asked to do, and logs failures to `99_meta/<SITE>_99_error_<step>.txt`.
`run.py` runs the steps in order and skips those already done.

```bash
PY=/home/tim/Code/sylva/.venv/bin/python
$PY run.py --dry-run                     # what is done and what would run, every plot
$PY run.py WOMBAT_2023                   # everything still to do for one plot
$PY run.py LITCH_2021 --from=08 --to=99  # a range of steps
$PY s05_ground.py CALP_2021 GWW_2021     # one step, two plots
```

| Step | Script | Makes | Notes |
|---|---|---|---|
| 01 | `s01_convert.py` | `01_scans` | export LAZ + `POINT_FILTERS.txt` (or the plot's override) |
| 03 | `s03_registration.py` | `03_registration` | export PRCS, `sylva.coreg`, or the TERN set: chosen per plot in `tern.SITES` |
| 02 | `s02_shots.py` | `02_shots` | raw `.rxp`, unfiltered, misses rebuilt; runs after 03 because it reads the scan list |
| 04 | `s04_merge.py` | `04_registered` | 2 cm merge, tiles and one LAZ |
| 05 | `s05_ground.py` | `05_ground`, `06_stems`, `90_qc` | DTM, CHM, automatic stems, QC figures |
| 99b | `s99_plot.py --bounds` | `99_meta/*_plot_bounds` | plot box from the field inventory |
| 07 | `s07_trees.py` | `07_trees` | manual trees onto the merge; review list |
| 07e | `s07_segfix.py` | `07_trees/*_segfix`, `*_trees_edited_tiles` | manual editing in segfix (below) |
| 08 | `s08_tree_models.py` | `08_qsm`, `09_leafwood` | raycloudtools and sylva QSMs, leaf/wood, plot meshes |
| 99 | `s99_plot.py` | `*_plot` files, `99_meta/*_plot_clip.json` | clip to the plot box, density and basal area |
| 10 | `s10_canopy.py` | `10_voxels`, `11_canopy` | ray-traced voxels, gap profiles |

`tern.py` holds the plot table (`SITES`: TERN site code, registration choice, manual trees, field inventory) and the
paths. Environment variables move them: `TERN_TLS_ROOT`, `TERN_TREES_ROOT`, `TERN_SCRATCH` (fast local disk for
temporary files, default `~/tern_scratch`), `RCT_SIF` (raycloudtools apptainer image) and `PYARROW_PYTHON` (a Python with
pyarrow for `merge_shots.py`; sylva's environment does not have it). Every finished step appends a line to
`99_meta/<SITE>_99_runs.jsonl` with the date and the sylva version.

## Manual editing in segfix

The transferred tree labels are a starting point; the edited ones are what stages 08, 09 and the plot totals use.

1. `run.py SITE` (or `s07_segfix.py prepare SITE`) writes a segfix project, `07_trees/<SITE>_07_segfix/`: every
   labelled point and the unlabelled points within 15 m of the plot box, all fields kept, with a signed `treeID`
   (> 0 tree, 0 unassigned, -1 noise). It is added to segfix's recent projects; `run.py` stops there for that plot.
2. Open it in `segfix` and work through the trees, starting with the top of `07_trees/<SITE>_07_review.csv` (the trees
   with the most unlabelled canopy inside their crown). `Space` marks a tree done; `Ctrl+S` saves. Segfix offers to
   downsample dense plots for the session; saving still labels every point.
3. `s07_segfix.py status SITE` shows how many trees are done and how much has changed.
4. `s07_segfix.py import SITE` writes the edited labels as `07_trees/<SITE>_07_trees_edited_tiles/` and
   `<SITE>_07_segfix_import.json` (trees added, removed and changed). It refuses while trees are not marked done
   (`--allow-unfinished` overrides).
5. `run.py SITE` again: the review is redone on the edited labels, stage 08 redoes only the trees whose points changed
   (and drops removed trees), and the plot totals are redone.

Editing again later: open the same project, save, import again, rerun. Plots without manual trees (Alice, Cumberland
Plain, Warra) have no labels to start from: segment them first (`sylva trees --segment` on a height-normalised cloud)
and open that file in segfix.

## Requirements

sylva (its `.venv` at `/home/tim/Code/sylva`), laspy, rasterio, scipy, matplotlib; apptainer and the raycloudtools
image for the rct QSMs; segfix for editing (`pip install segfix`, Python 3.10-3.12). Memory: stage 04 and 05 take
10-30 GB per plot, a stage-02 worker up to 12.5 GB (Warra), tracing in stage 10 is capped at 20 GB.

## History

These scripts replace the ones in `/home/tim/tumba_work` that made the outputs of October 2026 (`convert_exports.py`,
`build_transforms_from_exports.py`, `coreg_exports.py`, `report_exports.py`, `merge_site.py`, `ground_stems_from_merge.py`,
`transfer_trees.py`, `review_flags.py`, `tree_models.py`, `plot_meshes.py`, `convert_shots_unfiltered.py`,
`canopy_stages.py`, `plot_bounds_inventory.py`, `clip_to_plot.py`). The methods and settings are the same; the scan
lists of 01, 02 and 03 and the stage-03 matrices were checked to reproduce the existing outputs exactly. Those old scripts
use the stage numbers from before 2026-10-29 and no longer match the folders.
