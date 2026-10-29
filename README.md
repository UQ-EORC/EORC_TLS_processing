# TLS plot processing

Python scripts that turn RIEGL terrestrial laser scanning of forest plots into registered clouds, terrain and canopy
height models, tree segmentations corrected by hand, per-tree structure models, and canopy profiles, using
[sylva](https://github.com/UQ-EORC/Sylva) and, for the manual editing step, [segfix](https://github.com/UQ-EORC/Segfix).

Nothing here is tied to a dataset: a project is a folder holding one folder per plot and a `project.toml` that says
where the inputs are and how each plot differs from the defaults (see `project.example.toml`).

## Inputs

- RiSCAN PRO per-scan LAZ exports in scanner coordinates, with the project's `PRCS` matrices (`.DAT`) and a
  `POINT_FILTERS.txt`, one folder per plot (`<exports>/<PLOT>/`).
- The RiSCAN PRO project in `<PLOT>/00_raw/` (the `.rxp` scans), for the unfiltered pulses of stage 02 and for coreg.
- Optional: manually segmented trees (one point cloud per tree) and a field stem map per plot.

## Running

```bash
export TLS_PROJECT=/path/to/project      # or --project=PATH on any script, or run from the project folder
python run.py --dry-run                  # what is done and what would run, every plot
python run.py PLOT_A_2025                # everything still to do for one plot
python run.py PLOT_A_2025 --from=08 --to=99
python s05_ground.py PLOT_A_2025 PLOT_B_2025   # one step, two plots
```

Each script is one step, takes plot names (none = every plot in `project.toml`), and logs a failure to
`<PLOT>/99_meta/<PLOT>_99_error_<step>.txt` without stopping the other plots. `run.py` runs the steps in order, each in
its own process, and skips the ones already done. Every finished step adds a line to `99_meta/<PLOT>_99_runs.jsonl`
with the date and the sylva version.

| Step | Script | Writes | What |
|---|---|---|---|
| 01 | `s01_convert.py` | `01_scans` | export LAZ with the point filter applied, scanner frame |
| 03 | `s03_registration.py` | `03_registration` | export PRCS, `sylva.coreg` from them, or a previous set (per plot); registration report |
| 02 | `s02_shots.py` | `02_shots` | every pulse from the raw `.rxp`, unfiltered, misses rebuilt (runs after 03: it reads the scan list) |
| 04 | `s04_merge.py` | `04_registered` | 2 cm voxel-thinned merge, tiles and one LAZ |
| 05 | `s05_ground.py` | `05_ground`, `06_stems`, `90_qc` | DTM, CHM, automatic stems, QC figures |
| 99b | `s99_plot.py --bounds` | `99_meta/*_plot_bounds` | plot box from the field stem map |
| 07 | `s07_trees.py` | `07_trees` | manual trees carried onto the merge; review list |
| 07e | `s07_segfix.py` | `07_trees/*_segfix`, `*_trees_edited_tiles` | manual editing in segfix (below) |
| 08 | `s08_tree_models.py` | `08_qsm`, `09_leafwood` | raycloudtools and sylva QSMs, leaf/wood, plot meshes |
| 99 | `s99_plot.py` | `*_plot` files, `99_meta/*_plot_clip.json` | products clipped to the plot box; density and basal area |
| 10 | `s10_canopy.py` | `10_voxels`, `11_canopy` | ray-traced voxels, gap-fraction profiles |

Stage folders sit in `<project>/<PLOT>/`, files are named `<PLOT>_<stage>_<what>`: `00_raw`, `01_scans`, `02_shots`,
`03_registration`, `04_registered`, `05_ground`, `06_stems`, `07_trees`, `08_qsm`, `09_leafwood`, `10_voxels`,
`11_canopy`, `90_qc`, `99_meta`.

Notes on the methods:
- The canopy stages use every pulse. A point filter (reflectance floor, deviation cap) removes pulses whose echoes all
  fail it, which overstates gaps and understates plant area, so stage 02 reads the raw scans without one.
- The automatic stems of stage 06 over-count and inflate DBH; plot density and basal area come from the manual trees.
- The gap-fraction PAI is not meaningful for low, sparse canopy and saturates in dense canopy; use the voxel PAI there.

## Manual editing in segfix

The transferred tree labels are a starting point; the edited ones are what stages 08, 09 and the plot totals use.

1. `run.py PLOT` (or `s07_segfix.py prepare PLOT`) writes a segfix project, `07_trees/<PLOT>_07_segfix/`: every
   labelled point and the unlabelled points within 15 m of the plot box, all fields kept, with a signed `treeID`
   (> 0 tree, 0 unassigned, -1 noise). It is added to segfix's recent projects; `run.py` stops there for that plot.
2. Open it in `segfix` and work through the trees, starting with the top of `07_trees/<PLOT>_07_review.csv` (the trees
   with the most unlabelled canopy inside their crown). `Space` marks a tree done; `Ctrl+S` saves. Segfix offers to
   downsample dense plots for the session; saving still labels every point.
3. `s07_segfix.py status PLOT` shows how many trees are done and how much has changed.
4. `s07_segfix.py import PLOT` writes the edited labels as `07_trees/<PLOT>_07_trees_edited_tiles/` and
   `<PLOT>_07_segfix_import.json` (trees added, removed and changed). It refuses while trees are not marked done
   (`--allow-unfinished` overrides).
5. `run.py PLOT` again: the review is redone on the edited labels, stage 08 redoes only the trees whose points changed
   (and drops removed trees), and the plot totals are redone.

Plots without manual trees have no labels to start from: segment them first (for example `sylva trees --segment` on a
height-normalised cloud) and open that file in segfix.

## Requirements

Python 3.11+, sylva, laspy, rasterio, scipy, matplotlib; apptainer and a raycloudtools image for the raycloudtools
QSMs; a Python with pyarrow for `merge_shots.py`; segfix for editing (`pip install segfix`, Python 3.10-3.12, its own
environment). Memory: stages 04 and 05 take 10-30 GB per plot, a stage-02 worker can take over 12 GB on dense scans,
the stage-10 tracing is capped at 20 GB.
