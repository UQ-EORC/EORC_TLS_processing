# TLS plot processing

Python scripts that turn RIEGL terrestrial laser scanning of forest plots into registered clouds, terrain and canopy
height models, tree segmentations corrected by hand, per-tree structure models, and canopy profiles, using
[sylva](https://github.com/UQ-EORC/Sylva) and, for the manual editing step, [segfix](https://github.com/UQ-EORC/Segfix).

Nothing here is tied to a dataset: a project is a folder holding one folder per plot and a `project.toml` that says
where the inputs are and how each plot differs from the defaults (see `project.example.toml`).

## Inputs

- The RiSCAN PRO project of each plot, unpacked in `<PLOT>/00_raw/` (one `*.RiSCAN` folder), with:
  - `project.rsp`: the scan positions, their SOP matrices (scanner to project frame) and scan patterns;
  - `project.rdb/SCANS/<position>/SINGLESCANS/<scan>/<scan>.rdbx`: each scan as RiSCAN PRO imported it (echoes after
    MTA resolution), read by stage 01. Reading `.rdbx` needs RIEGL's `librdb` (RDB SDK), see `sylva.io.find_rdblib`;
  - `SCANS/<position>/SINGLESCANS/<scan>.rxp`, or for newer scanners the scanner's own project that RiSCAN PRO links
    them from (`<project>/<name>.PROJ/<position>.SCNPOS/scans/<scan>.rxp`): the raw scans, read by stage 02 for every
    pulse including the misses. Without them a plot gets no stages 02, 10 and 11.
- Optional: a point filter file (`filter` in `project.toml`), manually segmented trees (one point cloud per tree) and a
  field stem map per plot.

Plots processed before this layout can read per-scan RiSCAN PRO LAZ exports with their `PRCS` matrices and
`POINT_FILTERS.txt` instead (`source = "exports"`, files in `<exports>/<PLOT>/`).

## Running

```bash
export TLS_PROJECT=/path/to/project      # or --project=PATH on any script, or run from the project folder
python run.py --dry-run                  # what is done and what would run, every plot
python run.py PLOT_A_2025                # everything still to do for one plot
python run.py /path/to/project/PLOT_A_2025   # the same; a plot given as a folder sets the project root
python run.py PLOT_A_2025 --from=08 --to=99
python s05_ground.py PLOT_A_2025 PLOT_B_2025   # one step, two plots
```

Each script is one step, takes plot names (none = every plot in `project.toml`), and logs a failure to
`<PLOT>/99_meta/<PLOT>_99_error_<step>.txt` without stopping the other plots. `run.py` runs the steps in order, each in
its own process, and skips the ones already done. Every finished step adds a line to `99_meta/<PLOT>_99_runs.jsonl`
with the date and the sylva version.

| Step | Script | Writes | What |
|---|---|---|---|
| 01 | `s01_convert.py` | `01_scans` | each scan's `.rdbx` with the point filter applied, scanner frame |
| 03 | `s03_registration.py` | `03_registration` | the project's SOPs, `sylva.coreg` from them, or a previous set (per plot); registration report |
| 02 | `s02_shots.py` | `02_shots` | every pulse from the raw `.rxp`, unfiltered, misses rebuilt (runs after 03: it reads the scan list) |
| 04 | `s04_merge.py` | `04_registered` | 2 cm voxel-thinned merge, tiles and one LAZ |
| 05 | `s05_ground.py` | `05_ground`, `06_stems`, `90_qc` | DTM, CHM, automatic stems, QC figures |
| 99b | `s99_plot.py --bounds` | `99_meta/*_plot_bounds` | plot box from the field stem map |
| 07 | `s07_trees.py` | `07_trees` | tree labels: manual trees carried onto the merge, else automatic segmentation from the 06 stems; review list |
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
- The automatic stems of stage 06 over-count and inflate DBH; plot density and basal area come from the segmented trees
  after editing, with DBH from their QSMs.
- The gap-fraction PAI is not meaningful for low, sparse canopy and saturates in dense canopy; use the voxel PAI there.

## Manual editing in segfix

The transferred tree labels are a starting point; the edited ones are what stages 08, 09 and the plot totals use.

1. `run.py PLOT` (or `s07_segfix.py prepare PLOT`) writes a segfix project, `07_trees/<PLOT>_07_segfix/`: every
   labelled point and the unlabelled points within 15 m of the plot box, all fields kept, with a signed `treeID`
   (> 0 tree, 0 unassigned, -1 noise), and a leaf/wood class field `leafwood` (0 unclassified, 1 leaf, 2 wood) filled
   from the per-tree classification of stage 09 where it has run. It is added to segfix's recent projects, opens with
   `leafwood` as its point-class field, and `run.py` stops there for that plot. A project prepared before the class
   field existed gets it with `s07_segfix.py add-classes PLOT` (tree edits and the Done list are kept; close segfix first).
2. Open it in `segfix` and work through the trees, starting with the top of `07_trees/<PLOT>_07_review.csv` (the trees
   with the most unlabelled canopy inside their crown). `Space` marks a tree done; `Ctrl+S` saves. Segfix offers to
   downsample dense plots for the session; saving still labels every point. Leaf and wood are corrected in the same
   pass: `Shift+F` colours by class, `Ctrl+1`...`Ctrl+3` give the selection (or the whole tree) a class.
3. `s07_segfix.py status PLOT` shows how many trees are done and how much has changed.
4. `s07_segfix.py import PLOT` writes the edited labels as `07_trees/<PLOT>_07_trees_edited_tiles/` and
   `<PLOT>_07_segfix_import.json` (trees added, removed and changed, points reclassified). It refuses while trees are not marked done
   (`--allow-unfinished` overrides).
5. `run.py PLOT` again: the review is redone on the edited labels, stage 08 redoes the models of trees whose points
   changed and the leaf/wood split, leaf metrics and sylva QSM of trees whose classes changed (points classed by hand
   keep their class; the classifier fills in only the unclassified ones), drops removed trees, and redoes the plot totals.

Plots without manual trees start from an automatic segmentation (stage 07, `sylva.trees`): the stage-06 stems are grown
into trees over the merge tiles within the plot box plus 15 m (`sylva.als.tiles.segment_trees`, tile by tile), branches
of another stem merged and candidates under 3 m dropped. The stem finder over-counts and splits
trees, so expect more editing than for transferred manual trees: merge split crowns, delete false stems (shrubs, logs),
and check the trees the review list puts first. `segment_trees = false` in `project.toml` gives a plot no tree stages.

## Install

With conda (or mamba), from this folder:

```bash
conda env create -f environment.yml      # Python 3.12, sylva, segfix, laspy, rasterio, scipy, matplotlib, pyarrow
conda activate tls
python run.py --help
```

`environment.yml` installs everything the scripts import, segfix included, so the editing step runs from the same
environment (`segfix` opens the GUI). To update later: `conda env update -f environment.yml --prune`.

Three things come from outside conda:

- **RIEGL libraries** (proprietary, not redistributable; from RIEGL's members area): the RDB SDK (`librdb`) to read
  `.rdbx` in stage 01, and RiVLib (`libscanifc`) to read `.rxp` in stage 02. Unpack them in `~/.local/lib` (for example
  `~/.local/lib/rdblib-2.4.1-x86_64-linux/` and `~/.local/lib/rivlib-2_6_0-x86_64-linux-gcc11/`), where sylva finds them,
  or point `RDBLIB_PATH` and `RIVLIB_PATH` at them. Check with
  `python -c "import sylva.io as io; print(io.find_rdblib(), io.find_rivlib())"`.
- **raycloudtools** for the raycloudtools QSMs of stage 08: apptainer and an image with `rayextract`, given as
  `raycloudtools_sif` in `project.toml` (or `$RCT_SIF`). Without it stage 08 makes the sylva QSMs only.
- **The project folder**: a `project.toml` in the folder that holds the plots (copy `project.example.toml`), then
  `export TLS_PROJECT=/path/to/project` or pass `--project=PATH`.

Memory: stages 04 and 05 take 10-30 GB per plot, a stage-02 worker can take over 12 GB on dense scans, the stage-10
tracing is capped at 20 GB.
