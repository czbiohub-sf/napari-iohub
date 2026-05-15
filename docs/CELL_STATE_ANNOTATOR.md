# Cell State Annotator Tutorial

The Cell State Annotator is an interactive napari widget for annotating cell
states in time-lapse microscopy data. It supports two save modes:

- **Track-mapped CSV** (default): clicked centroids are matched to track IDs
  via segmentation lookup and written in an ultrack-compatible schema.
- **Centroid-only CSV**: when no tracking dataset is provided, the annotator
  writes a minimal CSV with one row per clicked point. Track IDs can be
  assigned later by re-mapping against any `tracking.zarr` (see
  *Centroid-only mode* below).

The four annotated state columns are:

- **cell_division_state**: `mitosis` vs `interphase`
- **infection_state**: `infected` vs `uninfected` (sticky-forward from first
  mark)
- **organelle_state**: `remodel` vs `noremodel`
- **cell_death_state**: `dead` vs `alive` (sticky-forward; overrides other
  state columns once a cell is dead)

## Launching the Annotator

### From the command line

A `napari-iohub-annotate` console script is installed with the package.

```bash
# Activate via uv (recommended; uses uv.lock)
uv sync --all-extras
uv run napari-iohub-annotate \
  -i /path/to/images.zarr \
  -t /path/to/tracks.zarr \
  -f A/1/0 \
  -o /path/to/output_dir

# Or activate the venv first
source .venv/bin/activate
napari-iohub-annotate -i ... -t ... -f A/1/0 -o ...
```

#### Flags

| Flag | Meaning |
|------|---------|
| `-i`, `--images PATH` | Images OME-Zarr (HCS plate). |
| `-t`, `--tracks PATH` | Tracks OME-Zarr (HCS plate with segmentation labels). Omit to enter **centroid-only mode**. |
| `-f`, `--fov ROW/WELL/FOV` | FOV to load (e.g. `A/1/000000`). |
| `-r`, `--resume-csv PATH` | Resume from an existing annotation CSV. Auto-detects whether the CSV is track-mapped or centroid-only. |
| `-o`, `--output PATH` | Folder to save annotation CSVs into. |
| `--project-2d` | Project images to 2D around the FOV's in-focus Z (read from `focus_slice.Phase3D.fov_statistics.z_focus_mean`). Phase channels are sliced at the focus Z; fluorescence channels are max-projected over `[focus-5, focus+10]`. Warns and falls back to the full Z-stack if no focus metadata is present. |
| `--no-expand-labels` | Don't repeat single-plane labels along Z to match the image Z-stack. Implicit when `--project-2d` is set. |
| `--no-load` | Pre-fill the fields but do not auto-load. |
| `-v`, `--verbose` | Debug logging. |

### From the napari GUI

1. Launch napari.
2. `Plugins` → `napari-iohub` → `Cell state annotation`.
3. Use the *Browse images / tracks / Resume CSV / Output folder* buttons (each
   opens at the path currently in the line-edit, so paste a path first if you
   want the dialog to start there).

### From Python

```python
import napari
from napari_iohub._cell_state_annotator import CellStateAnnotatorWidget

viewer = napari.Viewer()
widget = CellStateAnnotatorWidget(viewer)
viewer.window.add_dock_widget(widget, name="Cell state annotation")

widget.prefill(
    images="/path/to/images.zarr",
    tracks="/path/to/tracks.zarr",
    fov="A/1/0",
    output="/path/to/output",
    project_2d=True,
    load=True,
)
napari.run()
```

## Workflow

### 1. Load Your Data

1. **Browse images**: select an OME-Zarr dataset (NGFF v0.4 / zarr v2 or NGFF
   v0.5 / zarr v3 — both work).
2. **Browse tracks** *(optional)*: select an OME-Zarr dataset containing
   segmentation labels. Omitting this enters centroid-only mode.
3. **Browse Resume CSV** *(optional)*: pick an existing annotation CSV to
   continue from.
4. Pick **Row / Well / FOV** from the dropdowns. (The dropdowns are populated
   from the tracks dataset when available, otherwise from the images dataset.)
5. *(Optional)* Tick **Project to 2D around in-focus Z** and/or untick
   **Expand labels Z to match image** before loading.
6. Click **Load Data**.

### 2. Annotate

Four annotation layers are created, each with a distinct color:

| Layer | Color | State column | Mark value |
|-------|-------|--------------|------------|
| `_mitosis_events`  | Blue   | `cell_division_state` | `mitosis`  |
| `_infected_events` | Orange | `infection_state`     | `infected` |
| `_remodel_events`  | Purple | `organelle_state`     | `remodel`  |
| `_death_events`    | Red    | `cell_death_state`    | `dead`     |

To annotate:

1. Select an annotation layer (use `q`/`e` to cycle).
2. Navigate timepoints (`a` / `d`).
3. Click on a cell.

### 3. Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `a` | Step backward in time |
| `d` | Step forward in time |
| `q` | Cycle to previous annotation layer |
| `e` | Cycle to next annotation layer |
| `r` | Toggle interpolation mode |
| `s` | Save annotations |

### 4. Interpolation Mode

For events that span multiple consecutive frames:

1. Click on the cell at the **start** frame.
2. Press `r` to enable interpolation mode.
3. Navigate to the **end** frame.
4. Click on the cell again.

The widget inserts annotation points at every intermediate frame using linear
interpolation between the two centroids.

### 5. Save Annotations

Click **Save Annotations** (or press `s`). The output file location depends on
mode:

- **Resuming**: overwrites the resume CSV. A timestamped copy is kept under
  `<output>/backup/` (only the latest 5 are retained per base filename).
- **Fresh, track-mapped**: writes `<output>/<tracks_zarr_stem>_<row>_<well>_<fov>.csv`.
- **Fresh, centroid-only**: writes `<output>/<images_zarr_stem>_<row>_<well>_<fov>_centroids.csv`.

`annotation_version` starts at `1.0` and increments on every resume save
(`1.0 → 2.0 → 3.0 → ...`).

## 2D Projection Mode

When **Project to 2D around in-focus Z** is on, the annotator reads
`focus_slice.Phase3D.fov_statistics.z_focus_mean` from the FOV's zattrs and
applies a per-channel projection:

- **Phase channels** (channel name starts with `phase`, case-insensitive — so
  `Phase3D`, `Phase3D_recon`, etc. all qualify): a single Z slice at
  `z_focus_mean`.
- **Other channels** (fluorescence): max-projection over
  `[z_focus_mean - 5, z_focus_mean + 10]` (15 slices, clipped to the valid Z
  range).

In this mode:

- Annotation point layers become 3D (`(T, Y, X)`).
- The "Expand labels Z" toggle is disabled and ignored.
- The image and the labels layer share the same Y/X shape; a warning is logged
  if they don't.

If the FOV has no `focus_slice` metadata, a warning is shown and the annotator
falls back to the full Z-stack with 4D annotation points (`(T, Z, Y, X)`).

## Centroid-only Mode

Omit `--tracks` (or leave the Tracks field blank) to enter centroid-only mode:

- No segmentation labels layer is loaded.
- No tracks-lineage overlay is loaded.
- Image FOVs are populated directly from the images dataset.
- On save, the CSV has **no `track_id` / `parent_track_id` / `id` columns**.
  Each clicked point is one row.

### Centroid-only CSV schema

| Column | Description |
|--------|-------------|
| `fov_name` | FOV path (e.g. `A/1/000000`) |
| `t` | Timepoint |
| `z`, `y`, `x` | Click coordinates (z is the focus Z in 2D mode, else the napari z) |
| `cell_division_state` | `mitosis` or null |
| `infection_state` | `infected` or null |
| `organelle_state` | `remodel` or null |
| `cell_death_state` | `dead` or null |
| `annotator` | Username |
| `annotation_date` | ISO timestamp |
| `annotation_version` | Schema version (`1.0`, `2.0`, ...) |

Exactly one state column is non-null per row (the one for the layer the point
came from). To assign track IDs against a `tracking.zarr`, use the
`napari-iohub-map-centroids` CLI described below.

## Mapping centroids onto a tracking.zarr

`napari-iohub-map-centroids` takes a CSV (centroid-only OR a previously
track-mapped CSV) and a `tracking.zarr`, and emits an ultrack-compatible
track-mapped CSV.

```bash
napari-iohub-map-centroids \
  -c /path/to/centroid_or_old.csv \
  -t /path/to/tracking.zarr \
  -f A/1/000000 \
  -o /path/to/mapped.csv \
  [--kernel 2]
```

### Flags

| Flag | Meaning |
|------|---------|
| `-c`, `--csv PATH` | Input CSV (centroid-only or previously track-mapped). |
| `-t`, `--tracks PATH` | `tracking.zarr` (HCS plate with segmentation labels). |
| `-f`, `--fov ROW/WELL/FOV` | FOV to map (required; only this FOV's rows are processed). |
| `-o`, `--output PATH` | Output CSV path. |
| `--kernel N` | Half-size of the lookup window. Default `2` → 5×5 px. |
| `-v`, `--verbose` | Debug logging. |

### How mapping works

For each clicked centroid (auto-detected as: every annotated row for
per-frame columns `cell_division_state` / `organelle_state`; first
occurrence per old `track_id` for the sticky-forward columns
`infection_state` / `cell_death_state`; or every row when the input CSV
has no `track_id`):

1. Open the labels at `tracking.zarr[fov]["0"][t, 0, 0, y-k:y+k+1, x-k:x+k+1]`.
2. Take the **mode of the non-zero label values** in that window.
3. That value becomes the row's `track_id` in the output.

After all clicks are assigned, the events are expanded across the full
lineage in the FOV's `tracks_<fov>.csv` so the output schema matches
exactly what the annotator's track-mapped save produces.

### Background hits

If a kernel is entirely zero-labeled (the click landed on background),
the row is **dropped from the main output** and written to a sibling
file `<output_stem>_unmapped.csv` with the offending `(t, y, x)` and the
kernel size used. A summary line at the end of the CLI run prints how
many were mapped vs unmapped.

### Re-mapping after re-tracking

To carry annotations forward to a new tracking run:

```bash
napari-iohub-map-centroids \
  -c old_combined_annotations.csv \
  -t new_tracking.zarr \
  -f C/2/000000 \
  -o new_combined_annotations_C_2_000000.csv
```

The CLI ignores any `track_id` / `id` / `parent_track_id` columns in the
input and re-derives them from the new tracking by kernel lookup. Expect
a few percent drift in per-row state when re-tracking changes which
label sits under each click — small mis-assignments of sticky-forward
states (infection, death) get amplified by their propagation.

## Track-mapped CSV schema (default)

| Column | Description |
|--------|-------------|
| `fov_name` | FOV path |
| `track_id`, `id`, `parent_track_id`, `parent_id` | ultrack lineage columns |
| `t`, `z`, `y`, `x` | Timepoint + spatial coords |
| `cell_division_state` | `mitosis` or `interphase` |
| `infection_state` | `infected`, `uninfected`, or null |
| `organelle_state` | `remodel` or `noremodel` |
| `cell_death_state` | `dead`, `alive`, or null |
| `annotator`, `annotation_date`, `annotation_version` | Metadata |

### Annotation logic (track-mapped only)

- **Mitosis**: each annotated frame stays `mitosis`; all others are `interphase`.
- **Infection**: a row is `infected` for `t ≥ first_infection_t`, `uninfected`
  before. If a track has no infection annotation, the column is null for that
  track.
- **Organelle remodel**: each annotated frame stays `remodel`; all others are
  `noremodel`.
- **Cell death**: a row is `dead` for `t ≥ first_death_t`; once dead, all
  other state columns are set to null for that row.

## Resume

Pass `--resume-csv` (or use the *Browse Resume CSV* button). The annotator
auto-detects which schema the CSV uses:

- **Track-mapped CSV**: mitosis/remodel populate one point per annotated frame;
  infection/death populate one point per track at the first occurrence.
- **Centroid-only CSV**: every row becomes one point in its respective layer.

In both cases the points re-appear in the four annotation layers and you can
continue editing.

## Tips

- The status bar at the bottom shows the current state and any errors/warnings.
- In track-mapped mode, the centroid → label lookup uses a 10-pixel window
  around the clicked coordinate. Clicks landing on background log a warning.
- `Browse images / tracks / Resume CSV` dialogs open at the path currently in
  the line-edit — paste a path and click Browse to start the dialog there.
- For zarr v3 / NGFF v0.5 stores you need `iohub>=0.3.0`. The lockfile in this
  repo pins compatible versions.
