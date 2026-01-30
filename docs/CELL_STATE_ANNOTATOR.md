# Cell State Annotator Tutorial

The Cell State Annotator is an interactive napari widget for annotating cell states in time-lapse microscopy data. It exports annotations to ultrack-compatible CSV format.

It supports annotation of:

- **cell_division_state**: Track cells undergoing mitosis (`mitosis`) vs cells in interphase (`interphase`)
- **infection_state**: Mark cells as `infected` or `uninfected` (e.g., viral infection experiments)
- **organelle_state**: Annotate organelle remodeling events (`remodel` vs `noremodel`)
- **cell_death_state**: Track cell death events (`dead` vs `alive`) 

## Launching the Widget

### From the command line

```bash
napari --plugin napari-iohub "Cell state annotation"
```

### From Python

```python
import napari

viewer = napari.Viewer()
viewer.window.add_plugin_dock_widget("napari-iohub", "Cell state annotation")
napari.run()
```

### From the napari GUI

1. Launch napari
2. Go to `Plugins` → `napari-iohub` → `Cell state annotation`


## Workflow

### 1. Load Your Data

1. **Browse images**: Click "Browse images" and select your OME-Zarr dataset containing the raw images.

2. **Browse tracks**: Click "Browse tracks" and select your OME-Zarr dataset containing tracking labels. This will populate the Row/Well/FOV navigation dropdowns.

3. **Select FOV**: Use the Row, Well, and FOV dropdowns to navigate to the field of view you want to annotate.

4. **Load Data**: Click "Load Data" to load the images, tracking labels, and set up annotation layers.

### 2. Annotate Cell States

The widget creates four annotation layers, each with a distinct color:

| Layer | Color | Event Type |
|-------|-------|------------|
| `_mitosis_events` | Blue | Cell division |
| `_infected_events` | Orange | Infection |
| `_remodel_events` | Purple | Organelle remodeling |
| `_death_events` | Red | Cell death |

To annotate:
1. Select the appropriate annotation layer (use `q`/`e` to cycle)
2. Navigate to the timepoint of interest (use `a`/`d` to step through time)
3. Click on a cell to add an annotation point

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

For events that span multiple timepoints (e.g., a cell undergoing mitosis over several frames):

1. Click on the cell at the **start** timepoint
2. Press `r` to enable interpolation mode
3. Navigate to the **end** timepoint
4. Click on the cell again

The widget will automatically add annotation points for all intermediate timepoints with linear interpolation.

### 5. Save Annotations

1. **Set output folder**: Click "Output folder" to choose where to save the CSV files (defaults to the tracks dataset location).

2. **Save**: Click "Save Annotations" or press `s`.

The widget saves two CSV files:
- A timestamped version: `annotations_{fov_name}_{username}_{timestamp}.csv`
- A canonical version: `annotations_{fov_name}.csv`

## Output Format

The CSV output is ultrack-compatible with the following columns:

| Column | Description |
|--------|-------------|
| `fov_name` | Field of view identifier |
| `track_id` | Cell track identifier |
| `t` | Timepoint |
| `z`, `y`, `x` | Spatial coordinates |
| `cell_division_state` | `mitosis` or `interphase` |
| `infection_state` | `infected`, `uninfected`, or `null` |
| `organelle_state` | `remodel` or `noremodel` |
| `cell_death_state` | `dead`, `alive`, or `null` |
| `annotator` | Username of annotator |
| `annotation_date` | ISO format timestamp |
| `annotation_version` | Annotation schema version |

## Annotation Logic

- **Mitosis**: Marked at specific timepoints where the cell is dividing
- **Infection**: Once marked, the cell is considered infected from that timepoint onward
- **Organelle remodeling**: Once marked, the cell is considered remodeled from that timepoint onward
- **Death**: Once marked, the cell is considered dead from that timepoint onward (all other states become `null`)

## Tips

- The widget matches annotation points to cell labels using a 10-pixel diameter window around the clicked location
- Points that don't map to a cell (background) will generate a warning in the logs
- Use interpolation mode for events that span multiple consecutive timepoints to save time
- The status bar at the bottom shows the current state and any errors