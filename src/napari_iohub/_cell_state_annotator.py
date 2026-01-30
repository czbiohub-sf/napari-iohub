"""Cell state annotation widget for napari.

This module provides an interactive napari widget for annotating cell states
(mitosis, infection, organelle remodeling, death) and exports annotations
to ultrack-compatible CSV format.

Ported from VisCy PR #349: https://github.com/mehta-lab/VisCy/pull/349
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from iohub.ngff import open_ome_zarr
from napari import Viewer
from qtpy.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari_iohub._reader import fov_to_layers
from napari_iohub._view_tracks import _ultrack_read_csv
from napari_iohub._widget import _add_nav_combobox, _choose_dir

if TYPE_CHECKING:
    import napari

_logger = logging.getLogger(__name__)

# Index columns for ultrack-compatible CSV output
INDEX_COLUMNS = ["fov_name", "track_id", "t", "z", "y", "x"]

# Annotation layer definitions
ANNOTATION_LAYERS = [
    ("_mitosis_events", "blue", "cell_division_state", "mitosis"),
    ("_infected_events", "orange", "infection_state", "infected"),
    ("_remodel_events", "purple", "organelle_state", "remodel"),
    ("_death_events", "red", "cell_death_state", "dead"),
]


class CellStateAnnotatorWidget(QWidget):
    """Interactive cell state annotation widget.

    This widget allows users to:
    - Load images and tracking data from OME-Zarr datasets
    - Add point annotations for cell states (mitosis, infection, remodel, death)
    - Use keyboard shortcuts for efficient annotation
    - Save annotations to ultrack-compatible CSV format

    Keyboard shortcuts:
    - a/d: Step backward/forward in time
    - q/e: Cycle through annotation layers
    - r: Toggle interpolation mode (click start → press 'r' → click end)
    - s: Save annotations
    """

    def __init__(self, napari_viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self._tracks_fov = None
        self._tracks_csv_path: Path | None = None
        self._fov_name: str = ""
        self._output_path: Path = Path.cwd()

        # Interpolation state
        self._interpolation_mode = False
        self._start_point: np.ndarray | None = None

        # Current annotation layer index for cycling
        self._current_layer_index = 0

        self._setup_ui()
        self._keybindings_bound = False

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Cell State Annotation")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "Shortcuts: a/d=time, q/e=layers, r=interpolate, s=save"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Form for dataset paths
        form = QFormLayout()

        # Images dataset browser
        images_btn = QPushButton("Browse images")
        images_btn.clicked.connect(self._browse_images)
        images_btn.setToolTip("Select the OME-Zarr dataset with images")
        self._images_path_le = QLineEdit()
        self._images_path_le.editingFinished.connect(self._on_images_path_changed)
        form.addRow(images_btn, self._images_path_le)

        # Tracks dataset browser
        tracks_btn = QPushButton("Browse tracks")
        tracks_btn.clicked.connect(self._browse_tracks)
        tracks_btn.setToolTip("Select CSV file, or type path for OME-Zarr directory")
        self._tracks_path_le = QLineEdit()
        self._tracks_path_le.editingFinished.connect(self._on_tracks_path_changed)
        form.addRow(tracks_btn, self._tracks_path_le)

        # FOV navigation
        self._row_cb = _add_nav_combobox(
            self, label="Row", connect=self._load_row, form_layout=form
        )
        self._well_cb = _add_nav_combobox(
            self, label="Well", connect=self._load_well, form_layout=form
        )
        self._fov_cb = _add_nav_combobox(
            self, label="FOV", connect=self._load_fov_name, form_layout=form
        )

        layout.addLayout(form)

        # Load button
        load_btn = QPushButton("Load Data")
        load_btn.clicked.connect(self._load_data)
        load_btn.setToolTip("Load images, tracks, and set up annotation layers")
        layout.addWidget(load_btn)

        # Separator
        layout.addWidget(QLabel(""))

        # Output path
        output_form = QFormLayout()
        output_btn = QPushButton("Output folder")
        output_btn.clicked.connect(self._browse_output)
        output_btn.setToolTip("Select folder to save annotation CSVs")
        self._output_path_le = QLineEdit()
        self._output_path_le.setReadOnly(True)
        self._output_path_le.setText(str(self._output_path))
        output_form.addRow(output_btn, self._output_path_le)
        layout.addLayout(output_form)

        # Save button
        save_btn = QPushButton("Save Annotations")
        save_btn.clicked.connect(self._save_annotations)
        save_btn.setToolTip("Save annotations to CSV (also bound to 's' key)")
        layout.addWidget(save_btn)

        # Status label
        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        layout.addStretch()
        self.setLayout(layout)

    def _browse_images(self) -> None:
        """Browse for images dataset."""
        path = _choose_dir(self)
        if not path:
            return
        self._images_path_le.setText(path)
        self._on_images_path_changed()

    def _browse_tracks(self) -> None:
        """Browse for tracks CSV file. For OME-Zarr directories, type the path."""
        path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select tracks CSV (or type path for OME-Zarr)",
            directory=os.getcwd(),
            filter="CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        self._tracks_path_le.setText(path)
        self._on_tracks_path_changed()

    def _on_images_path_changed(self) -> None:
        """Handle path entry for images dataset."""
        path = self._images_path_le.text()
        if not path:
            return
        try:
            self._images_dataset = open_ome_zarr(path)
            self._status_label.setText(f"Images: {Path(path).name}")
        except Exception as e:
            self._status_label.setText(f"Error opening images: {e}")

    def _on_tracks_path_changed(self) -> None:
        """Handle path entry for tracks dataset (CSV or OME-Zarr directory)."""
        path = self._tracks_path_le.text()
        if not path:
            return

        path_obj = Path(path)

        if path_obj.suffix.lower() == ".csv":
            # CSV file selected - store path directly
            self._tracks_csv_path = path_obj
            self._tracks_dataset = None
            # Clear FOV navigation (not needed for direct CSV)
            self._row_cb.clear()
            self._well_cb.clear()
            self._fov_cb.clear()
            self._status_label.setText(f"Tracks CSV: {path_obj.name}")
        else:
            # Assume OME-Zarr directory
            try:
                self._tracks_dataset = open_ome_zarr(path)
                self._tracks_csv_path = None
                # Populate row combobox
                self._row_cb.clear()
                self._well_cb.clear()
                self._fov_cb.clear()
                self._row_names = [
                    row.name for row in self._tracks_dataset.metadata.rows
                ]
                self._col_names = [
                    col.name for col in self._tracks_dataset.metadata.columns
                ]
                self._plate_wells = self._tracks_dataset.metadata.wells
                self._row_cb.addItems(self._row_names)
                self._status_label.setText(f"Tracks: {Path(path).name}")
            except Exception as e:
                self._status_label.setText(f"Error opening tracks: {e}")

    def _load_row(self, row_name: str) -> None:
        """Load wells for selected row."""
        if not row_name or not hasattr(self, "_tracks_dataset"):
            return
        self._row = self._tracks_dataset[row_name]
        self._well_cb.clear()
        self._fov_cb.clear()
        self._well_names = []
        for well in self._plate_wells:
            r, c = well.path.split("/")
            if r == row_name:
                self._well_names.append(c)
        self._well_cb.addItems(self._well_names)

    def _load_well(self, well_name: str) -> None:
        """Load FOVs for selected well."""
        if not well_name:
            return
        self._well = self._row[well_name]
        self._fov_cb.clear()
        self._fov_names = []
        for img_meta in self._well.zattrs["well"]["images"]:
            self._fov_names.append(img_meta["path"])
        self._fov_cb.addItems(self._fov_names)

    def _load_fov_name(self, fov_name: str) -> None:
        """Store selected FOV name."""
        if not fov_name:
            return
        row_name = self._row_cb.currentText()
        well_name = self._well_cb.currentText()
        self._fov_name = f"{row_name}/{well_name}/{fov_name}"
        _logger.debug(f"Selected FOV: {self._fov_name}")

    def _browse_output(self) -> None:
        """Browse for output folder."""
        path = _choose_dir(self)
        if not path:
            return
        self._output_path = Path(path)
        self._output_path_le.setText(path)

    def _load_data(self) -> None:
        """Load images, tracks, and set up annotation layers."""
        if not hasattr(self, "_images_dataset"):
            self._status_label.setText("Error: Select images dataset first")
            return

        # Check if we have a direct CSV or need OME-Zarr tracks
        has_direct_csv = self._tracks_csv_path is not None
        has_tracks_dataset = (
            hasattr(self, "_tracks_dataset") and self._tracks_dataset is not None
        )

        if not has_direct_csv and not has_tracks_dataset:
            self._status_label.setText("Error: Select tracks (CSV or OME-Zarr)")
            return

        if not has_direct_csv and not self._fov_name:
            self._status_label.setText("Error: Select a FOV first")
            return

        # Determine FOV name from CSV if loaded directly
        if has_direct_csv and not self._fov_name:
            # Try to get FOV name from CSV
            try:
                df = pd.read_csv(self._tracks_csv_path, nrows=1)
                if "fov_name" in df.columns:
                    self._fov_name = df["fov_name"].iloc[0]
                else:
                    self._status_label.setText("Error: CSV missing fov_name column")
                    return
            except Exception as e:
                self._status_label.setText(f"Error reading CSV: {e}")
                return

        self._status_label.setText(f"Loading {self._fov_name}...")

        try:
            # Load image layers
            _logger.info(f"Loading images for {self._fov_name}")
            image_fov = self._images_dataset[self._fov_name]
            image_layers = fov_to_layers(image_fov)

            # Load tracking labels if we have a tracks dataset
            if has_tracks_dataset:
                _logger.info(f"Loading tracking labels for {self._fov_name}")
                self._tracks_fov = self._tracks_dataset[self._fov_name]
                labels_layer = fov_to_layers(self._tracks_fov, layer_type="labels")[0]
                # Fix dtype for napari issue #7327
                labels_layer[0][0] = labels_layer[0][0].astype("uint32")

                # Expand Z if needed
                image_z = image_fov["0"].slices
                _logger.info(f"Expanding tracks to Z={image_z}")
                labels_layer[0][0] = labels_layer[0][0].repeat(image_z, axis=1)

                image_layers.append(labels_layer)
            else:
                self._tracks_fov = None

            # Get image Z for track positioning
            image_z = image_fov["0"].slices

            # Load tracks CSV (use direct path or find in tracks directory)
            if not has_direct_csv:
                tracks_dir = (
                    Path(self._tracks_path_le.text()) / self._fov_name.strip("/")
                )
                self._tracks_csv_path = next(tracks_dir.glob("*.csv"))
            _logger.info(f"Loading tracks from {self._tracks_csv_path}")
            tracks_layer = _ultrack_read_csv(self._tracks_csv_path)
            tracks_z_index = image_z // 2
            tracks_layer[0].insert(loc=2, column="z", value=tracks_z_index)
            image_layers.append(tracks_layer)

            # Clear existing layers and add new ones
            self.viewer.layers.clear()
            for layer_data, layer_kwargs, layer_type in image_layers:
                add_method = getattr(self.viewer, f"add_{layer_type}")
                add_method(layer_data, **layer_kwargs)

            # Set up annotation layers
            self._setup_annotation_layers()

            # Load existing annotations from CSV if present
            tracks_df = pd.read_csv(self._tracks_csv_path)
            self._load_annotations_from_csv(tracks_df, tracks_z_index)

            # Bind keybindings (only once)
            if not self._keybindings_bound:
                self._bind_keybindings()
                self._keybindings_bound = True

            self._status_label.setText(f"Loaded {self._fov_name}")
            _logger.info(f"Successfully loaded {len(self.viewer.layers)} layers")

        except Exception as e:
            self._status_label.setText(f"Error: {e}")
            _logger.exception(f"Failed to load data: {e}")

    def _setup_annotation_layers(self) -> None:
        """Create annotation point layers for each event type."""
        for layer_name, color, _, _ in ANNOTATION_LAYERS:
            layer = self.viewer.add_points(
                ndim=4,
                size=20,
                face_color=color,
                name=layer_name,
            )
            layer.mode = "add"
            # Add mouse callback for interpolation
            layer.mouse_drag_callbacks.append(self._on_point_added)

        # Set initial active layer
        self.viewer.layers.selection.active = self.viewer.layers["_mitosis_events"]
        self._current_layer_index = 0
        _logger.info("Annotation layers created")

    def _load_annotations_from_csv(self, df: pd.DataFrame, z_index: int) -> None:
        """Load existing annotations from CSV into point layers.

        Detects annotation columns and populates the corresponding layers:
        - cell_division_state == "mitosis" → all matching points
        - infection_state == "infected" → first occurrence per track
        - organelle_state == "remodel" → first occurrence per track
        - cell_death_state == "dead" → first occurrence per track
        """
        annotation_cols = [col for _, _, col, _ in ANNOTATION_LAYERS]
        if not any(col in df.columns for col in annotation_cols):
            _logger.info("No annotation columns found in CSV, starting fresh")
            return

        total_points = 0

        for layer_name, _, col_name, value in ANNOTATION_LAYERS:
            if col_name not in df.columns:
                continue

            layer = self.viewer.layers[layer_name]

            # Filter rows with the annotation value
            annotated = df[df[col_name] == value]
            if annotated.empty:
                continue

            if col_name == "cell_division_state":
                # Mitosis: add all marked timepoints
                points_df = annotated
            else:
                # Others: add only first occurrence per track
                points_df = annotated.loc[annotated.groupby("track_id")["t"].idxmin()]

            # Add points to layer
            points = []
            for _, row in points_df.iterrows():
                t, y, x = int(row["t"]), float(row["y"]), float(row["x"])
                points.append([t, z_index, y, x])

            if points:
                layer.add(np.array(points))
                total_points += len(points)
                _logger.info(f"Loaded {len(points)} points into {layer_name}")

        if total_points > 0:
            self._status_label.setText(f"Loaded {total_points} existing annotations")

    def _on_point_added(self, layer, event) -> None:
        """Handle point addition for interpolation tracking."""
        coords = np.array(layer.world_to_data(event.position))

        if self._interpolation_mode and self._start_point is not None:
            # Interpolate between start and end points
            end_coords = coords
            start_coords = self._start_point

            t1, t2 = int(start_coords[0]), int(end_coords[0])
            if t1 > t2:
                t1, t2 = t2, t1
                start_coords, end_coords = end_coords, start_coords

            # Add intermediate timepoints
            for t in range(t1 + 1, t2):
                alpha = (t - t1) / (t2 - t1)
                interpolated = start_coords + alpha * (end_coords - start_coords)
                interpolated[0] = t
                layer.add(interpolated)

            _logger.info(f"Interpolated {t2 - t1 - 1} points between t={t1} and t={t2}")
            self._interpolation_mode = False
            self._start_point = None
            self._status_label.setText("Interpolation complete")
        else:
            # Track point for potential interpolation
            self._start_point = coords

    def _bind_keybindings(self) -> None:
        """Bind keyboard shortcuts to the viewer."""

        @self.viewer.bind_key("a")
        def step_backward(viewer: Viewer) -> None:
            current_step = viewer.dims.current_step
            if current_step[0] > 0:
                viewer.dims.current_step = (current_step[0] - 1, *current_step[1:])
                self._status_label.setText(f"Time: {viewer.dims.current_step[0]}")

        @self.viewer.bind_key("d")
        def step_forward(viewer: Viewer) -> None:
            current_step = viewer.dims.current_step
            max_step = viewer.dims.range[0][1] - 1
            if current_step[0] < max_step:
                viewer.dims.current_step = (current_step[0] + 1, *current_step[1:])
                self._status_label.setText(f"Time: {viewer.dims.current_step[0]}")

        @self.viewer.bind_key("q")
        def cycle_backward(viewer: Viewer) -> None:
            self._current_layer_index = (self._current_layer_index - 1) % len(
                ANNOTATION_LAYERS
            )
            layer_name = ANNOTATION_LAYERS[self._current_layer_index][0]
            viewer.layers.selection.active = viewer.layers[layer_name]
            self._status_label.setText(f"Layer: {layer_name}")

        @self.viewer.bind_key("e")
        def cycle_forward(viewer: Viewer) -> None:
            self._current_layer_index = (self._current_layer_index + 1) % len(
                ANNOTATION_LAYERS
            )
            layer_name = ANNOTATION_LAYERS[self._current_layer_index][0]
            viewer.layers.selection.active = viewer.layers[layer_name]
            self._status_label.setText(f"Layer: {layer_name}")

        @self.viewer.bind_key("r")
        def toggle_interpolation(viewer: Viewer) -> None:
            if self._start_point is not None:
                self._interpolation_mode = True
                start_t = int(self._start_point[0])
                self._status_label.setText(
                    f"Interpolation ON - click end point (start t={start_t})"
                )
            else:
                self._status_label.setText("Add a point first, then press 'r'")

        @self.viewer.bind_key("s")
        def save_shortcut(viewer: Viewer) -> None:
            self._save_annotations()

    def _save_annotations(self) -> None:
        """Save annotations to ultrack-compatible CSV."""
        if self._tracks_csv_path is None or self._tracks_fov is None:
            self._status_label.setText("Error: Load data first")
            return

        try:
            # Load tracks CSV for all track-timepoint combinations
            tracks_df = pd.read_csv(self._tracks_csv_path)

            # Annotation metadata
            annotator = os.getlogin()
            annotation_date = datetime.now().isoformat()
            annotation_version = "1.0"

            # Collect marked events from each layer
            marked_events: dict[str, list[dict]] = {
                "cell_division_state": [],
                "infection_state": [],
                "organelle_state": [],
                "cell_death_state": [],
            }

            # Process annotation layers
            for layer_name, _, event_type, _ in ANNOTATION_LAYERS:
                if layer_name not in self.viewer.layers:
                    continue

                points_layer = self.viewer.layers[layer_name]
                points_data = points_layer.data

                for point in points_data:
                    t, z, y, x = [int(coord) for coord in point]

                    # Load segmentation for this timepoint
                    labels_image = self._tracks_fov["0"][t, 0, 0]

                    # Get label value in window around point
                    diameter = 10
                    y_slice = slice(
                        max(0, y - diameter),
                        min(labels_image.shape[0], y + diameter),
                    )
                    x_slice = slice(
                        max(0, x - diameter),
                        min(labels_image.shape[1], x + diameter),
                    )
                    label_value = int(labels_image[y_slice, x_slice].mean())

                    if label_value > 0:
                        marked_events[event_type].append(
                            {"track_id": label_value, "t": t}
                        )
                    else:
                        _logger.warning(
                            f"Point at t={t}, y={y}, x={x} maps to background"
                        )

            # Expand annotations to all timepoints
            all_annotations = []
            all_track_timepoints = tracks_df[["track_id", "t"]].drop_duplicates()

            for track_id in all_track_timepoints["track_id"].unique():
                track_timepoints = all_track_timepoints[
                    all_track_timepoints["track_id"] == track_id
                ]["t"].sort_values()

                # Get marked events for this track
                division_events = [
                    e
                    for e in marked_events["cell_division_state"]
                    if e["track_id"] == track_id
                ]
                mitosis_timepoints = [e["t"] for e in division_events]

                infection_events = [
                    e
                    for e in marked_events["infection_state"]
                    if e["track_id"] == track_id
                ]
                first_infected_t = (
                    min([e["t"] for e in infection_events])
                    if infection_events
                    else None
                )

                organelle_events = [
                    e
                    for e in marked_events["organelle_state"]
                    if e["track_id"] == track_id
                ]
                first_remodel_t = (
                    min([e["t"] for e in organelle_events])
                    if organelle_events
                    else None
                )

                death_events = [
                    e
                    for e in marked_events["cell_death_state"]
                    if e["track_id"] == track_id
                ]
                first_death_t = (
                    min([e["t"] for e in death_events]) if death_events else None
                )

                # Create row for each timepoint
                for t in track_timepoints:
                    is_dead = first_death_t is not None and t >= first_death_t

                    if is_dead:
                        cell_division_state = None
                        infection_state = None
                        organelle_state = None
                        cell_death_state = "dead"
                    else:
                        cell_division_state = (
                            "mitosis" if t in mitosis_timepoints else "interphase"
                        )
                        infection_state = (
                            "infected"
                            if (first_infected_t is not None and t >= first_infected_t)
                            else "uninfected"
                            if first_infected_t is not None
                            else None
                        )
                        organelle_state = (
                            "remodel"
                            if (first_remodel_t is not None and t >= first_remodel_t)
                            else "noremodel"
                        )
                        cell_death_state = (
                            "alive" if first_death_t is not None else None
                        )

                    all_annotations.append(
                        {
                            "track_id": track_id,
                            "t": t,
                            "cell_division_state": cell_division_state,
                            "infection_state": infection_state,
                            "organelle_state": organelle_state,
                            "cell_death_state": cell_death_state,
                            "annotator": annotator,
                            "annotation_date": annotation_date,
                            "annotation_version": annotation_version,
                        }
                    )

            if not all_annotations:
                self._status_label.setText("No annotations to save")
                return

            # Create output dataframe
            annotations_df = pd.DataFrame(all_annotations)
            tracks_df["fov_name"] = self._fov_name

            # Merge with tracks to preserve index columns
            merged_df = tracks_df.merge(
                annotations_df, on=["track_id", "t"], how="left"
            )

            # Reorder columns
            index_cols = [col for col in INDEX_COLUMNS if col in merged_df.columns]
            annotation_cols = [
                "cell_division_state",
                "infection_state",
                "organelle_state",
                "cell_death_state",
            ]
            metadata_cols = ["annotator", "annotation_date", "annotation_version"]
            column_order = index_cols + annotation_cols + metadata_cols
            merged_df = merged_df[column_order]

            # Save file
            plate_name = Path(self._tracks_path_le.text()).stem
            row, well, fov = self._fov_name.split("/")
            base_name = f"{plate_name}_{row}_{well}_{fov}"
            output_csv = self._output_path / f"{base_name}.csv"

            merged_df.to_csv(output_csv, index=False)

            self._status_label.setText(f"Saved to {output_csv.name}")
            _logger.info(f"Saved annotations to {output_csv}")

        except Exception as e:
            self._status_label.setText(f"Save error: {e}")
            _logger.exception(f"Failed to save annotations: {e}")
