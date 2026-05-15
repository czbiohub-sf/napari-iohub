"""Cell state annotation widget for napari.

This module provides an interactive napari widget for annotating cell states
(mitosis, infection, organelle remodeling, death) and exports annotations
to ultrack-compatible CSV format.

Ported from VisCy PR #349: https://github.com/mehta-lab/VisCy/pull/349
"""

from __future__ import annotations

import getpass
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from iohub.ngff import open_ome_zarr
from napari import Viewer
from qtpy.QtWidgets import (
    QCheckBox,
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
ULTRACK_INDEX_COLUMNS = [
    "fov_name",
    "track_id",
    "t",
    "id",
    "parent_track_id",
    "parent_id",
    "z",
    "y",
    "x",
]

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
        self._resume_csv_path: Path | None = (
            None  # Optional CSV for resuming annotations
        )
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
        instructions = QLabel("Shortcuts: a/d=time, q/e=layers, r=interpolate, s=save")
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
        tracks_btn.setToolTip("Select OME-Zarr directory with segmentation labels")
        self._tracks_path_le = QLineEdit()
        self._tracks_path_le.editingFinished.connect(self._on_tracks_path_changed)
        form.addRow(tracks_btn, self._tracks_path_le)

        # Resume CSV browser (optional)
        resume_btn = QPushButton("Resume CSV")
        resume_btn.clicked.connect(self._browse_resume_csv)
        resume_btn.setToolTip("Optional: select existing CSV to resume annotations")
        self._resume_csv_path_le = QLineEdit()
        self._resume_csv_path_le.editingFinished.connect(self._on_resume_csv_changed)
        form.addRow(resume_btn, self._resume_csv_path_le)

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

        # Display options
        self._project_2d_cb = QCheckBox("Project to 2D around in-focus Z")
        self._project_2d_cb.setToolTip(
            "Project each timepoint to a single 2D plane using the in-focus Z "
            "from FOV metadata 'focus_slice.Phase3D.per_timepoint' "
            "(fallback: z_focus_mean). "
            "Phase channels are sliced at the in-focus Z; fluorescence "
            "channels are max-projected over [focus-5, focus+10] (15 slices). "
            "Warns and does nothing if focus metadata is missing."
        )
        self._project_2d_cb.toggled.connect(self._on_project_2d_toggled)
        layout.addWidget(self._project_2d_cb)

        self._expand_labels_cb = QCheckBox("Expand labels Z to match image")
        self._expand_labels_cb.setChecked(True)
        self._expand_labels_cb.setToolTip(
            "If on, repeat the single-plane segmentation labels along Z "
            "to match the image Z-stack. Automatically disabled in 2D mode."
        )
        layout.addWidget(self._expand_labels_cb)

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
        """Browse for images dataset, starting from the current text-field value."""
        path = _choose_dir(self, directory=self._images_path_le.text())
        if not path:
            return
        self._images_path_le.setText(path)
        self._on_images_path_changed()

    def _browse_tracks(self) -> None:
        """Browse for tracks OME-Zarr directory, starting from the current text-field value."""
        path = _choose_dir(self, directory=self._tracks_path_le.text())
        if not path:
            return
        self._tracks_path_le.setText(path)
        self._on_tracks_path_changed()

    def _browse_resume_csv(self) -> None:
        """Browse for an existing CSV, starting from the current text-field value."""
        current = self._resume_csv_path_le.text()
        if current:
            current_expanded = os.path.expanduser(current)
            if os.path.isfile(current_expanded):
                start_dir = os.path.dirname(current_expanded)
            elif os.path.isdir(current_expanded):
                start_dir = current_expanded
            else:
                parent = os.path.dirname(current_expanded)
                start_dir = parent if os.path.isdir(parent) else os.getcwd()
        else:
            start_dir = os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select CSV to resume annotations",
            directory=start_dir,
            filter="CSV files (*.csv)",
        )
        if not path:
            return
        self._resume_csv_path_le.setText(path)
        self._on_resume_csv_changed()

    def _on_resume_csv_changed(self) -> None:
        """Handle path entry for resume CSV."""
        path = self._resume_csv_path_le.text()
        if not path:
            self._resume_csv_path = None
            return

        path_obj = Path(path)
        if path_obj.suffix.lower() != ".csv":
            self._status_label.setText("Error: Resume path must be a CSV file")
            return

        if not path_obj.exists():
            self._status_label.setText(f"Error: CSV not found: {path_obj.name}")
            return

        self._resume_csv_path = path_obj
        self._status_label.setText(f"Resume CSV: {path_obj.name}")

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
        """Handle path entry for tracks OME-Zarr directory."""
        path = self._tracks_path_le.text()
        if not path:
            return

        try:
            self._tracks_dataset = open_ome_zarr(path)
            self._tracks_csv_path = None
            # Populate row combobox
            self._row_cb.clear()
            self._well_cb.clear()
            self._fov_cb.clear()
            self._row_names = [row.name for row in self._tracks_dataset.metadata.rows]
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
        """Load FOVs for selected well.

        Uses ``Well.metadata.images`` (provided by iohub for both NGFF v0.4
        and v0.5) rather than reaching into raw ``zattrs``, since the
        attribute layout differs between zarr v2 and zarr v3 stores.
        """
        if not well_name:
            return
        self._well = self._row[well_name]
        self._fov_cb.clear()
        self._fov_names = [img.path for img in self._well.metadata.images]
        self._fov_cb.addItems(self._fov_names)

    def _load_fov_name(self, fov_name: str) -> None:
        """Store selected FOV name."""
        if not fov_name:
            return
        row_name = self._row_cb.currentText()
        well_name = self._well_cb.currentText()
        self._fov_name = f"{row_name}/{well_name}/{fov_name}"
        _logger.debug(f"Selected FOV: {self._fov_name}")

    def prefill(
        self,
        images: str | Path | None = None,
        tracks: str | Path | None = None,
        resume_csv: str | Path | None = None,
        fov: str | None = None,
        output: str | Path | None = None,
        project_2d: bool = False,
        expand_labels: bool = True,
        load: bool = False,
    ) -> None:
        """Programmatically populate widget fields and optionally trigger load.

        Parameters
        ----------
        images : str or Path, optional
            Path to the images OME-Zarr dataset.
        tracks : str or Path, optional
            Path to the tracks OME-Zarr dataset.
        resume_csv : str or Path, optional
            Path to a CSV to resume annotations from.
        fov : str, optional
            FOV path of the form ``"row/well/fov"`` (e.g. ``"A/1/0"``).
            Requires ``tracks`` to be set first.
        output : str or Path, optional
            Output folder for saved annotations.
        project_2d : bool, optional
            If True, tick the "Project to 2D around in-focus Z" checkbox.
        expand_labels : bool, optional
            Whether to expand single-plane labels along Z (default ``True``).
            Ignored when ``project_2d`` is True.
        load : bool, optional
            If True, call :meth:`_load_data` after populating fields.
        """
        if images is not None:
            self._images_path_le.setText(str(images))
            self._on_images_path_changed()
        if tracks is not None:
            self._tracks_path_le.setText(str(tracks))
            self._on_tracks_path_changed()
        if resume_csv is not None:
            self._resume_csv_path_le.setText(str(resume_csv))
            self._on_resume_csv_changed()
        if output is not None:
            self._output_path = Path(output)
            self._output_path_le.setText(str(output))
        # Display option toggles before FOV select so signal handlers see the right state
        self._project_2d_cb.setChecked(bool(project_2d))
        if not project_2d:
            self._expand_labels_cb.setChecked(bool(expand_labels))
        if fov is not None:
            try:
                row_name, well_name, fov_name = fov.strip("/").split("/")
            except ValueError as err:
                self._status_label.setText(
                    f"Error: --fov must be 'row/well/fov', got {fov!r}"
                )
                _logger.error(f"Invalid FOV format: {fov!r} ({err})")
                return
            self._row_cb.setCurrentText(row_name)
            self._well_cb.setCurrentText(well_name)
            self._fov_cb.setCurrentText(fov_name)
            self._fov_name = f"{row_name}/{well_name}/{fov_name}"
        if load:
            self._load_data()

    def _browse_output(self) -> None:
        """Browse for output folder, starting from the current text-field value."""
        path = _choose_dir(self, directory=self._output_path_le.text())
        if not path:
            return
        self._output_path = Path(path)
        self._output_path_le.setText(path)

    def _on_project_2d_toggled(self, checked: bool) -> None:
        """When 2D projection is on, expanding labels along Z is unnecessary."""
        if checked:
            self._expand_labels_cb.setChecked(False)
            self._expand_labels_cb.setEnabled(False)
        else:
            self._expand_labels_cb.setEnabled(True)
            self._expand_labels_cb.setChecked(True)

    @staticmethod
    def _is_phase_channel(name: str) -> bool:
        """Return True if a channel name looks like a Phase3D-style channel.

        Phase channels are kept as a single in-focus Z slice rather than
        max-projected, since projection blurs phase contrast.
        """
        return name.lower().startswith("phase")

    def _get_focus_z(self, fov) -> int | None:
        """Return the FOV-level in-focus Z index, or ``None`` if metadata is missing.

        Reads ``focus_slice.Phase3D.fov_statistics.z_focus_mean`` from the
        FOV's zattrs and rounds it to an int. Emits a UI warning when no
        metadata is found, leaving the caller to fall back to the full Z-stack.
        """
        zattrs = dict(fov.zattrs)
        focus_slice = zattrs.get("focus_slice") or {}
        phase_meta = focus_slice.get("Phase3D") if isinstance(focus_slice, dict) else None
        stats = phase_meta.get("fov_statistics") if isinstance(phase_meta, dict) else None
        z_mean = stats.get("z_focus_mean") if isinstance(stats, dict) else None
        if not isinstance(z_mean, (int, float)):
            msg = (
                f"No focus_slice.Phase3D.fov_statistics.z_focus_mean for "
                f"{self._fov_name}; cannot project to 2D. "
                "Loading full Z-stack instead."
            )
            _logger.warning(msg)
            self._status_label.setText(msg)
            return None
        return int(round(float(z_mean)))

    @staticmethod
    def _project_channel(
        channel_data,
        focus_z: int,
        is_phase: bool,
        z_size: int,
        below: int = 5,
        above: int = 10,
    ):
        """Project a (T, Z, Y, X) channel array to (T, Y, X).

        Phase channels are sliced at ``focus_z``. Fluorescence channels are
        max-projected over ``[focus_z - below, focus_z + above]`` (inclusive,
        clipped to the valid Z range).
        """
        if is_phase:
            return channel_data[:, focus_z]
        lo = max(0, focus_z - below)
        hi = min(z_size, focus_z + above + 1)
        return channel_data[:, lo:hi].max(axis=1)

    def _load_data(self) -> None:
        """Load images, tracks, and set up annotation layers."""
        if not hasattr(self, "_images_dataset"):
            self._status_label.setText("Error: Select images dataset first")
            return

        has_tracks_dataset = (
            hasattr(self, "_tracks_dataset") and self._tracks_dataset is not None
        )

        if not has_tracks_dataset:
            self._status_label.setText("Error: Select tracks OME-Zarr directory")
            return

        if not self._fov_name:
            self._status_label.setText("Error: Select a FOV first")
            return

        self._status_label.setText(f"Loading {self._fov_name}...")

        try:
            # Load image layers (list of (data, kwargs, "image") tuples, one per channel)
            _logger.info(f"Loading images for {self._fov_name}")
            image_fov = self._images_dataset[self._fov_name]
            image_layers = fov_to_layers(image_fov)
            image_z = image_fov["0"].slices

            # Decide whether to project to 2D
            project_2d = self._project_2d_cb.isChecked()
            focus_z: int | None = None
            if project_2d:
                focus_z = self._get_focus_z(image_fov)
                if focus_z is None:
                    project_2d = False  # warning already shown
                else:
                    _logger.info(
                        f"Projecting to 2D around z_focus_mean={focus_z} "
                        f"(phase channels sliced, fluorescence max-projected ±[5, 10])"
                    )

            # Apply per-channel projection if 2D mode active
            if project_2d:
                projected_layers = []
                for data, kwargs, ltype in image_layers:
                    channel_name = str(kwargs.get("name", ""))
                    is_phase = self._is_phase_channel(channel_name)
                    new_data = [
                        self._project_channel(
                            arr, focus_z=focus_z, is_phase=is_phase, z_size=image_z,
                        )
                        for arr in data
                    ]
                    projected_layers.append((new_data, kwargs, ltype))
                image_layers = projected_layers

            # Load tracking labels from OME-Zarr
            _logger.info(f"Loading tracking labels for {self._fov_name}")
            self._tracks_fov = self._tracks_dataset[self._fov_name]
            labels_layer = fov_to_layers(self._tracks_fov, layer_type="labels")[0]
            # Fix dtype for napari issue #7327
            labels_layer[0][0] = labels_layer[0][0].astype("uint32")

            # Shape parity check: image YX vs labels YX must match for the
            # segmentation lookup at save time to work correctly.
            image_yx = image_fov["0"].shape[-2:]
            labels_yx = self._tracks_fov["0"].shape[-2:]
            if image_yx != labels_yx:
                _logger.warning(
                    "Image (Y,X)=%s does not match labels (Y,X)=%s; "
                    "segmentation lookup may be incorrect.",
                    image_yx,
                    labels_yx,
                )

            # Labels Z handling. In 2D mode we drop the Z axis entirely so
            # labels stay 1-plane and the annotation layers can use ndim=3.
            # In 4D mode, expand the single-Z labels along Z only if the
            # user asked for it (mirrors the single-cell-features plugin).
            expand_labels = self._expand_labels_cb.isChecked()
            if project_2d:
                labels_layer[0][0] = labels_layer[0][0][:, 0]  # (T,Z=1,Y,X) -> (T,Y,X)
                tracks_z_index = 0
            elif expand_labels:
                _logger.info(f"Expanding tracks to Z={image_z}")
                labels_layer[0][0] = labels_layer[0][0].repeat(image_z, axis=1)
                tracks_z_index = image_z // 2
            else:
                tracks_z_index = 0  # single-plane labels, place tracks at z=0

            image_layers.append(labels_layer)

            # Load tracks CSV from OME-Zarr directory
            tracks_dir = Path(self._tracks_path_le.text()) / self._fov_name.strip("/")
            self._tracks_csv_path = next(tracks_dir.glob("*.csv"))
            _logger.info(f"Loading tracks from {self._tracks_csv_path}")
            tracks_layer = _ultrack_read_csv(self._tracks_csv_path)
            if not project_2d:
                tracks_layer[0].insert(loc=2, column="z", value=tracks_z_index)
            image_layers.append(tracks_layer)

            # Clear existing layers and add new ones
            self.viewer.layers.clear()
            for layer_data, layer_kwargs, layer_type in image_layers:
                add_method = getattr(self.viewer, f"add_{layer_type}")
                add_method(layer_data, **layer_kwargs)

            # Set up annotation layers (3D in projected mode, 4D otherwise)
            self._annotation_ndim = 3 if project_2d else 4
            self._project_2d_active = project_2d
            self._focus_z_used = focus_z if project_2d else None
            self._setup_annotation_layers()

            # Load existing annotations from resume CSV if provided, otherwise from tracks CSV
            if self._resume_csv_path is not None:
                _logger.info(
                    f"Loading annotations from resume CSV: {self._resume_csv_path}"
                )
                annotation_df = pd.read_csv(self._resume_csv_path)
            else:
                annotation_df = pd.read_csv(self._tracks_csv_path)
            self._load_annotations_from_csv(annotation_df, tracks_z_index)

            # Bind keybindings (only once)
            if not self._keybindings_bound:
                self._bind_keybindings()
                self._keybindings_bound = True

            status_msg = f"Loaded {self._fov_name}"
            if self._resume_csv_path:
                status_msg += f" (resuming from {self._resume_csv_path.name})"
            self._status_label.setText(status_msg)
            _logger.info(f"Successfully loaded {len(self.viewer.layers)} layers")

        except Exception as e:
            self._status_label.setText(f"Error: {e}")
            _logger.exception(f"Failed to load data: {e}")

    def _setup_annotation_layers(self) -> None:
        """Create annotation point layers for each event type.

        Uses ``self._annotation_ndim`` (set by :meth:`_load_data`) to pick
        3D coords ``(T, Y, X)`` for projected 2D images or 4D coords
        ``(T, Z, Y, X)`` for full Z-stacks. Defaults to 4D if unset.
        """
        ndim = getattr(self, "_annotation_ndim", 4)
        for layer_name, color, _, _ in ANNOTATION_LAYERS:
            layer = self.viewer.add_points(
                ndim=ndim,
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
        _logger.info("Annotation layers created (ndim=%d)", ndim)

    def _load_annotations_from_csv(self, df: pd.DataFrame, z_index: int) -> None:
        """Load existing annotations from CSV into point layers.

        Detects annotation columns and populates the corresponding layers:
        - cell_division_state == "mitosis" → all matching timepoints (duration)
        - organelle_state == "remodel" → all matching timepoints (duration)
        - infection_state == "infected" → first occurrence per track only
        - cell_death_state == "dead" → first occurrence per track only
        """
        annotation_cols = [col for _, _, col, _ in ANNOTATION_LAYERS]
        if not any(col in df.columns for col in annotation_cols):
            _logger.info("No annotation columns found in CSV, starting fresh")
            return

        # Layers whose points cover the entire duration of the state
        duration_cols = {"cell_division_state", "organelle_state"}

        # Restrict to the current FOV if the column is present
        if "fov_name" in df.columns and self._fov_name:
            df = df[df["fov_name"] == self._fov_name]

        total_points = 0

        for layer_name, _, col_name, value in ANNOTATION_LAYERS:
            if col_name not in df.columns:
                _logger.info(f"Column {col_name} missing from CSV, skipping {layer_name}")
                continue
            if layer_name not in self.viewer.layers:
                _logger.warning(f"Layer {layer_name} not found in viewer")
                continue

            layer = self.viewer.layers[layer_name]

            annotated = df[df[col_name] == value]
            if annotated.empty:
                _logger.info(f"No '{value}' rows for {layer_name}")
                continue

            if col_name in duration_cols:
                points_df = annotated
            else:
                points_df = annotated.loc[annotated.groupby("track_id")["t"].idxmin()]

            points = []
            ndim = getattr(self, "_annotation_ndim", 4)
            for _, row in points_df.iterrows():
                t = int(row["t"])
                y = float(row["y"])
                x = float(row["x"])
                if ndim == 3:
                    points.append([t, y, x])
                else:
                    points.append([t, z_index, y, x])

            if points:
                layer.data = np.asarray(points, dtype=float)
                layer.refresh()
                total_points += len(points)
                _logger.info(f"Loaded {len(points)} points into {layer_name}")

        if total_points > 0:
            self._status_label.setText(f"Loaded {total_points} existing annotations")
        else:
            _logger.info("No annotations matched in resume CSV")

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

    def _backup_csv(self, output_path: Path) -> None:
        """Create a timestamped backup of the CSV file.

        Parameters
        ----------
        output_path : Path
            Path to the CSV file to backup.

        Backups are stored in a 'backup' folder relative to the CSV location.
        Only the latest 5 backups per unique base filename are kept.
        """
        if not output_path.exists():
            return

        backup_dir = output_path.parent / "backup"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup_name = f"{output_path.stem}_{timestamp}.csv"
        backup_path = backup_dir / backup_name

        shutil.copy2(output_path, backup_path)
        _logger.info(f"Created backup: {backup_path}")

        # Keep only latest 5 backups for this base filename
        base_stem = output_path.stem
        all_backups = sorted(
            backup_dir.glob(f"{base_stem}_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old_backup in all_backups[5:]:
            old_backup.unlink()
            _logger.info(f"Removed old backup: {old_backup}")

    def _save_annotations(self) -> None:
        """Save annotations to ultrack-compatible CSV."""
        if self._tracks_csv_path is None or self._tracks_fov is None:
            self._status_label.setText("Error: Load data first")
            return

        try:
            # Load tracks CSV for all track-timepoint combinations
            tracks_df = pd.read_csv(self._tracks_csv_path)

            # Annotation metadata
            try:
                annotator = os.getlogin()
            except OSError:
                # Fallback for VMs/containers where getlogin() fails
                annotator = getpass.getuser()
            annotation_date = datetime.now().isoformat()

            # Determine version: increment if re-saving from resume CSV, otherwise start at 1.0
            if self._resume_csv_path is not None:
                resume_df = pd.read_csv(self._resume_csv_path)
                if "annotation_version" in resume_df.columns:
                    old_version = resume_df["annotation_version"].iloc[0]
                    try:
                        major, minor = str(old_version).split(".")
                        annotation_version = f"{int(major) + 1}.0"
                    except (ValueError, AttributeError):
                        annotation_version = "2.0"
                else:
                    annotation_version = "2.0"
            else:
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
                    # Points are (T,Z,Y,X) in full Z mode or (T,Y,X) in 2D mode
                    coords = [int(c) for c in point]
                    if len(coords) == 4:
                        t, _z, y, x = coords
                    else:
                        t, y, x = coords

                    # Load segmentation for this timepoint (labels are single-Z)
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
                remodel_timepoints = [e["t"] for e in organelle_events]

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
                            "remodel" if t in remodel_timepoints else "noremodel"
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
            index_cols = [
                col for col in ULTRACK_INDEX_COLUMNS if col in merged_df.columns
            ]
            annotation_cols = [
                "cell_division_state",
                "infection_state",
                "organelle_state",
                "cell_death_state",
            ]
            metadata_cols = ["annotator", "annotation_date", "annotation_version"]
            column_order = index_cols + annotation_cols + metadata_cols
            merged_df = merged_df[column_order]

            # Determine output path
            if self._resume_csv_path is not None:
                # Save back to the resume CSV
                output_csv = self._resume_csv_path
            else:
                # Generate new filename from plate/row/well/fov
                plate_name = Path(self._tracks_path_le.text()).stem
                row, well, fov = self._fov_name.split("/")
                base_name = f"{plate_name}_{row}_{well}_{fov}"
                output_csv = self._output_path / f"{base_name}.csv"

            # Create backup before overwriting
            self._backup_csv(output_csv)

            merged_df.to_csv(output_csv, index=False)

            self._status_label.setText(
                f"Saved to {output_csv.name} (v{annotation_version})"
            )
            _logger.info(f"Saved annotations to {output_csv}")

        except Exception as e:
            self._status_label.setText(f"Save error: {e}")
            _logger.exception(f"Failed to save annotations: {e}")
