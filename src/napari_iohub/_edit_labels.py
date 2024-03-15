from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from iohub.ngff import ImageArray, Plate, Position, Row, Well, open_ome_zarr
from napari_ome_zarr._reader import napari_get_reader
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari_iohub._reader import fov_to_layers
from napari_iohub._widget import _add_nav_combobox, _choose_dir

if TYPE_CHECKING:
    import napari
    from napari.types import LayerData
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)


class _LoadFOV(QWidget):
    def __init__(self, parent: EditLabelsWidget) -> None:
        super().__init__(parent)
        self.label_channel_pattern = os.getenv(
            "NAPARI_IOHUB_LABEL_CHANNEL_PATTERN", "label"
        )
        self.viewer = parent.viewer
        self.dataset: Plate | None = None
        layout = QVBoxLayout()
        label = QLabel("Load FOV")
        layout.addWidget(label)
        form = QFormLayout()
        load_btn = QPushButton("Browse dataset")
        load_btn.clicked.connect(self._load_dataset)
        load_btn.setToolTip("Select an existing NGFF HCS plate dataset.")
        self.dataset_path_le = QLineEdit()
        self.dataset_path_le.setReadOnly(True)
        form.addRow(load_btn, self.dataset_path_le)
        self.row_cb = _add_nav_combobox(
            self, label="Row", connect=self._load_row, form_layout=form
        )
        self.well_cb = _add_nav_combobox(
            self, label="Well", connect=self._load_well, form_layout=form
        )
        self.fov_cb = _add_nav_combobox(
            self, label="FOV", connect=self._load_fov, form_layout=form
        )
        layout.addLayout(form)
        load_button = QPushButton(text="Load")
        load_button.clicked.connect(self._load_image_and_labels)
        layout.addWidget(load_button)
        self.setLayout(layout)

    def _load_dataset(self):
        path = _choose_dir(self)
        _logger.debug(f"Got load dataset path '{path}'")
        if not path:
            return
        self.dataset_path_le.setText(path)
        self.dataset = open_ome_zarr(
            path, layout="hcs", mode="r", version="0.4"
        )
        self.well_cb.clear()
        self.row_cb.clear()
        self.fov_cb.clear()
        self.row_names = [row.name for row in self.dataset.metadata.rows]
        self.col_names = [col.name for col in self.dataset.metadata.columns]
        self.fov_names = []
        self.plate_wells = self.dataset.metadata.wells
        _logger.debug(f"Opened dataset with rows {self.row_names}")
        self.row_cb.addItems(self.row_names)

    def _load_row(self, row_name: str):
        _logger.debug(f"Got row name '{row_name}'")
        self.row: Row = self.dataset[row_name]
        self.well_cb.clear()
        self.fov_cb.clear()
        self.well_names = []
        for well in self.plate_wells:
            r, c = well.path.split("/")
            if r == row_name:
                self.well_names.append(c)
        _logger.debug(f"Found wells {self.well_names} under row {row_name}")
        self.well_cb.addItems(self.well_names)

    def _load_well(self, well_name: str):
        self.fov_cb.clear()
        self.fov_names = []
        if not well_name:
            return
        _logger.debug(f"Got well name '{well_name}'")
        self.well: Well = self.row[well_name]
        for img_meta in self.well.zattrs["well"]["images"]:
            self.fov_names.append(img_meta["path"])
        _logger.debug(f"Found FOVs {self.fov_names} under well {well_name}")
        self.fov_cb.addItems(self.fov_names)

    def _load_fov(self, fov_name: str):
        if not fov_name:
            return
        _logger.debug(f"Got FOV name '{fov_name}'")
        self.fov: Position = self.well[fov_name]

    def _load_image_and_labels(self):
        if not self.dataset:
            warnings.warn("No image to load. Doing nothing.")
            return
        layers = fov_to_layers(self.fov)
        for data, meta, layer_type in layers:
            # HACK: if multiscale does not decrease in size napari will error
            # https://github.com/napari/napari/issues/984 was not fixed properly
            if isinstance(data, list):
                data = data[0]
            meta["blending"] = "additive"
            name = meta["name"]
            if self.label_channel_pattern.lower() in name.lower():
                layer_type = "labels"
                if not np.issubdtype(data.dtype, np.integer):
                    _logger.info(
                        f"Casting labels data to uint16. Original type was {data.dtype}"
                    )
                    data = data.astype("uint16", casting="unsafe")
                self.labels_channel = name
                if "colormap" in meta:
                    del meta["colormap"]
            if name not in self.viewer.layers:
                add_method = getattr(self.viewer, f"add_{layer_type.lower()}")
                add_method(data, **meta)
            else:
                self.viewer.layers[name].data = data


class _SaveFOV(QWidget):
    def __init__(self, parent: EditLabelsWidget):
        super().__init__(parent)
        self.viewer = parent.viewer
        self.dataset: Plate | None = None
        self.loader = parent.loader
        layout = QVBoxLayout()
        label = QLabel(
            "Save current timepoint\n"
            "Click 'save' for each timepoint edited!"
        )
        layout.addWidget(label)
        form = QFormLayout()
        load_btn = QPushButton("Browse dataset")
        load_btn.clicked.connect(self._load_dataset)
        load_btn.setToolTip(
            "Select the NGFF HCS plate dataset to save to. "
            "The dataset will be created if it does not exist."
        )
        self.dataset_path_le = QLineEdit()
        self.dataset_path_le.setReadOnly(True)
        form.addRow(load_btn, self.dataset_path_le)
        layout.addLayout(form)
        load_button = QPushButton(text="Save")
        load_button.clicked.connect(self._save_image)
        layout.addWidget(load_button)
        self.setLayout(layout)

    def _load_dataset(self):
        if not self.viewer.layers:
            warnings.warn("Must load images before saving. Doing nothing.")
            return
        dialog = QFileDialog(
            parent=self,
            directory=os.getcwd(),
            caption="Zarr store to save to",
            filter="Zarr (*.zarr)",
        )
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        if dialog.exec():
            path = dialog.selectedFiles()[0]
        _logger.debug(f"Got save dataset path '{path}'")
        if not path:
            return
        self.dataset_path_le.setText(path)
        if Path(path).exists():
            channel_names = None
            _logger.info("Save dataset exists, editing...")
        else:
            channel_names = [self.loader.labels_channel]
            _logger.info("Save dataset does not exist, creating...")
        self.dataset = open_ome_zarr(
            path,
            layout="hcs",
            mode="a",
            version="0.4",
            channel_names=channel_names,
        )
        _logger.info(
            f"Opened save dataset with channels {self.dataset.channel_names}"
        )

    def _save_image(self):
        if self.dataset is None:
            warnings.warn("No image to save. Doing nothing.")
            return
        fov_name = self.loader.fov.zgroup.path
        _logger.debug(f"Got save FOV name {fov_name}")
        if fov_name in self.dataset.zgroup:
            fov = self.dataset[fov_name]
            _logger.info(f"Opened existing save FOV {fov_name}")
        else:
            fov = self.dataset.create_position(*fov_name.strip("/").split("/"))
            _logger.info(f"Created save FOV {fov_name}")
        data = self.viewer.layers[self.loader.labels_channel].data
        _logger.debug(
            f"Found labels layer with shape {data.shape} and type {data.dtype}"
        )
        if "0" in fov:
            image = fov["0"]
            _logger.debug(f"Saving to existing image with shape {image.shape}")
        else:
            image = fov.create_zeros(
                "0",
                shape=(data.shape[0], 1, *data.shape[-3:]),
                dtype=data.dtype,
            )
            _logger.debug(f"Created new image with shape {image.shape}")
        steps = self.viewer.dims.current_step
        _logger.info(f"Saving labels layer at coordinate {steps}")
        ch_idx = fov.get_channel_index(self.loader.labels_channel)
        image[steps[0], ch_idx] = data[steps[0]]


class EditLabelsWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        layout = QVBoxLayout()
        self.loader = _LoadFOV(self)
        layout.addWidget(self.loader)
        self.saver = _SaveFOV(self)
        layout.addWidget(self.saver)
        self.setLayout(layout)
