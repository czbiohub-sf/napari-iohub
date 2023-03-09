import logging
import os
from typing import Callable

import napari
from iohub.ngff import Plate, Row, Well, open_ome_zarr
from napari.layers import Layer
from napari.utils.notifications import show_info
from napari.qt.threading import create_worker
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

from napari_iohub._reader import well_to_layers


class MainWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.cwd = os.getcwd()
        self.dataset = None
        self.mode = "stitch"
        self.main_layout = QVBoxLayout()
        self._add_load_dataset_layout()
        view_btn = QPushButton("Show well")
        view_btn.clicked.connect(self._show_well)
        self.main_layout.addWidget(view_btn)
        self.setLayout(self.main_layout)

    def _add_load_dataset_layout(self):
        form_layout = QFormLayout()
        btn = QPushButton("Load dataset")
        btn.clicked.connect(self._load_dataset)
        self.dataset_path_le = QLineEdit()
        self.dataset_path_le.setReadOnly(True)
        form_layout.addRow(btn, self.dataset_path_le)
        self.row_cb = self._add_nav_cb("Row", self._load_row, form_layout)
        self.well_cb = self._add_nav_cb("Well", self._load_well, form_layout)
        self.main_layout.addLayout(form_layout)

    def _add_nav_cb(
        self, label: str, connect: Callable, form_layout: QFormLayout
    ):
        cb = QComboBox(self)
        label = QLabel(cb, text=label)
        cb.currentIndexChanged[int].connect(connect)
        form_layout.addRow(label, cb)
        return cb

    def _choose_dir(self, caption="Open a directory"):
        path = QFileDialog.getExistingDirectory(
            parent=self,
            caption=caption,
            directory=self.cwd,
        )
        return path

    def _load_dataset(self):
        path = self._choose_dir(
            "Select a directory of the NGFF HCS plate dataset"
        )
        logging.debug(f"Got dataset path '{path}'")
        if path:
            self.dataset_path_le.setText(path)
            self.dataset: Plate = open_ome_zarr(
                path, layout="hcs", mode="r", version="0.4"
            )
            self.well_cb.clear()
            self.row_cb.clear()
            self.row_names = [row.name for row in self.dataset.metadata.rows]
            self.plate_wells = self.dataset.metadata.wells
            logging.debug(f"Opened dataset with rows {self.row_names}")
            self.row_cb.addItems(self.row_names)

    def _load_row(self, index: int):
        logging.debug(f"Got row index {index}")
        row_name = self.row_names[index]
        self.row: Row = self.dataset[row_name]
        self.well_cb.clear()
        self.well_names = []
        for well in self.plate_wells:
            r, c = well.path.split("/")
            if r == row_name:
                self.well_names.append(c)
        logging.debug(f"Found wells {self.well_names} under row {row_name}")
        self.well_cb.addItems(self.well_names)

    def _load_well(self, index: int):
        logging.debug(f"Got well index {index}")
        well_name = self.well_names[index]
        self.well: Well = self.row[well_name]

    def _show_well(self):
        show_info(f"Showing well {self.well.zgroup.name}")
        self.well.print_tree()
        worker = create_worker(
            well_to_layers, well=self.well, mode=self.mode, layer_type="image"
        )
        worker.returned.connect(self._update_layers)
        logging.debug("Starting well data loading worker")
        worker.start()

    def _update_layers(self, layers: list[tuple]):
        logging.debug("Clearing existing layers in the viewer")
        if self.viewer.layers:
            self.viewer.layers.clear()
        for layer_data in layers:
            logging.debug(f"Adding layer from {layer_data}")
            layer = Layer.create(*layer_data)
            _ = self.viewer.add_layer(layer)
