import logging
import os
from typing import Callable

import napari
from iohub.ngff import Plate, Row, Well, open_ome_zarr
from napari.layers import Layer, image
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
        self.view_mode = "stitch"
        self.main_layout = QVBoxLayout()
        self._add_load_dataset_layout()
        view_btn = QPushButton("Show well")
        view_btn.clicked.connect(self._show_well)
        self.main_layout.addWidget(view_btn)
        self.setLayout(self.main_layout)

    def _add_load_dataset_layout(self):
        form_layout = QFormLayout()
        load_btn = QPushButton("Load dataset")
        load_btn.clicked.connect(self._load_dataset)
        load_btn.setToolTip("Select a directory of the NGFF HCS plate dataset")
        self.dataset_path_le = QLineEdit()
        self.dataset_path_le.setReadOnly(True)
        form_layout.addRow(load_btn, self.dataset_path_le)
        self.row_cb = self._add_nav_cb("Row", self._load_row, form_layout)
        self.well_cb = self._add_nav_cb("Well", self._load_well, form_layout)
        mode_cb = self._add_nav_cb("Mode", self._view_mode, form_layout)
        mode_cb.addItems(["stitch", "stack"])
        mode_cb.setToolTip(
            (
                "View mode for multiple positions:\n"
                "'stitch' will stitch all the positions;\n"
                "'stack' will stack all the positions "
                "on the outer-most dimension"
            )
        )
        self.main_layout.addLayout(form_layout)

    def _add_nav_cb(
        self, label: str, connect: Callable, form_layout: QFormLayout
    ):
        cb = QComboBox(self)
        label = QLabel(cb, text=label)
        cb.currentTextChanged[str].connect(connect)
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
        path = self._choose_dir()
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

    def _load_row(self, row_name: str):
        logging.debug(f"Got row name '{row_name}'")
        self.row: Row = self.dataset[row_name]
        self.well_cb.clear()
        self.well_names = []
        for well in self.plate_wells:
            r, c = well.path.split("/")
            if r == row_name:
                self.well_names.append(c)
        logging.debug(f"Found wells {self.well_names} under row {row_name}")
        self.well_cb.addItems(self.well_names)

    def _load_well(self, well_name: str):
        logging.debug(f"Got well name '{well_name}'")
        self.well: Well = self.row[well_name]

    def _view_mode(self, view_mode: str):
        logging.debug(f"Got well name '{view_mode}'")
        self.view_mode = view_mode

    def _show_well(self):
        show_info(f"Showing well '{self.well.zgroup.name}' \n")
        self.well.print_tree()
        worker = create_worker(
            well_to_layers,
            well=self.well,
            mode=self.view_mode,
            layer_type="image",
        )
        worker.returned.connect(self._update_layers)
        logging.debug("Starting well data loading worker")
        worker.start()

    def _update_layers(self, layers: list[tuple]):
        logging.debug("Clearing existing layers in the viewer")
        # FIXME: different dimensions can cause errors
        # here it clears all layers which will clear user settings too
        self.viewer.layers.clear()
        for layer_data in layers:
            logging.debug(f"Updating layer from {layer_data}")
            # FIXME: this is a workaround copied from
            # https://github.com/napari/napari/blob/7ae2404f7636ce3e1e6db1386b96c69b88a52691/napari/components/viewer_model.py#L1375-L1376 # noqa
            # constructing layer directly cause cryptic color map errors
            add_method = getattr(self.viewer, "add_" + layer_data[2].lower())
            add_method(layer_data[0], **layer_data[1])
