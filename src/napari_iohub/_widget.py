import logging
import os
from typing import Callable

import napari
from iohub.ngff.nodes import Plate, Row, Well, open_ome_zarr
from napari.qt.threading import create_worker
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider

from napari_iohub._reader import plate_to_layers, well_to_layers


def _add_nav_combobox(
    parent: QWidget, label: str, connect: Callable, form_layout: QFormLayout
):
    cb = QComboBox(parent)
    label = QLabel(cb, text=label)
    cb.currentTextChanged[str].connect(connect)
    form_layout.addRow(label, cb)
    return cb


def _choose_dir(parent: QWidget, caption="Open a directory"):
    path = QFileDialog.getExistingDirectory(
        parent=parent,
        caption=caption,
        directory=os.getcwd(),
    )
    return path


class QHLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.HLine)


class MainWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.dataset = None
        self.view_mode = "stitch"
        self.main_layout = QVBoxLayout()
        self._add_load_data_layout()
        self.main_layout.addWidget(QHLine())
        self._add_plate_layout()
        self.main_layout.addWidget(QHLine())
        self._add_well_layout()
        self.setLayout(self.main_layout)

    def _add_load_data_layout(self):
        form_layout = QFormLayout()
        load_btn = QPushButton("Load dataset")
        load_btn.clicked.connect(self._load_dataset)
        load_btn.setToolTip("Select a directory of the NGFF HCS plate dataset")
        self.dataset_path_le = QLineEdit()
        self.dataset_path_le.setReadOnly(True)
        form_layout.addRow(load_btn, self.dataset_path_le)
        self.main_layout.addLayout(form_layout)

    def _add_plate_layout(self):
        outer_layout = QVBoxLayout()
        form_layout = QFormLayout()
        label = QLabel(text="View one FOV from every well")
        outer_layout.addWidget(label)
        self.row_range_rs = self._add_range_slider("Row range", form_layout)
        self.col_range_rs = self._add_range_slider("Column range", form_layout)
        outer_layout.addLayout(form_layout)
        self.view_plate_btn = QPushButton("Show plate")
        outer_layout.addWidget(self.view_plate_btn)
        self.main_layout.addLayout(outer_layout)

    def _add_well_layout(self):
        outer_layout = QVBoxLayout()
        label = QLabel(text="View all FOVs in a well")
        outer_layout.addWidget(label)
        form_layout = QFormLayout()
        self.row_cb = _add_nav_combobox(
            self, "Row", self._load_row, form_layout
        )
        self.well_cb = _add_nav_combobox(
            self, "Well", self._load_well, form_layout
        )
        mode_cb = _add_nav_combobox(self, "Mode", self._view_mode, form_layout)
        mode_cb.addItems(["stitch", "stack"])
        mode_cb.setToolTip(
            (
                "View mode for multiple positions:\n"
                "'stitch' will stitch all the positions;\n"
                "'stack' will stack all the positions "
                "on the outer-most dimension"
            )
        )
        outer_layout.addLayout(form_layout)
        self.view_well_btn = QPushButton("Show well")
        outer_layout.addWidget(self.view_well_btn)
        self.main_layout.addLayout(outer_layout)

    def _add_range_slider(self, label: str, form_layout: QFormLayout):
        slider = QRangeSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 1)
        slider.setValue((0, 1))
        # FIXME: adding ticks crashes Qt
        # slider.setTickInterval(1)
        # slider.setTickPosition(QRangeSlider.TickPosition.TicksBothSides)
        label = QLabel(slider, text=label)
        form_layout.addRow(label, slider)
        return slider

    def _load_dataset(self):
        path = _choose_dir(self)
        logging.debug(f"Got dataset path '{path}'")
        if not path:
            return
        self.dataset_path_le.setText(path)
        self.dataset: Plate = open_ome_zarr(
            path, layout="hcs", mode="r", version="0.4"
        )
        self.well_cb.clear()
        self.row_cb.clear()
        self.row_names = [row.name for row in self.dataset.metadata.rows]
        self.col_names = [col.name for col in self.dataset.metadata.columns]
        self.plate_wells = self.dataset.metadata.wells
        logging.debug(f"Opened dataset with rows {self.row_names}")
        self.row_cb.addItems(self.row_names)
        rr = (0, len(self.row_names))
        self.row_range_rs.setRange(*rr)
        self.row_range_rs.setValue(rr)
        cr = (0, len(self.col_names))
        self.col_range_rs.setRange(*cr)
        self.col_range_rs.setValue(cr)
        self.view_plate_btn.clicked.connect(self._show_plate)
        self.view_well_btn.clicked.connect(self._show_well)

    def _show_plate(self):
        row_range = self.row_range_rs.value()
        col_range = self.col_range_rs.value()
        show_info(
            f"Showing rows in range {row_range}, "
            f"columns in range {col_range}\n"
        )
        worker = create_worker(
            plate_to_layers,
            plate=self.dataset,
            row_range=row_range,
            col_range=col_range,
        )
        worker.returned.connect(self._update_layers)
        logging.debug("Starting plate data loading worker")
        worker.start()

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
        axis_labels = self.viewer.layers[0].axis_labels
        if (
            len(self.viewer.dims.axis_labels)
            == len(self.viewer.layers[0].data.shape) + 1
        ):
            axis_labels = ["P"] + axis_labels
        elif not len(self.viewer.dims.axis_labels) == len(axis_labels):
            return
        self.viewer.dims.axis_labels = axis_labels
