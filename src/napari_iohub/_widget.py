import logging
import os
from typing import Callable, List, Optional
import napari
from iohub.ngff import Plate, Row, Well, open_ome_zarr
from napari.qt.threading import create_worker
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt
from superqt import QRangeSlider
from napari_iohub._reader import plate_to_layers, well_to_layers
import pandas as pd

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


def _add_nav_combobox(
    parent: QWidget, label: str, connect: Callable, form_layout: QFormLayout
):
    """
    Add a navigation combo box widget to the form layout.

    Parameters
    ----------
    parent : QWidget
        The parent widget.
    label : str
        The label for the combo box.
    connect : Callable
        The callback function to connect to the combo box's currentTextChanged signal.
    form_layout : QFormLayout
        The form layout to add the combo box to.

    Returns
    -------
    QComboBox
        The created combo box widget.
    """
    cb = QComboBox(parent)
    label = QLabel(cb, text=label)
    cb.currentTextChanged[str].connect(connect)
    form_layout.addRow(label, cb)
    return cb


def _choose_dir(parent: QWidget, caption="Open a directory"):
    """
    Open a file dialog to choose a directory.

    Parameters
    ----------
    parent : QWidget
        The parent widget.
    caption : str, optional
        The caption for the file dialog, by default "Open a directory"

    Returns
    -------
    str
        The selected directory path.
    """
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
    """
    A widget for napari that provides functionality to load and view NGFF HCS plate datasets.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The napari viewer instance.

    Attributes
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    dataset : Plate
        The loaded NGFF HCS plate dataset.
    view_mode : str
        The view mode for multiple positions ('stitch' or 'stack').
    main_layout : QVBoxLayout
        The main layout of the widget.
    dataset_path_le : QLineEdit
        The line edit widget to display the path of the loaded dataset.
    row_range_rs : QRangeSlider
        The range slider widget for selecting the row range.
    col_range_rs : QRangeSlider
        The range slider widget for selecting the column range.
    view_plate_btn : QPushButton
        The button to show the plate.
    row_cb : QComboBox
        The combo box widget for selecting the row.
    well_cb : QComboBox
        The combo box widget for selecting the well.
    view_well_btn : QPushButton
        The button to show the well.

    Methods
    -------
    _add_load_data_layout()
        Add the layout for loading the dataset.
    _add_plate_layout()
        Add the layout for viewing one FOV from every well.
    _add_well_layout()
        Add the layout for viewing all FOVs in a well.
    _add_range_slider(label, form_layout)
        Add a range slider widget to the form layout.
    _load_dataset()
        Load the dataset from the selected directory.
    _show_plate()
        Show the plate with the selected row and column range.
    _load_row(row_name)
        Load the selected row and update the well combo box.
    _load_well(well_name)
        Load the selected well.
    _view_mode(view_mode)
        Set the view mode for multiple positions.
    _show_well()
        Show the well with the selected view mode.
    _update_layers(layers)
        Update the layers in the viewer with the given layer data.
    """

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.dataset = None
        self.view_mode = "stitch"
        self.main_layout = QVBoxLayout()
        self._add_load_data_layout()
        self.main_layout.addWidget(QHLine())
        self._add_metadata_layout()
        self.main_layout.addWidget(QHLine())
        self._add_plate_layout()
        self.main_layout.addWidget(QHLine())
        self._add_well_layout()
        self.setLayout(self.main_layout)

    def _add_load_data_layout(self):
        """
        Add the layout for loading the dataset.

        This layout includes a button to load the dataset, a line edit widget to display the path of the loaded dataset,
        a button to load the metadata file, and a tooltip for the load buttons.
        """
        form_layout = QFormLayout()
        load_zarr_btn = QPushButton("Load zarr store")
        load_zarr_btn.clicked.connect(self._load_dataset)
        load_zarr_btn.setToolTip(
            "Select a directory of the NGFF HCS plate zarr store."
        )
        self.dataset_path_le = QLineEdit()
        self.dataset_path_le.setReadOnly(True)
        form_layout.addRow(load_zarr_btn, self.dataset_path_le)

        load_metadata_btn = QPushButton("Load metadata file")
        load_metadata_btn.clicked.connect(self._load_metadata)
        load_metadata_btn.setToolTip(
            "Select an Excel file containing the metadata."
        )
        self.metadata_path_le = QLineEdit()
        self.metadata_path_le.setReadOnly(True)
        form_layout.addRow(load_metadata_btn, self.metadata_path_le)

        self.main_layout.addLayout(form_layout)

    def _add_plate_layout(self):
        """
        Add the layout for viewing one FOV from every well.

        This layout includes a label, two range slider widgets for selecting the row and column range,
        and a button to show the plate.
        """
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
        """
        Add the layout for viewing all FOVs in a well.

        This layout includes a label, two combo box widgets for selecting the row and well,
        a combo box widget for selecting the view mode, and a button to show the well.
        """
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

    def _add_metadata_layout(self, meta_list: Optional[List[str]] = None):
        """
        Add the layout for selecting wells by metadata.

        This layout includes a label and several combo box widgets for selecting well metadata,
        and an action button to filter the wells based on the selected metadata.

        Parameters
        ----------
        meta_list : List[str], optional
            The list of well metadata names, by default None.
        """
        outer_layout = QVBoxLayout()
        label = QLabel(text="Select metadata")
        outer_layout.addWidget(label)
        form_layout = QFormLayout()
        self.meta1_cb = _add_nav_combobox(
            self, "Meta1", self._update_well_combobox, form_layout
        )
        self.meta2_cb = _add_nav_combobox(
            self, "Meta2", self._update_well_combobox, form_layout
        )
        self.meta3_cb = _add_nav_combobox(
            self, "Meta3", self._update_well_combobox, form_layout
        )
        if meta_list:
            for i, meta_name in enumerate(meta_list, start=1):
                meta_cb = _add_nav_combobox(
                    self,
                    f"Meta{i}",
                    self._update_well_combobox,
                    form_layout,
                )
                setattr(self, f"meta{i}_cb", meta_cb)
        # filter_btn = QPushButton("Filter wells by metadata") #Not implemented yet.
        # form_layout.addRow(filter_btn)
        outer_layout.addLayout(form_layout)
        self.main_layout.addLayout(outer_layout)

    def _add_range_slider(self, label: str, form_layout: QFormLayout):
        """
        Add a range slider widget to the form layout.

        Parameters
        ----------
        label : str
            The label for the range slider.
        form_layout : QFormLayout
            The form layout to add the range slider to.

        Returns
        -------
        QRangeSlider
            The range slider widget.
        """
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
        """
        Load the dataset from the selected directory.

        This method opens a file dialog to select a directory, sets the dataset path line edit widget,
        and loads the dataset using the open_ome_zarr function from iohub.ngff module.
        It also updates the row and column combo box widgets with the available row and column names.
        """
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

    def _load_metadata(self):
        """
        Load the metadata file.

        This method opens a file dialog to select an Excel file containing the metadata.
        It then reads the file and saves the column headings in self.meta_list.
        """
        path = QFileDialog.getOpenFileName(
            self,
            "Open metadata file",
            os.getcwd(),
            "Excel files (*.xlsx *.xls)",
        )[0]
        logging.debug(f"Got metadata path '{path}'")
        if not path:
            return
        self.metadata_path_le.setText(path)

        try:
            workbook = pd.read_excel(path)
            self.meta_list = list(workbook.columns)
        except Exception as e:
            logging.error(f"Error loading metadata file: {e}")
            return

        self.meta1_cb.addItems(self.meta_list)
        self.meta2_cb.addItems(self.meta_list)
        self.meta3_cb.addItems(self.meta_list)

    def _show_plate(self):
        """
        Show the plate with the selected row and column range.

        This method creates a worker thread to load the plate data and update the layers in the viewer.
        """
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

    def _update_well_combobox(self):
        """
        Select wells by metadata.

        This method updates the row and well combo box widgets based on the selected metadata.
        """
        logging.debug("Selecting wells by metadata")

    def _load_row(self, row_name: str):
        """
        Load the selected row and update the well combo box.

        Parameters
        ----------
        row_name : str
            The selected row name.
        """
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
        """
        Load the selected well.

        Parameters
        ----------
        well_name : str
            The selected well name.
        """
        logging.debug(f"Got well name '{well_name}'")
        self.well: Well = self.row[well_name]

    def _view_mode(self, view_mode: str):
        """
        Set the view mode for multiple positions.

        Parameters
        ----------
        view_mode : str
            The selected view mode.
        """
        logging.debug(f"Got well name '{view_mode}'")
        self.view_mode = view_mode

    def _show_well(self):
        """
        Show the well with the selected view mode.

        This method creates a worker thread to load the well data and update the layers in the viewer.
        """
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
        """
        Update the layers in the viewer with the given layer data.

        Parameters
        ----------
        layers : list[tuple]
            The layer data to update the viewer with.
        """
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
            add_method(layer_data[0], **layer_data[1], blending="additive")
        axis_labels = [
            ax.name for ax in self.dataset.axes if ax.type != "channel"
        ]
        if len(self.viewer.dims.axis_labels) == len(axis_labels) + 1:
            axis_labels = ["P"] + axis_labels
        elif not len(self.viewer.dims.axis_labels) == len(axis_labels):
            return
        self.viewer.dims.axis_labels = axis_labels
