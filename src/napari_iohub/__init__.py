import logging
import os

from ._edit_labels import EditLabelsWidget
from ._reader import napari_get_reader
from ._view_tracks import open_image_and_tracks
from ._widget import MainWidget
from .vispy_plotter import vispy_plotter_widget

__all__ = (
    "napari_get_reader",
    "MainWidget",
    "EditLabelsWidget",
    "open_image_and_tracks",
    "vispy_plotter_widget",
)

_logger = logging.getLogger(__name__)
_logger.setLevel(os.getenv("NAPARI_IOHUB_LOGGING_LEVEL", logging.DEBUG))
