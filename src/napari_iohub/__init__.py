import logging
import os

from ._reader import napari_get_reader
from ._widget import MainWidget
from ._edit_labels import EditLabelsWidget
from ._view_tracks import open_image_and_tracks

__all__ = (
    "napari_get_reader",
    "MainWidget",
    "EditLabelsWidget",
    "open_image_and_tracks",
)

_logger = logging.getLogger(__name__)
_logger.setLevel(os.getenv("NAPARI_IOHUB_LOGGING_LEVEL", logging.DEBUG))
