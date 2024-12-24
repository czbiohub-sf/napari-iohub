import logging
import os

from ._edit_labels import EditLabelsWidget
from ._reader import napari_get_reader
from ._widget import MainWidget

__all__ = (
    "napari_get_reader",
    "MainWidget",
    "EditLabelsWidget",
)

try:
    from ._view_tracks import open_image_and_tracks
    __all__ += ("open_image_and_tracks",)
except ImportError:
    # napari-iohub[clustering] is not installed
    pass

_logger = logging.getLogger(__name__)
_logger.setLevel(os.getenv("NAPARI_IOHUB_LOGGING_LEVEL", logging.DEBUG))
