__version__ = "0.0.1"

import logging
import os

from ._reader import napari_get_reader
from ._widget import MainWidget

__all__ = (
    "napari_get_reader",
    "MainWidget",
    "example_magic_widget",
)

_logger = logging.getLogger(__name__)
_logger.setLevel(os.getenv("NAPARI_IOHUB_LOGGING_LEVEL", logging.DEBUG))
