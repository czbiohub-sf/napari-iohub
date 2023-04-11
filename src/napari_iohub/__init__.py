__version__ = "0.0.1"

from ._reader import napari_get_reader
from ._widget import MainWidget

__all__ = (
    "napari_get_reader",
    "MainWidget",
    "example_magic_widget",
)
