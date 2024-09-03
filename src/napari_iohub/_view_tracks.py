from __future__ import annotations

import logging
import pathlib
import typing
from typing import TYPE_CHECKING

import pandas as pd
from iohub.ngff import open_ome_zarr
from magicgui import magic_factory
from ultrack.reader.napari_reader import read_csv
from xarray import open_zarr

from napari_iohub._reader import fov_to_layers

if TYPE_CHECKING:
    import napari

_logger = logging.getLogger(__name__)


def _zarr_modes(label: str) -> dict[str, str]:
    return {"mode": "d", "label": label}


@magic_factory(
    call_button="Load",
    images_dataset=_zarr_modes("images (late)"),
    tracks_dataset=_zarr_modes("tracking labels"),
    features_dir=_zarr_modes("features directory"),
)
def open_image_and_tracks(
    images_dataset: pathlib.Path,
    tracks_dataset: pathlib.Path,
    features_dataset: pathlib.Path,
    fov_name: str,
    features_type: str,
    expand_z_for_tracking_labels: bool = True,
) -> typing.List[napari.types.LayerDataTuple]:
    """
    Load images and tracking labels.
    Also load features from a directory and associate them with the tracking labels.
    To be used with napari-clusters-plotter plugin.

    Parameters
    ----------
    images_dataset : pathlib.Path
        Path to the images dataset (HCS OME-Zarr).
    tracks_dataset : pathlib.Path
        Path to the tracking labels dataset (HCS OME-Zarr).
        Potentially with a singleton Z dimension.
    features_dataset : pathlib.Path
        Path to the predicted embeddings.
    fov_name : str
        Name of the FOV to load, e.g. `"A/12/2"`.
    features_type : str
        Name of the subdirectory containing the feature files.
    expand_z_for_tracking_labels : bool
        Whether to expand the tracking labels to the Z dimension of the images.

    Returns
    -------
    List[napari.types.LayerDataTuple]
        List of layers to add to the viewer.
        (image layers and one labels layer)
    """
    _logger.info(f"Loading images from {images_dataset}")
    image_plate = open_ome_zarr(images_dataset)
    image_fov = image_plate[fov_name]
    image_layers = fov_to_layers(image_fov)
    _logger.info(f"Loading {features_type} from {str(features_dataset)}")
    features = (
        open_zarr(features_dataset)
        .set_index(sample=["fov_name", "track_id"])[features_type]
        .sel(fov_name=fov_name)
    )
    _logger.info(f"Loading tracking labels from {tracks_dataset}")
    tracks_plate = open_ome_zarr(tracks_dataset)
    tracks_fov = tracks_plate[fov_name]
    labels_layer = fov_to_layers(tracks_fov, layer_type="labels")[0]
    if expand_z_for_tracking_labels:
        image_z = image_fov["0"].slices
        _logger.info(f"Expanding tracks to Z={image_z}")
        labels_layer[0][0] = labels_layer[0][0].repeat(image_z, axis=1)
    tracks_csv = next((tracks_dataset / fov_name).glob("*.csv"))
    _logger.info(f"Loading tracks from {str(tracks_csv)}")
    tracks_layer = read_csv(tracks_csv)
    image_layers.append(tracks_layer)
    df = pd.read_csv(tracks_csv)
    tracks = (
        df[["track_id", "t"]]
        .rename(columns={"track_id": "label", "t": "frame"})
        .set_index("track_id")
    )
    tracks = tracks[features["track_id"]].reset_index()
    labels_layer[1]["features"] = pd.DataFrame(
        index=tracks, data=features.values
    ).reset_index()
    image_layers.append(labels_layer)
    return image_layers
