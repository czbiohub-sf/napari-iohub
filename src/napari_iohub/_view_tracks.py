from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from iohub.ngff import open_ome_zarr
from magicgui import magic_factory

from napari_iohub._reader import fov_to_layers

if TYPE_CHECKING:
    import napari

_logger = logging.getLogger(__name__)


def _zarr_modes(label: str) -> dict[str, str]:
    return {"mode": "d", "label": label}


@magic_factory(
    call_button="Load",
    images_dataset=_zarr_modes("Images (Plate)"),
    tracks_dataset=_zarr_modes("Tracking labels"),
)
def open_image_and_tracks(
    images_dataset: pathlib.Path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/2.1-register/registered.zarr",
    tracks_dataset: pathlib.Path = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_02_04_A549_DENV_ZIKV_timelapse/5-finaltrack/track_labels_final.zarr",
    fov_name: str = "B/4/4",
    expand_z_for_tracks: bool = True,
    points_z_position: int = 36,
) -> list[napari.types.LayerDataTuple]:
    _logger.info(f"Loading images from {images_dataset}")
    image_plate = open_ome_zarr(images_dataset)
    image_fov = image_plate[fov_name]
    image_layers = fov_to_layers(image_fov)
    _logger.info(f"Loading tracks from {tracks_dataset}")
    tracks_plate = open_ome_zarr(tracks_dataset)
    tracks_fov = tracks_plate[fov_name]
    labels_layer = fov_to_layers(tracks_fov, layer_type="labels")[0]
    if expand_z_for_tracks:
        image_z = image_fov["0"].slices
        _logger.info(f"Expanding tracks to Z={image_z}")
        labels_layer[0][0] = labels_layer[0][0].repeat(image_z, axis=1)
    image_layers.append(labels_layer)
    tracks_csv = next((tracks_dataset / fov_name).glob("*.csv"))
    _logger.info(f"Loading tracks from {str(tracks_csv)}")
    df = pd.read_csv(tracks_csv)
    if expand_z_for_tracks:
        df["z"] = np.ones_like(df["t"].to_numpy()) * points_z_position
    elif "z" not in df.columns:
        raise ValueError(
            "Tracks CSV must contain a 'z' column when not expanding Z."
        )
    points = df[["t", "z", "y", "x"]]
    features = df[["t", "y", "x"]].rename(
        columns={"t": "frame", "y": "centroid_y", "x": "centroid_x"}
    )
    image_layers.append(
        (points, {"name": "Tracked nodes", "features": features}, "points")
    )
    return image_layers