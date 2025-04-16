from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import pandas as pd
from iohub.ngff import open_ome_zarr
from magicgui import magic_factory
from xarray import open_zarr

from napari_iohub._reader import fov_to_layers

if TYPE_CHECKING:
    from pathlib import Path

    import napari
    from napari.types import LayerDataTuple

_logger = logging.getLogger(__name__)

_ULTRACK_NO_PARENT = -1
_ULTRACK_TRACKS_HEADER = ("track_id", "t", "z", "y", "x")


def _ultrack_inv_tracks_df_forest(
    df: pd.DataFrame, no_parent=_ULTRACK_NO_PARENT
) -> dict[int, int]:
    """
    Vendored from ultrack.tracks.graph.inv_tracks_df_forest.
    """
    for col in ["track_id", "parent_track_id"]:
        if col not in df.columns:
            raise ValueError(
                f"The input dataframe does not contain the column '{col}'."
            )

    df = df.drop_duplicates("track_id")
    df = df[df["parent_track_id"] != no_parent]
    graph = {}
    for parent_id, id in zip(df["parent_track_id"], df["track_id"]):
        graph[id] = parent_id
    return graph


def _ultrack_read_csv(path: Path | str) -> LayerDataTuple:
    """
    Vendored from ultrack.reader.napari_reader.read_csv.
    """
    if isinstance(path, str):
        path = Path(path)

    df = pd.read_csv(path)

    _logger.info(f"Read {len(df)} tracks from {path}")
    _logger.info(df.head())

    tracks_cols = list(_ULTRACK_TRACKS_HEADER)
    if "z" not in df.columns:
        tracks_cols.remove("z")

    if "parent_track_id" in df.columns:
        graph = _ultrack_inv_tracks_df_forest(df)
        _logger.info(f"Track lineage graph with length {len(graph)}")
    else:
        graph = None

    kwargs = {
        "features": df,
        "name": path.name.removesuffix(".csv"),
        "graph": graph,
    }

    return (df[tracks_cols], kwargs, "tracks")


def _zarr_modes(label: str) -> dict[str, str]:
    return {"mode": "d", "label": label}


def _load_features(features_path: pathlib.Path, fov_name: str) -> pd.DataFrame:
    """Load UMAP and PHATE coordinates from Zarr store if they exist."""
    ds = open_zarr(features_path)
    coords = ["fov_name", "track_id", "t"]
    reductions = ["UMAP1", "UMAP2", "PHATE1", "PHATE2"]
    for dim in reductions:
        if dim in ds:
            coords.append(dim)
    ds = ds.set_index(sample=coords)
    return (
        ds.sel(fov_name=fov_name)["sample"]
        .to_dataframe()
        .reset_index(drop=True)[coords[1:]]
        .rename(columns={"track_id": "label", "t": "frame"})
    )


@magic_factory(
    call_button="Load",
    images_dataset=_zarr_modes("images (.zarr)"),
    tracks_dataset=_zarr_modes("tracking labels (.zarr)"),
    features_dataset=_zarr_modes("features (.zarr)"),
)
def open_image_and_tracks(
    images_dataset: pathlib.Path,
    tracks_dataset: pathlib.Path,
    features_dataset: pathlib.Path | None,
    fov_name: str,
    expand_z_for_tracking_labels: bool = True,
    load_tracks_layer: bool = True,
    tracks_z_index: int = -1,
) -> list[napari.types.LayerDataTuple]:
    """
    Load images and tracking labels.
    Also load predicted features (if supplied)
    and associate them with the tracking labels.
    To be used with napari-clusters-plotter plugin.

    Parameters
    ----------
    images_dataset : pathlib.Path
        Path to the images dataset (HCS OME-Zarr).
    tracks_dataset : pathlib.Path
        Path to the tracking labels dataset (HCS OME-Zarr).
        Potentially with a singleton Z dimension.
    features_dataset : pathlib.Path | None
        Path to the predicted embeddings, optional.
    fov_name : str
        Name of the FOV to load, e.g. `"A/12/2"`.
    expand_z_for_tracking_labels : bool
        Whether to expand the tracking labels to the Z dimension of the images.
    load_tracks_layer : bool
        Whether to load the tracks layer.
    tracks_z_index : int
        Index of the Z slice to place the 2D tracks, by default -1 (middle slice).

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
    _logger.info(f"Loading tracking labels from {tracks_dataset}")
    tracks_plate = open_ome_zarr(tracks_dataset)
    tracks_fov = tracks_plate[fov_name]
    labels_layer = fov_to_layers(tracks_fov, layer_type="labels")[0]
    # TODO: remove this after https://github.com/napari/napari/issues/7327 is fixed
    labels_layer[0][0] = labels_layer[0][0].astype("uint32")
    image_z = image_fov["0"].slices
    if expand_z_for_tracking_labels:
        _logger.info(f"Expanding tracks to Z={image_z}")
        labels_layer[0][0] = labels_layer[0][0].repeat(image_z, axis=1)
    if features_dataset is not None:
        _logger.info(f"Loading features from {str(features_dataset)}")
        labels_layer[1]["features"] = _load_features(
            features_dataset, fov_name
        )
    image_layers.append(labels_layer)
    tracks_csv = next((tracks_dataset / fov_name.strip("/")).glob("*.csv"))
    if load_tracks_layer:
        _logger.info(f"Loading tracks from {str(tracks_csv)} with ultrack")
        tracks_layer = _ultrack_read_csv(tracks_csv)
        if tracks_z_index is not None:
            tracks_z_index = image_z // 2
        _logger.info(f"Placing tracks at Z={tracks_z_index}")
        tracks_layer[0].insert(loc=2, column="z", value=tracks_z_index)
        image_layers.append(tracks_layer)
    _logger.info(f"Finished loading {len(image_layers)} layers")
    _logger.debug(f"Layers: {image_layers}")
    return image_layers
