from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import dask.array as da
from iohub.ngff import MultiScaleMeta, Position, Well, _open_store, OMEROMeta
from pydantic.color import Color

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


def stitch_dataset(path: StrOrBytesPath, row_wrap: int = None):
    if not row_wrap:
        row_wrap = 4
    try:
        zgroup = _open_store(path, mode="r", version="0.4")
    except Exception as e:
        raise RuntimeError(e)
    if "well" in zgroup.attrs:
        first_pos_grp = next(zgroup.groups())[1]
        channel_names = Position(first_pos_grp).channel_names
        well = Well(
            group=zgroup,
            parse_meta=True,
            channel_names=channel_names,
            version="0.4",
        )
    else:
        raise KeyError(f"NGFF well metadata not found under {zgroup.name}")
    return _stitch_well_by_channel(well, row_wrap=row_wrap)


def _stitch_well_by_channel(well: Well, row_wrap: int):
    logging.info(f"Stitching well: {well.zgroup.name}")
    levels = []
    pyramids: list[list] = []
    for i, (pos_name, pos) in enumerate(well.positions()):
        l, ims = _get_multiscales(pos)
        levels.append(l)
        pyramids.append(ims)
        if i == 0:
            layers_kwargs = _ome_to_napari_by_channel(pos.metadata)
    stitched_arrays = []
    for i in range(max(levels)):
        ims = [p[i] for p in pyramids if i < len(p)]
        grid = _make_grid(ims, cols=row_wrap)
        stitched_arrays.append(da.block(grid))
    return layers_kwargs, well, stitched_arrays


def _get_multiscales(pos: Position):
    ms: MultiScaleMeta = pos.metadata.multiscales[0]
    images = [dataset.path for dataset in ms.datasets]
    multiscales = []
    for im in images:
        try:
            multiscales.append(da.from_zarr(pos[im]))
        except Exception as e:
            logging.warning(
                f"Skipped array '{im}' at position {pos.zgroup.name}: {e}"
            )
    return len(multiscales), multiscales


def _make_grid(elements: list[da.Array], cols: int):
    ct = len(elements)
    rows = ct // cols + int(bool(ct % cols))
    grid = [elements[r * cols : (r + 1) * cols] for r in range(rows)]
    diff = len(grid[0]) - len(grid[-1])
    if diff > 0:
        fill_shape = grid[0][0].shape
        fill_type = grid[0][0].dtype
        grid[-1].extend([da.zeros(fill_shape, fill_type)] * diff)
    return grid


def _ome_to_napari_by_channel(metadata):
    omero: OMEROMeta = metadata.omero
    layers_kwargs = []
    for channel in omero.channels:
        metadata = {"name": channel.label}
        if channel.color:
            # alpha channel is optional
            rgb = Color(channel.color).as_rgb_tuple(alpha=None)
            start = [0] * 3
            if len(rgb) == 4:
                start += [1]
            metadata["colormap"] = (
                start,
                [v / np.iinfo(np.uint8).max for v in rgb],
            )
        layers_kwargs.append(metadata)
    return layers_kwargs


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not (os.path.isdir(path) and ".zarr" in path):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def _find_ch_axis(dataset):
    for i, axis in enumerate(dataset.axes):
        if axis.type == "channel":
            return i


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    layer_type = "image"
    layers_kwargs, dataset, arrays = stitch_dataset(path)
    layers = []
    ch_axis = _find_ch_axis(dataset)
    if ch_axis is not None:
        if ch_axis > 0:
            pre_idx = [slice(None)] * ch_axis
    else:
        raise IndexError("Cannot index channel axis")
    for i, kwargs in enumerate(layers_kwargs):
        slc = tuple(pre_idx + [slice(i, i + 1)])
        data = [da.squeeze(arr[slc], axis=ch_axis) for arr in arrays]
        layer = (data, kwargs, layer_type)
        layers.append(layer)
    return layers
