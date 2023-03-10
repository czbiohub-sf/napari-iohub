from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Literal

import dask.array as da
import numpy as np
from iohub.ngff import (
    MultiScaleMeta,
    OMEROMeta,
    Position,
    Well,
    Plate,
    _open_store,
    open_ome_zarr,
)
from pydantic.color import Color

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


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
    if not os.path.isfile(os.path.join(path, ".zattrs")):
        return None
    # otherwise we return the *function* that can read ``path``.
    return reader_function


def _get_node(path: StrOrBytesPath):
    try:
        zgroup = _open_store(path, mode="r", version="0.4")
    except Exception as e:
        raise RuntimeError(e)
    if "well" in zgroup.attrs:
        first_pos_grp = next(zgroup.groups())[1]
        channel_names = Position(first_pos_grp).channel_names
        node = Well(
            group=zgroup,
            parse_meta=True,
            channel_names=channel_names,
            version="0.4",
        )
    elif "plate" in zgroup.attrs:
        zgroup.store.close()
        node = open_ome_zarr(store_path=path, layout="hcs", mode="r")
    else:
        raise KeyError(
            f"NGFF plate or well metadata not found under '{zgroup.name}'"
        )
    return node


def stitch_well_by_channel(well: Well, row_wrap: int):
    logging.debug(f"Stitching well: {well.zgroup.name}")
    levels = []
    pyramids: list[list] = []
    for i, (_, pos) in enumerate(well.positions()):
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
    return layers_kwargs, _find_ch_axis(well), stitched_arrays


def stack_well_by_position(well: Well):
    logging.debug(f"Stacking well: {well.zgroup.name}")
    levels = []
    pyramids: list[list] = []
    for i, (_, pos) in enumerate(well.positions()):
        l, ims = _get_multiscales(pos)
        levels.append(l)
        pyramids.append(ims)
        if i == 0:
            layers_kwargs = _ome_to_napari_by_channel(pos.metadata)
    stacked_arrays = []
    for i in range(max(levels)):
        ims = [p[i] for p in pyramids if i < len(p)]
        stacked_arrays.append(da.stack(ims, axis=0))
    return layers_kwargs, _find_ch_axis(well), stacked_arrays


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
            start = [0.0] * 3
            if len(rgb) == 4:
                start += [1]
            metadata["colormap"] = np.array(
                [
                    start,
                    [v / np.iinfo(np.uint8).max for v in rgb],
                ]
            )
        layers_kwargs.append(metadata)
    return layers_kwargs


def _find_ch_axis(dataset: Well):
    for i, axis in enumerate(dataset.axes):
        if axis.type == "channel":
            return i


def layers_from_arrays(
    layers_kwargs: list,
    ch_axis: int,
    arrays: list,
    mode: Literal["stitch", "stack"],
    layer_type="image",
):
    if mode == "stack":
        ch_axis += 1
    elif mode != "stitch":
        raise ValueError(f"Unknown mode '{mode}'")
    if ch_axis is not None:
        if ch_axis == 0:
            pre_idx = []
        else:
            pre_idx = [slice(None)] * ch_axis
    else:
        raise IndexError("Cannot index channel axis")
    layers = []
    for i, kwargs in enumerate(layers_kwargs):
        slc = tuple(pre_idx + [slice(i, i + 1)])
        data = [da.squeeze(arr[slc], axis=ch_axis) for arr in arrays]
        layer = (data, kwargs, layer_type)
        layers.append(layer)
    return layers


def well_to_layers(
    well: Well, mode: Literal["stitch", "stack"], layer_type: str
):
    if mode == "stitch":
        layers_kwargs, ch_axis, arrays = stitch_well_by_channel(
            well, row_wrap=4
        )
    elif mode == "stack":
        layers_kwargs, ch_axis, arrays = stack_well_by_position(well)
    return layers_from_arrays(
        layers_kwargs, ch_axis, arrays, mode=mode, layer_type=layer_type
    )


def plate_to_layers(plate: Plate):
    plate_arrays = []
    for row_meta in plate.metadata.rows:
        row_name = row_meta.name
        row_arrays = []
        for col_meta in plate.metadata.columns:
            col_name = col_meta.name
            if row_name + "/" + col_name in [
                w.path for w in plate.metadata.wells
            ]:
                well = plate[row_name][col_name]
                layers_kwargs, ch_axis, arrays = stack_well_by_position(well)
                row_arrays.append([a[0] for a in arrays])
            else:
                row_arrays.append(None)
        plate_arrays.append(row_arrays)
    first_blocks = next(a for a in plate_arrays[0] if a is not None)
    fill_args = [(b.shape, b.dtype) for b in first_blocks]
    plate_levels = []
    for level, _ in enumerate(first_blocks):
        plate_level = []
        for r in plate_arrays:
            row_level = []
            for c in r:
                if c is None:
                    arr = da.zeros(
                        shape=fill_args[level][0], dtype=fill_args[level][1]
                    )
                else:
                    arr = c[level]
                row_level.append(arr)
            plate_level.append(row_level)
        plate_levels.append(da.block(plate_level))
    return layers_from_arrays(
        layers_kwargs,
        ch_axis,
        plate_levels,
        mode="stitch",
        layer_type="image",
    )


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
    node = _get_node(path)
    if isinstance(node, Well):
        return well_to_layers(node, mode="stitch", layer_type="image")
    else:
        return plate_to_layers(node)
