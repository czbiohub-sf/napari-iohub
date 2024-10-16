import numpy as np
import pytest
from iohub import open_ome_zarr


@pytest.fixture(scope="session")
def hcs_path(tmp_path_factory):
    store_path = tmp_path_factory.mktemp("data") / "hcs.zarr"
    position_list = (
        ("A", "1", "0"),
        ("A", "1", "42"),
        ("A", "1", "pos65535"),
        ("A", "1", "randomAlphanumericName000"),
        ("H", 10, 0),
        ("Control", "Blank", "0"),
        ("Control", "Blank", "a"),
    )
    with open_ome_zarr(
        store_path,
        layout="hcs",
        mode="w-",
        channel_names=["DAPI", "GFP", "Brightfield"],
    ) as dataset:
        for row, col, fov in position_list:
            position = dataset.create_position(row, col, fov)
            for resolution in (0, 1, 2):
                xy = 32 // (2**resolution)
                position[resolution] = np.random.randint(
                    0,
                    np.iinfo(np.uint16).max,
                    size=(4, 3, 2, xy, xy),
                    dtype=np.uint16,
                )
    yield str(store_path)
