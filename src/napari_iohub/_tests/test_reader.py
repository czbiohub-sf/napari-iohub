import os

from iohub import open_ome_zarr
from numpy.testing import assert_equal

from napari_iohub import napari_get_reader


# tmp_path is a pytest fixture
def test_reader(hcs_path):
    """Test reader contribution."""
    well_path = os.path.join(hcs_path, "A", "1")
    reader = napari_get_reader(well_path)
    assert callable(reader)
    layer_data_list = reader(well_path)
    assert isinstance(layer_data_list, list) and len(layer_data_list) == 3
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) == 3
    with open_ome_zarr(hcs_path) as plate:
        pos = plate["A/1/0"]
        tzyx = pos["0"][:, 0]
    assert_equal(layer_data_tuple[0][0][..., :32, :32], tzyx)


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None
    reader = napari_get_reader("/tmp/")
    assert reader is None
