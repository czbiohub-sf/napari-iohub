from napari_iohub._edit_labels import EditLabelsWidget


def test_edit_labels_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    _ = EditLabelsWidget(viewer)
