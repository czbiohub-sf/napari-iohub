def test_view_tracks_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    _ = viewer.window.add_plugin_dock_widget("napari-iohub", "View tracks")
