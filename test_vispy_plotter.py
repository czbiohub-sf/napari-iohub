#!/usr/bin/env python
"""
Test script for the Vispy plotter widget.

This script creates a napari viewer with test data and adds the Vispy plotter widget.
"""

import numpy as np
import pandas as pd
import napari
from napari.layers import Points, Labels
from napari_iohub.vispy_plotter import vispy_plotter_widget

def main():
    """Run the test script."""
    # Create a napari viewer
    viewer = napari.Viewer()
    
    # Generate random 3D points
    n_points = 100
    points_data = np.random.rand(n_points, 3) * 10
    
    # Generate random labels for the points
    labels = np.random.randint(1, 6, n_points)
    
    # Create a points layer
    points_layer = viewer.add_points(
        points_data,
        size=5,
        name='test_points',
        face_color='blue'
    )
    
    # Create a features dataframe
    features = pd.DataFrame({
        'x': points_data[:, 0],
        'y': points_data[:, 1],
        'z': points_data[:, 2],
        'intensity': np.random.rand(n_points) * 100,
        'size': np.random.rand(n_points) * 10,
        'label': labels,
        'cluster_id': labels  # Use labels as cluster IDs
    })
    
    # Set the features as the layer's features
    points_layer.features = features
    
    # Create an instance of the widget with the viewer
    plotter = vispy_plotter_widget(viewer)
    
    # Add the widget to the viewer
    viewer.window.add_dock_widget(plotter, name='Vispy Plotter')
    
    # Plot some features
    plotter.layer_combo.setCurrentText('test_points')
    plotter.x_combo.setCurrentText('x')
    plotter.y_combo.setCurrentText('y')
    plotter.cluster_combo.setCurrentText('cluster_id')
    plotter.plot_clicked()
    
    # Run the napari viewer
    napari.run()

if __name__ == '__main__':
    main() 