"""
Main widget for the Vispy plotter.

This module provides the main widget for the Vispy plotter, which integrates
with napari and provides the user interface for interactive visualization.
"""

import numpy as np
import pandas as pd
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

from napari.layers import Image, Labels, Layer, Points, Surface
from napari.utils.colormaps import ALL_COLORMAPS
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._canvas import VispyCanvas
from ._utils import estimate_bin_number, get_cluster_colors, points_in_current_slice


def vispy_plotter_widget(napari_viewer):
    """Create a Vispy plotter widget.
    
    This function is the entry point for napari to create the widget.
    
    Args:
        napari_viewer: The napari viewer instance.
        
    Returns:
        The Vispy plotter widget instance.
    """
    # Create and return an instance of the widget with the viewer
    return VispyPlotterWidget(napari_viewer)


class VispyPlotterWidget(QWidget):
    """Main widget for the Vispy plotter.
    
    This class provides the main widget for the Vispy plotter, which integrates
    with napari and provides the user interface for interactive visualization.
    
    Attributes:
        viewer: The napari viewer instance.
        canvas: The Vispy canvas for visualization.
    """
    
    def __init__(self, napari_viewer):
        """Initialize the widget.
        
        Args:
            napari_viewer: The napari viewer instance.
        """
        super().__init__()
        self.viewer = napari_viewer
        
        # Initialize variables
        self.analysed_layer = None
        self.data_x = None
        self.data_y = None
        self.plot_data = None
        self.data_indices = None
        self.features_df = None
        self.colors = None
        self.original_colors = None
        self.class_assignments = None
        self.current_class = 0
        self.colormap_name = "viridis"  # Initialize colormap name
        self.class_colors = [
            [1.0, 0.0, 0.0, 1.0],  # Red
            [0.0, 1.0, 0.0, 1.0],  # Green
            [0.0, 0.0, 1.0, 1.0],  # Blue
            [1.0, 1.0, 0.0, 1.0],  # Yellow
            [1.0, 0.0, 1.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0, 1.0],  # Orange
            [0.5, 0.0, 1.0, 1.0],  # Purple
        ]
        
        # Get the current frame
        self.frame = (
            self.viewer.dims.current_step[0] 
            if self.viewer and self.viewer.dims and self.viewer.dims.ndim > 0 
            else 0
        )
        
        # Store the active layer and selection state
        self.napari_active_layer = None
        self.napari_selected_indices = []
        self.last_selection_state = None
        
        # Initialize UI
        self._init_ui()
        
        # Connect to viewer
        self._connect_to_viewer()
        
    def _connect_to_viewer(self):
        """Connect to the napari viewer.
        
        This method connects the widget to the napari viewer events.
        It should be called when the viewer becomes available.
        """
        # Connect to viewer events
        self.viewer.layers.events.inserted.connect(self.update_layer_list)
        self.viewer.layers.events.removed.connect(self.update_layer_list)
        self.viewer.layers.events.reordered.connect(self.update_layer_list)
        self.viewer.dims.events.current_step.connect(self.frame_changed)
        self.viewer.dims.events.ndisplay.connect(self._on_ndisplay_change)
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_change)
        
        # Connect to mouse click events
        self.viewer.mouse_drag_callbacks.append(self._on_napari_mouse_click)
        
        # Update the layer list
        self.update_layer_list()
        
    def _init_ui(self):
        """Initialize the user interface."""
        # Create the main layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create the layer selection section
        layer_layout = QHBoxLayout()
        layer_label = QLabel("Layer:")
        layer_label.setObjectName("layer_label")
        self.layer_combo = QComboBox()
        self.layer_combo.setObjectName("layer_combo")
        layer_layout.addWidget(layer_label)
        layer_layout.addWidget(self.layer_combo)
        
        # Add extract features button
        self.extract_features_button = QPushButton("Extract Features")
        self.extract_features_button.setObjectName("extract_features_button")
        self.extract_features_button.setToolTip("Extract features from the selected layer")
        self.extract_features_button.clicked.connect(self.extract_features_from_selected_layer)
        layer_layout.addWidget(self.extract_features_button)
        
        layout.addLayout(layer_layout)
        
        # Create the axes selection section
        axes_layout = QHBoxLayout()
        x_label = QLabel("X:")
        x_label.setObjectName("x_label")
        self.x_combo = QComboBox()
        self.x_combo.setObjectName("x_combo")
        y_label = QLabel("Y:")
        y_label.setObjectName("y_label")
        self.y_combo = QComboBox()
        self.y_combo.setObjectName("y_combo")
        axes_layout.addWidget(x_label)
        axes_layout.addWidget(self.x_combo)
        axes_layout.addWidget(y_label)
        axes_layout.addWidget(self.y_combo)
        layout.addLayout(axes_layout)
        
        # Create the clustering selection section
        cluster_layout = QHBoxLayout()
        cluster_label = QLabel("Color by:")
        cluster_label.setObjectName("cluster_label")
        self.cluster_combo = QComboBox()
        self.cluster_combo.setObjectName("cluster_combo")
        cluster_layout.addWidget(cluster_label)
        cluster_layout.addWidget(self.cluster_combo)
        layout.addLayout(cluster_layout)
        
        # Create the colormap selection section
        colormap_layout = QHBoxLayout()
        colormap_label = QLabel("Colormap:")
        colormap_label.setObjectName("colormap_label")
        self.colormap_combo = QComboBox()
        self.colormap_combo.setObjectName("colormap_combo")
        self.colormap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "jet", "rainbow", "tab10", "tab20"])
        self.colormap_combo.setCurrentText("viridis")
        colormap_layout.addWidget(colormap_label)
        colormap_layout.addWidget(self.colormap_combo)
        layout.addLayout(colormap_layout)
        
        # Create the plot button
        button_layout = QHBoxLayout()
        self.plot_button = QPushButton("Plot")
        self.plot_button.setObjectName("plot_button")
        button_layout.addWidget(self.plot_button)
        layout.addLayout(button_layout)
        
        # Create the canvas for the plot
        self.canvas = VispyCanvas()
        self.canvas.setObjectName("canvas")
        self.canvas.selection_changed.connect(self.on_selection_changed)
        self.canvas.point_clicked.connect(self.on_point_clicked)
        layout.addWidget(self.canvas, 1)
        
        # Create the selection info label
        self.selection_info_label = QLabel("No selection")
        self.selection_info_label.setObjectName("selection_info_label")
        layout.addWidget(self.selection_info_label)
        
        # Create a timer for temporary highlighting
        self.highlight_timer = QTimer()
        self.highlight_timer.setObjectName("highlight_timer")
        self.highlight_timer.setSingleShot(True)
        self.highlight_timer.timeout.connect(self._update_point_highlighting)
        
        # Create a timer to check for selection changes
        self.selection_timer = QTimer()
        self.selection_timer.setObjectName("selection_timer")
        self.selection_timer.setInterval(100)  # Check every 100ms
        self.selection_timer.timeout.connect(self._check_napari_selection)
        self.selection_timer.start()
        
        # Connect signals for UI elements
        self.layer_combo.currentIndexChanged.connect(self.update_axes_and_clustering_id_lists)
        self.colormap_combo.currentIndexChanged.connect(self.update_colormap)
        self.cluster_combo.currentIndexChanged.connect(self.plot_clicked)
        self.plot_button.clicked.connect(self.plot_clicked)

    def update_layer_list(self, event=None):
        """Update the layer list combobox with available layers.
        
        Args:
            event: The event that triggered the update (optional).
        """
        # Get the layer combobox
        layer_combo = self.findChild(QComboBox, "layer_combo")
        if layer_combo is None:
            return
            
        # Remember the current selection
        current_layer_name = layer_combo.currentText()
        
        # Clear the combobox
        layer_combo.clear()
        
        # Add all layers to the dropdown
        for layer in self.viewer.layers:
            layer_combo.addItem(layer.name)
            
        # Restore the previous selection if possible
        index = layer_combo.findText(current_layer_name)
        if index >= 0:
            layer_combo.setCurrentIndex(index)
            
        # Update the axes lists
        self.update_axes_and_clustering_id_lists()

    def get_selected_layer(self):
        """Get the currently selected layer.
        
        Returns:
            The selected layer, or None if no layer is selected.
        """
        layer_name = self.layer_combo.currentText()
        if layer_name:
            return self.viewer.layers[layer_name]
        return None

    def update_axes_and_clustering_id_lists(self):
        """Update the axes and clustering ID lists based on the selected layer."""
        # Clear the comboboxes
        self.x_combo.clear()
        self.y_combo.clear()
        self.cluster_combo.clear()
        self.cluster_combo.addItem("None")
        
        # Get the selected layer
        layer_name = self.layer_combo.currentText()
        if not layer_name or layer_name not in self.viewer.layers:
            # If no layer is selected or the layer doesn't exist, add a message and disable the run button
            self.x_combo.addItem("No layer selected")
            self.y_combo.addItem("No layer selected")
            self.plot_button.setEnabled(False)
            print(f"No layer selected or layer '{layer_name}' not found")
            return
            
        layer = self.viewer.layers[layer_name]
        print(f"Selected layer: {layer_name} ({type(layer).__name__})")
        
        # Get features from the layer
        features = self.get_layer_tabular_data(layer)
        if features is None:
            # If no features are available, add a message and disable the run button
            self.x_combo.addItem("No features available")
            self.y_combo.addItem("No features available")
            
            # Show a more helpful message in the selection info label
            layer_type = type(layer).__name__
            self.selection_info_label.setText(
                f"Layer '{layer_name}' ({layer_type}) has no features. "
                f"Add features to this layer to enable plotting."
            )
            
            self.plot_button.setEnabled(False)
            print(f"No features available for layer '{layer_name}'")
            return
        
        print(f"Features available: {', '.join(features.columns)}")
        
        # Add all columns to the axes lists
        for column in features.columns:
            self.x_combo.addItem(column)
            self.y_combo.addItem(column)
            self.cluster_combo.addItem(column)  # Add to color feature dropdown
        
        # Try to select default axes
        default_x_axes = ["UMAP_1", "PCA_1", "x", "centroid_x"]
        default_y_axes = ["UMAP_2", "PCA_2", "y", "centroid_y"]
        
        # Try to select default x axis
        x_selected = False
        for axis in default_x_axes:
            index = self.x_combo.findText(axis)
            if index >= 0:
                self.x_combo.setCurrentIndex(index)
                print(f"Selected default X axis: {axis}")
                x_selected = True
                break
        
        # If no default x axis was found, select the first item
        if not x_selected and self.x_combo.count() > 0:
            self.x_combo.setCurrentIndex(0)
            print(f"Selected first available X axis: {self.x_combo.currentText()}")
        
        # Try to select default y axis
        y_selected = False
        for axis in default_y_axes:
            index = self.y_combo.findText(axis)
            if index >= 0:
                self.y_combo.setCurrentIndex(index)
                print(f"Selected default Y axis: {axis}")
                y_selected = True
                break
        
        # If no default y axis was found, select the second item or the first if only one is available
        if not y_selected:
            if self.y_combo.count() > 1:
                self.y_combo.setCurrentIndex(1)
                print(f"Selected second available Y axis: {self.y_combo.currentText()}")
            elif self.y_combo.count() > 0:
                self.y_combo.setCurrentIndex(0)
                print(f"Selected first available Y axis: {self.y_combo.currentText()}")
        
        # Reset the selection info label
        self.selection_info_label.setText("No selection")
        
        # Enable the run button
        self.plot_button.setEnabled(True)

    def get_layer_tabular_data(self, layer):
        """Get tabular data from a layer.
        
        Args:
            layer: The layer to get data from.
        
        Returns:
            A pandas DataFrame with the layer's tabular data, or None if no data is available.
        """
        # Check if the layer has features
        if hasattr(layer, 'features') and layer.features is not None:
            # Check if features is not empty
            if len(layer.features) > 0:
                return layer.features
            else:
                print(f"Warning: Layer {layer.name} has an empty features table")
        else:
            print(f"Warning: Layer {layer.name} does not have features")
        
        return None

    def extract_image_features(self, image_layer):
        """Extract basic features from an image layer.
        
        This method extracts basic features from an image layer, such as pixel values
        and coordinates, and adds them as features to the layer.
        
        Args:
            image_layer: The image layer to extract features from.
            
        Returns:
            True if features were successfully extracted, False otherwise.
        """
        from napari.layers import Image
        import pandas as pd
        import numpy as np
        
        # Check if the layer is an image layer
        if not isinstance(image_layer, Image):
            self.selection_info_label.setText(f"Layer '{image_layer.name}' is not an image layer")
            return False
            
        # Get the image data
        data = image_layer.data
        
        # Check if the image is too large
        if data.size > 1_000_000:
            self.selection_info_label.setText(
                f"Image '{image_layer.name}' is too large for automatic feature extraction. "
                f"Please downsample or crop the image first."
            )
            return False
            
        # Extract features based on the dimensionality of the image
        if data.ndim == 2:
            # 2D image
            height, width = data.shape
            y, x = np.mgrid[:height, :width]
            
            # Flatten the arrays
            values = data.flatten()
            x = x.flatten()
            y = y.flatten()
            
            # Create a DataFrame with the features
            features = pd.DataFrame({
                'x': x,
                'y': y,
                'value': values,
                'normalized_value': (values - values.min()) / (values.max() - values.min() + 1e-10)
            })
            
        elif data.ndim == 3:
            # 3D image (z, y, x) or (c, y, x)
            if image_layer.rgb:
                # RGB image (c, y, x)
                height, width, channels = data.shape
                y, x = np.mgrid[:height, :width]
                
                # Flatten the arrays
                x = x.flatten()
                y = y.flatten()
                
                # Create a DataFrame with the features
                features = pd.DataFrame({
                    'x': np.tile(x, channels),
                    'y': np.tile(y, channels),
                    'channel': np.repeat(np.arange(channels), height * width),
                    'value': data.reshape(-1)
                })
                
            else:
                # 3D image (z, y, x)
                depth, height, width = data.shape
                z, y, x = np.mgrid[:depth, :height, :width]
                
                # Flatten the arrays
                values = data.flatten()
                x = x.flatten()
                y = y.flatten()
                z = z.flatten()
                
                # Create a DataFrame with the features
                features = pd.DataFrame({
                    'x': x,
                    'y': y,
                    'z': z,
                    'value': values,
                    'normalized_value': (values - values.min()) / (values.max() - values.min() + 1e-10)
                })
                
        else:
            # Higher dimensional image - not supported
            self.selection_info_label.setText(
                f"Image '{image_layer.name}' has {data.ndim} dimensions, which is not supported "
                f"for automatic feature extraction."
            )
            return False
            
        # Set the features on the layer
        image_layer.features = features
        
        # Update the UI
        self.selection_info_label.setText(
            f"Extracted {len(features)} features from image '{image_layer.name}'"
        )
        
        # Update the axes lists
        self.update_axes_and_clustering_id_lists()
        
        return True

    def _debug_compare_coordinates(self, layer):
        """Compare coordinates between the layer and the plot for debugging.
        
        Args:
            layer: The layer to compare coordinates with.
        """
        if isinstance(layer, Points):
            # Get the coordinates from the layer
            coords = layer.data
            
            # Print coordinate ranges
            print(f"\nCoordinate debug info for Points layer '{layer.name}':")
            for i, dim in enumerate(['x', 'y', 'z'][:coords.shape[1]]):
                values = coords[:, i]
                print(f"  {dim}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")
        
        elif isinstance(layer, Labels):
            # Get the features from the layer
            features = self.get_layer_tabular_data(layer)
            if features is not None:
                # Print coordinate ranges
                print(f"\nCoordinate debug info for Labels layer '{layer.name}':")
                for dim in ['centroid_x', 'centroid_y', 'centroid_z']:
                    if dim in features.columns:
                        values = features[dim].values
                        print(f"  {dim}: min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")

    def frame_changed(self, event):
        """Handle changes to the current frame.
        
        Args:
            event: Event that triggered the change.
        """
        # Update the current frame
        if self.viewer.dims.ndim > 0:
            old_frame = self.frame
            self.frame = self.viewer.dims.current_step[0]
            
            print(f"Frame changed from {old_frame} to {self.frame}")
            
            # Update point highlighting based on current slice
            self._update_point_highlighting()
            
            # If we have a Labels layer, we may need to update the features
            if self.analysed_layer is not None and isinstance(self.analysed_layer, Labels):
                # Check if we have features with a frame/time column
                features = self.get_layer_tabular_data(self.analysed_layer)
                if features is not None:
                    frame_columns = [col for col in features.columns if col.lower() in ['frame', 'time', 't']]
                    if frame_columns:
                        frame_column = frame_columns[0]
                        print(f"Found frame column: {frame_column}")
                        
                        # Filter features for the current frame
                        frame_features = features[features[frame_column] == self.frame]
                        if not frame_features.empty:
                            print(f"Found {len(frame_features)} features for frame {self.frame}")
                            
                            # Store the filtered features
                            self.features_df = frame_features
                            
                            # Replot with the filtered features
                            if hasattr(self, 'data_x') and hasattr(self, 'data_y'):
                                print("Replotting with filtered features")
                                self.plot()
                        else:
                            print(f"No features found for frame {self.frame}")
                    else:
                        print("No frame column found in features")
            
            # For Points layers, check if we need to update visibility based on frame
            elif self.analysed_layer is not None and isinstance(self.analysed_layer, Points):
                features = self.get_layer_tabular_data(self.analysed_layer)
                if features is not None:
                    frame_columns = [col for col in features.columns if col.lower() in ['frame', 'time', 't']]
                    if frame_columns:
                        frame_column = frame_columns[0]
                        print(f"Found frame column: {frame_column}")
                        
                        # Filter features for the current frame
                        frame_features = features[features[frame_column] == self.frame]
                        if not frame_features.empty:
                            print(f"Found {len(frame_features)} features for frame {self.frame}")
                            
                            # Store the filtered features
                            self.features_df = frame_features
                            
                            # Replot with the filtered features
                            if hasattr(self, 'data_x') and hasattr(self, 'data_y'):
                                print("Replotting with filtered features")
                                self.plot()
                        else:
                            print(f"No features found for frame {self.frame}")
                    else:
                        print("No frame column found in features")
                        
            # If we have z-values, update the plot highlighting
            if hasattr(self, 'z_values') and self.z_values is not None:
                self._update_point_highlighting()

    def _on_ndisplay_change(self, event=None):
        """Handle changes to the ndisplay mode.
        
        Args:
            event: Event that triggered the change (optional).
        """
        # Update the class layers when switching between 2D and 3D
        if self.analysed_layer is not None and self.class_assignments is not None:
            self._update_class_layers()

    def _on_active_layer_change(self, event=None):
        """Handle changes in the active layer in napari.
        
        Args:
            event: Event that triggered the change (optional).
        """
        # Get the active layer
        active_layer = self.viewer.layers.selection.active
        
        # Store the active layer for reference
        self.napari_active_layer = active_layer
        
        # Reset the last selection state
        self.last_selection_state = None

    def _check_napari_selection(self):
        """Periodically check for selection changes in the active layer."""
        if not hasattr(self, 'napari_active_layer') or self.napari_active_layer is None:
            return
        
        # Get the active layer
        layer = self.napari_active_layer
        
        # Only process Points and Labels layers
        if not isinstance(layer, (Points, Labels)):
            return
        
        # For Points layers, check the selected property
        if isinstance(layer, Points) and hasattr(layer, 'selected'):
            # Get the current selection state
            current_selection = layer.selected.copy() if hasattr(layer.selected, 'copy') else layer.selected
            
            # Check if the selection has changed
            if self.last_selection_state is None or not np.array_equal(current_selection, self.last_selection_state):
                # Update the last selection state
                self.last_selection_state = current_selection.copy() if hasattr(current_selection, 'copy') else current_selection
                
                # Get the selected indices
                selected_indices = np.where(current_selection)[0]
                
                # Store the selected indices
                self.napari_selected_indices = selected_indices
                
                # Update selection info
                if len(selected_indices) > 0:
                    self.selection_info_label.setText(f"{len(selected_indices)} points selected from napari viewer")
                    
                    # Highlight the selected points in the plot
                    if hasattr(self, 'plot_data') and self.plot_data is not None:
                        # Create a mask for the selected points
                        mask = np.zeros(self.plot_data.shape[0], dtype=bool)
                        
                        # Map the selected indices to the plot data indices
                        if hasattr(self, 'data_indices') and self.data_indices is not None:
                            # Find which indices in data_indices match the selected indices
                            for idx in selected_indices:
                                mask |= (self.data_indices == idx)
                        else:
                            # If no mapping is available, try direct indexing
                            if len(selected_indices) > 0 and np.max(selected_indices) < self.plot_data.shape[0]:
                                mask[selected_indices] = True
                        
                        # Highlight the selected points
                        self._highlight_napari_selection(selected_indices)
                else:
                    # Clear the highlighting if no points are selected
                    self._highlight_napari_selection(np.zeros(self.plot_data.shape[0] if hasattr(self, 'plot_data') else 0, dtype=bool))

    def _highlight_napari_selection(self, selected_indices):
        """Highlight selected points in napari.
        
        Args:
            selected_indices: Indices of selected points.
        """
        if self.analysed_layer is None or not isinstance(self.analysed_layer, (Points, Labels)):
            return
        
        # For Points layer
        if isinstance(self.analysed_layer, Points):
            # Store original colors
            if not hasattr(self, 'original_point_colors'):
                self.original_point_colors = self.analysed_layer.face_color.copy()
            
            # Clear current selection
            self.analysed_layer.selected_data = {}
            
            # Get the next class ID and color
            next_class = 1
            if hasattr(self, 'class_assignments') and self.class_assignments is not None and np.any(self.class_assignments > 0):
                next_class = np.max(self.class_assignments) + 1
            
            # Limit to the number of available colors
            color_idx = (next_class - 1) % len(self.class_colors)
            highlight_color = self.class_colors[color_idx]
            
            # Create a temporary bright color for the selected points
            face_colors = self.original_point_colors.copy()
            
            # Check if we have a single color or per-point colors
            if face_colors.ndim == 1:  # Single color for all points
                # Convert to per-point colors
                face_colors = np.tile(face_colors, (len(self.analysed_layer.data), 1))
            
            # Set the selected points to the class color
            for idx in selected_indices:
                if idx < len(self.analysed_layer.data):
                    face_colors[idx] = highlight_color  # Use class color with full opacity
            
            # Update the layer colors
            self.analysed_layer.face_color = face_colors
            
            # Also select the points to show their border
            self.analysed_layer.selected_data = set(selected_indices)
            
            # Set a timer to reset the color after 2 seconds
            if hasattr(self, 'color_reset_timer'):
                self.color_reset_timer.stop()
            else:
                self.color_reset_timer = QTimer()
                self.color_reset_timer.setSingleShot(True)
                self.color_reset_timer.timeout.connect(self._reset_point_colors)
            
            self.color_reset_timer.start(2000)  # 2 seconds
        
        # For Labels layer
        elif isinstance(self.analysed_layer, Labels) and len(selected_indices) > 0:
            # Check if we have features_df and it has a label column
            if not hasattr(self, 'features_df') or self.features_df is None:
                # Try to get features for the layer
                self.features_df = self.get_layer_tabular_data(self.analysed_layer)
                if self.features_df is None or 'label' not in self.features_df.columns:
                    # Can't highlight without features
                    return
            
            # Get the label values for the selected indices
            if 'label' in self.features_df.columns:
                selected_labels = []
                for idx in selected_indices:
                    if idx < len(self.features_df):
                        label = self.features_df.iloc[idx]['label']
                        selected_labels.append(label)
                
                if selected_labels:
                    # Create a temporary layer to highlight the selected labels
                    highlight_data = np.zeros_like(self.analysed_layer.data)
                    for label in selected_labels:
                        highlight_data[self.analysed_layer.data == label] = 1
                    
                    # Store the original visibility state
                    original_visible = self.analysed_layer.visible
                    
                    # Get the next class ID and color
                    next_class = 1
                    if hasattr(self, 'class_assignments') and self.class_assignments is not None and np.any(self.class_assignments > 0):
                        next_class = np.max(self.class_assignments) + 1
                    
                    # Limit to the number of available colors
                    color_idx = (next_class - 1) % len(self.class_colors)
                    highlight_color = self.class_colors[color_idx]
                    
                    # Convert to hex color for labels layer
                    hex_color = f"#{int(highlight_color[0]*255):02x}{int(highlight_color[1]*255):02x}{int(highlight_color[2]*255):02x}"
                    
                    # Check if highlight layer already exists
                    highlight_layer_name = f"{self.analysed_layer.name}_highlight"
                    if highlight_layer_name in self.viewer.layers:
                        # Update existing layer
                        self.viewer.layers[highlight_layer_name].data = highlight_data
                        self.viewer.layers[highlight_layer_name].colormap = {1: hex_color}
                    else:
                        # Create new layer
                        self.viewer.add_labels(
                            highlight_data,
                            name=highlight_layer_name,
                            colormap={1: hex_color},
                            opacity=0.7
                        )
                    
                    # Make sure the original layer is still visible
                    self.analysed_layer.visible = original_visible
                    

    def _reset_point_colors(self):
        """Reset point colors to their original values."""
        print("Reset point colors to original values")
        if hasattr(self, 'analysed_layer') and isinstance(self.analysed_layer, Points) and hasattr(self, 'original_point_colors'):
            # Make sure original_point_colors is not None
            if self.original_point_colors is not None:
                self.analysed_layer.face_color = self.original_point_colors.copy()
            else:
                # If original colors are None, set a default color
                default_color = np.array([0.5, 0.5, 1.0, 0.7])  # Light blue
                if len(self.analysed_layer.data) > 0:
                    self.analysed_layer.face_color = np.tile(default_color, (len(self.analysed_layer.data), 1))
            
            # Clear selection
            self.analysed_layer.selected_data = {}


    def _remove_class_layers(self):
        """Remove all class layers from napari."""
        if self.viewer is not None and hasattr(self.viewer, 'layers'):
            for layer in self.viewer.layers:
                if layer.name.endswith('_classes'):
                    self.viewer.layers.remove(layer.name)

    def update_colormap(self):
        """Update the colormap used for plotting."""
        # Get the selected colormap
        self.colormap_name = self.colormap_combo.currentText()
        print(f"Updated colormap to: {self.colormap_name}")
        
        # Replot if data is available
        if self.data_x is not None and self.data_y is not None:
            self.plot()

    def plot_clicked(self):
        """Handle the run button click."""
        # Check if a layer is selected
        layer = self.get_selected_layer()
        if layer is None:
            print("No layer selected")
            return
        
        # Check if the layer has features
        features = self.get_layer_tabular_data(layer)
        if features is None:
            print(f"Layer {layer.name} has no features")
            return
        
        # Check if axes are selected
        if self.x_combo.currentText() == "" or self.y_combo.currentText() == "":
            print("Please select axes for plotting!")
            return
        
        # Run the plotting
        self.plot()

    def plot(self):
        """Run the plotting operation."""
        # Get the selected layer
        self.analysed_layer = self.get_selected_layer()
        if self.analysed_layer is None:
            print("No layer selected")
            return
        
        # Get the features
        all_features = self.get_layer_tabular_data(self.analysed_layer)
        if all_features is None:
            print(f"Layer {self.analysed_layer.name} has no features")
            return
            
        # Check if we need to filter features by frame
        features = all_features
        frame_columns = [col for col in all_features.columns if col.lower() in ['frame', 'time', 't']]
        if frame_columns:
            frame_column = frame_columns[0]
            print(f"Found frame column: {frame_column}")
            
            # Get the current frame
            current_frame = self.frame
            
            # Filter features for the current frame
            frame_features = all_features[all_features[frame_column] == current_frame]
            if not frame_features.empty:
                print(f"Filtering to {len(frame_features)} features for frame {current_frame}")
                features = frame_features
            else:
                print(f"No features found for frame {current_frame}, using all features")
        
        # Store the features for later use
        self.features_df = features
        
        # Get the selected axes
        x_axis = self.x_combo.currentText()
        y_axis = self.y_combo.currentText()
        
        # Check if the axes are valid
        if x_axis not in features.columns or y_axis not in features.columns:
            print(f"Invalid axes: {x_axis}, {y_axis}")
            return
        
        # Get the data
        x_data = features[x_axis].values
        y_data = features[y_axis].values
        
        # Store the data for later use
        self.data_x = x_data
        self.data_y = y_data
        
        # Store the indices for mapping between data and plot
        self.data_indices = np.arange(len(x_data))
        
        # Set the axis labels
        self.canvas.set_labels(x_axis, y_axis)
        
        # Get the selected color feature
        color_feature = self.cluster_combo.currentText()
        
        # Import matplotlib for colormaps
        try:
            import matplotlib.cm as cm
        except ImportError:
            print("Warning: matplotlib not available, using default coloring")
            cm = None
        
        # Determine how to color the points
        if color_feature == "None":
            # Use the existing cluster detection logic
            cluster_column = None
            for col in features.columns:
                if "CLUSTER" in col.upper() or "CLASS" in col.upper():
                    cluster_column = col
                    break
            
            # Get colors based on clustering
            if cluster_column is not None:
                cluster_ids = features[cluster_column].values
                self.colors = get_cluster_colors(cluster_ids, self.colormap_name)
                print(f"Using {cluster_column} for coloring points")
            else:
                # Default coloring
                self.colors = np.ones((len(x_data), 4)) * np.array([0.5, 0.5, 1.0, 0.7])  # Light blue
        else:
            # Use the selected feature for coloring
            if color_feature in features.columns:
                feature_values = features[color_feature].values
                
                # Normalize the values to [0, 1] range for coloring
                if len(feature_values) > 0:
                    min_val = np.min(feature_values)
                    max_val = np.max(feature_values)
                    
                    # Avoid division by zero
                    if max_val > min_val:
                        normalized_values = (feature_values - min_val) / (max_val - min_val)
                    else:
                        normalized_values = np.zeros_like(feature_values)
                    
                    # Create a colormap
                    if cm is not None:
                        try:
                            cmap = cm.get_cmap(self.colormap_name)
                            # Apply the colormap to get RGBA colors
                            self.colors = cmap(normalized_values)
                            print(f"Using {color_feature} for coloring points (min: {min_val}, max: {max_val})")
                        except Exception as e:
                            print(f"Error creating colormap: {e}")
                            # Default coloring if colormap fails
                            self.colors = np.ones((len(x_data), 4)) * np.array([0.5, 0.5, 1.0, 0.7])  # Light blue
                    else:
                        # Default coloring if matplotlib not available
                        self.colors = np.ones((len(x_data), 4)) * np.array([0.5, 0.5, 1.0, 0.7])  # Light blue
                else:
                    # Default coloring if no values
                    self.colors = np.ones((len(x_data), 4)) * np.array([0.5, 0.5, 1.0, 0.7])  # Light blue
            else:
                # Default coloring if feature not found
                self.colors = np.ones((len(x_data), 4)) * np.array([0.5, 0.5, 1.0, 0.7])  # Light blue
                print(f"Feature {color_feature} not found, using default coloring")
        
        # Initialize sizes with default values
        self.sizes = np.ones(len(x_data)) * 10
        
        # Get point sizes based on z-dimension if available
        z_column = None
        for col in ["z", "depth", "slice", "frame", "time", "t", "centroid_z"]:
            if col in features.columns:
                z_column = col
                break
        
        if z_column is not None:
            self.z_values = features[z_column].values
            # Calculate weights based on distance from current frame
            weights = points_in_current_slice(self.z_values, self.frame)
            
            # Create size values (larger for current slice, smaller with distance)
            base_size = 10  # Base point size
            self.sizes = base_size * (0.5 + 0.5 * weights)  # Scale between 50% and 100% of base size
            
            # Update opacities in colors
            for i in range(len(self.colors)):
                self.colors[i, 3] = max(weights[i], 0.1)  # Ensure minimum opacity of 0.1
        else:
            # Ensure all points have at least 0.5 opacity
            if self.colors.shape[1] == 4:  # If alpha channel exists
                self.colors[:, 3] = np.maximum(self.colors[:, 3], 0.5)
        
        # Store the original colors for selection highlighting
        self.original_colors = self.colors.copy()
        
        # Pass data to the canvas for selection
        self.canvas.x_data = x_data
        self.canvas.y_data = y_data
        self.canvas.original_colors = self.original_colors
        
        # Create the scatter plot
        try:
            self.canvas.make_scatter_plot(x_data, y_data, self.colors, self.sizes)
            print(f"Created scatter plot with {len(x_data)} points")
        except Exception as e:
            print(f"Error creating scatter plot: {e}")
        
        # Update class visualization if we have class assignments
        if hasattr(self, 'class_assignments') and self.class_assignments is not None and len(self.class_assignments) > 0:
            # Make sure class_assignments has the right length
            if len(self.class_assignments) != len(x_data):
                print(f"Resetting class assignments (length mismatch: {len(self.class_assignments)} vs {len(x_data)})")
                self.class_assignments = np.zeros(len(x_data), dtype=int)
            self._update_class_visualization()

    def estimate_bin_number(self):
        """Estimate a reasonable number of bins for histograms."""
        # Get the selected layer
        layer = self.get_selected_layer()
        if layer is None:
            return 100
        
        # Get the features
        features = self.get_layer_tabular_data(layer)
        if features is None:
            return 100
        
        # Get the selected axis
        axis = self.x_combo.currentText()
        if axis not in features.columns:
            return 100
        
        # Get the data
        data = features[axis].values
        
        # Estimate the bin number
        return estimate_bin_number(data)

    def clear_classes(self):
        """Clear all class assignments."""
        if hasattr(self, 'class_assignments') and self.class_assignments is not None:
            self.class_assignments = np.zeros_like(self.class_assignments)
            self._update_class_visualization()
            
            # Remove class layers from napari
            self._remove_class_layers()

    def _update_class_visualization(self):
        """Update the visualization of classes in the plot."""
        if self.analysed_layer is None or not hasattr(self.canvas, 'scatter') or self.canvas.scatter is None:
            return
        
        if not hasattr(self, 'class_assignments') or self.class_assignments is None or len(self.class_assignments) == 0 or np.all(self.class_assignments == 0):
            # If no class assignments, use original colors
            if hasattr(self, 'colors') and self.colors is not None:
                # Make sure all points have some opacity
                colors = self.colors.copy()
                if colors.shape[1] == 4:  # If alpha channel exists
                    # Ensure all points have at least 0.5 opacity
                    colors[:, 3] = np.maximum(colors[:, 3], 0.5)
                
                self.canvas.scatter.set_data(
                    pos=np.column_stack([self.data_x, self.data_y]),
                    face_color=colors,
                    size=self.sizes,
                    edge_width=0
                )
            return
        
        # Create colors based on class assignments
        colors = self.original_colors.copy() if hasattr(self, 'original_colors') else self.colors.copy()
        
        # Make sure all points have some opacity
        if colors.shape[1] == 4:  # If alpha channel exists
            # Ensure all points have at least 0.5 opacity
            colors[:, 3] = np.maximum(colors[:, 3], 0.5)
        
        # Assign colors to each class
        for class_id in range(1, np.max(self.class_assignments) + 1):
            # Create mask for this class
            mask = self.class_assignments == class_id
            
            # Skip if no points in this class
            if not np.any(mask):
                continue
            
            # Get color for this class
            color_idx = (class_id - 1) % len(self.class_colors)
            color = self.class_colors[color_idx]
            
            # Assign color with full opacity
            colors[mask, :3] = color[:3]
            if colors.shape[1] == 4:  # If alpha channel exists
                colors[mask, 3] = 1.0  # Full opacity for classified points
        
        # Update scatter plot
        self.canvas.scatter.set_data(
            pos=np.column_stack([self.data_x, self.data_y]),
            face_color=colors,
            size=self.sizes,
            edge_width=0
        )
        
        # Update class layers in napari
        self._update_class_layers()

    def _update_class_layers(self):
        """Update the class layers in napari."""
        if self.viewer is None or self.analysed_layer is None:
            return
        
        # Check if we have any class assignments
        if not hasattr(self, 'class_assignments') or self.class_assignments is None or not np.any(self.class_assignments > 0):
            return
            
        # For Points layer
        if isinstance(self.analysed_layer, Points):
            # Get the points that have class assignments
            class_mask = self.class_assignments > 0
            if not np.any(class_mask):
                return
            
            # Get the points and their classes
            classified_points = self.analysed_layer.data[class_mask]
            class_ids = self.class_assignments[class_mask]
            
            # Create colors for each point
            point_colors = []
            for class_id in class_ids:
                color_idx = (class_id - 1) % len(self.class_colors)
                point_colors.append(self.class_colors[color_idx])
            
            # Check if class layer already exists
            class_layer_name = f"{self.analysed_layer.name}_classes"
            if class_layer_name in self.viewer.layers:
                # Update existing layer
                self.viewer.layers[class_layer_name].data = classified_points
                self.viewer.layers[class_layer_name].face_color = point_colors
                self.viewer.layers[class_layer_name].n_dimensional = True
            else:
                # Create new layer
                self.viewer.add_points(
                    classified_points,
                    name=class_layer_name,
                    size=10,  # Use a fixed size for all points
                    face_color=point_colors,
                    border_color='white',
                    border_width=0.5,
                    opacity=0.7,
                    n_dimensional=True
                )
        
        # For Labels layer
        elif isinstance(self.analysed_layer, Labels):
            # Get the labels data
            labels_data = self.analysed_layer.data
            
            # Create a new labels data with only classified labels
            classified_labels = np.zeros_like(labels_data)
            
            # Get features for mapping
            if not hasattr(self, 'features_df') or self.features_df is None:
                self.features_df = self.get_layer_tabular_data(self.analysed_layer)
            
            if self.features_df is None:
                print("No features available for Labels layer")
                return
                
            # Check if 'label' column exists in features
            if 'label' not in self.features_df.columns:
                # Try to find a column that might contain label values
                label_columns = [col for col in self.features_df.columns if 'label' in col.lower()]
                if label_columns:
                    label_column = label_columns[0]
                    print(f"Using '{label_column}' as label column")
                else:
                    print("No label column found in features")
                    return
            else:
                label_column = 'label'
                
            print(f"Using label column: {label_column}")
            print(f"Features columns: {', '.join(self.features_df.columns)}")
            print(f"Class assignments shape: {self.class_assignments.shape}")
            print(f"Features shape: {self.features_df.shape}")
            
            # Map class assignments to labels
            for i, class_id in enumerate(self.class_assignments):
                if class_id > 0 and i < len(self.features_df):
                    try:
                        label_value = self.features_df.iloc[i][label_column]
                        # Convert to int if it's not already
                        if not isinstance(label_value, (int, np.integer)):
                            label_value = int(label_value)
                            
                        # Set the label value in the classified_labels array
                        classified_labels[labels_data == label_value] = class_id
                        print(f"Mapped class {class_id} to label {label_value}")
                    except (KeyError, ValueError, IndexError) as e:
                        print(f"Error mapping class {class_id} to label: {e}")
            
            # Create colors for each class
            class_colors = {}
            for class_id in range(1, np.max(self.class_assignments) + 1):
                color_idx = (class_id - 1) % len(self.class_colors)
                color = self.class_colors[color_idx]
                # Convert to hex color
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(color[0] * 255),
                    int(color[1] * 255),
                    int(color[2] * 255))
                class_colors[class_id] = hex_color
            
            # Check if any labels were classified
            if np.any(classified_labels > 0):
                # Check if class layer already exists
                class_layer_name = f"{self.analysed_layer.name}_classes"
                if class_layer_name in self.viewer.layers:
                    # Update existing layer
                    self.viewer.layers[class_layer_name].data = classified_labels
                    # Use colormap for Labels layer
                    self.viewer.layers[class_layer_name].colormap = class_colors
                else:
                    # Create new layer
                    self.viewer.add_labels(
                        classified_labels,
                        name=class_layer_name,
                        colormap=class_colors,
                        opacity=0.7
                    )
                print(f"Created/updated class layer with {len(class_colors)} classes")
            else:
                print("No labels were classified")
                # Remove the class layer if it exists
                class_layer_name = f"{self.analysed_layer.name}_classes"
                if class_layer_name in self.viewer.layers:
                    self.viewer.layers.remove(class_layer_name)

    def on_selection_changed(self, selected_indices):
        """Handle selection changes from the canvas.
        
        Args:
            selected_indices: Indices of selected points.
        """
        if selected_indices is None or len(selected_indices) == 0:
            # Clear all classes if an empty selection is made
            self.clear_classes()
            return
            
        if self.analysed_layer is None:
            return
            
        # Initialize class assignments if needed
        if not hasattr(self, 'class_assignments') or self.class_assignments is None:
            self.class_assignments = np.zeros(len(self.data_x), dtype=int)
            
        # Get the next class ID
        next_class_id = np.max(self.class_assignments) + 1 if np.any(self.class_assignments > 0) else 1
        
        # Assign the selected points to the new class
        for idx in selected_indices:
            if idx < len(self.class_assignments):
                self.class_assignments[idx] = next_class_id
                
        # Update the class visualization
        self._update_class_visualization()
        
        # Update the selection info
        n_selected = len(selected_indices)
        self.selection_info_label.setText(f"Selected: {n_selected} points (Class {next_class_id})")
        
        # Highlight the selected points in napari
        if isinstance(self.analysed_layer, Points):
            # For Points layer, highlight the selected points
            self._highlight_napari_selection(selected_indices)
        elif isinstance(self.analysed_layer, Labels) and hasattr(self, 'features_df') and self.features_df is not None:
            # For Labels layer, we need to map the selected indices to label values
            # Find the label column
            label_column = 'label'
            if label_column not in self.features_df.columns:
                # Try to find a column that might contain label values
                label_columns = [col for col in self.features_df.columns if 'label' in col.lower()]
                if label_columns:
                    label_column = label_columns[0]
                else:
                    print("No label column found in features")
                    return
                    
            # Get the label values for the selected indices
            try:
                selected_labels = []
                for idx in selected_indices:
                    if idx < len(self.features_df):
                        label_value = self.features_df.iloc[idx][label_column]
                        # Convert to int if it's not already
                        if not isinstance(label_value, (int, np.integer)):
                            label_value = int(label_value)
                        selected_labels.append(label_value)
                
                # Highlight the selected labels in napari
                if selected_labels:
                    # Create a temporary layer to highlight the selected labels
                    temp_layer_name = f"{self.analysed_layer.name}_selection"
                    
                    # Create a mask for the selected labels
                    mask = np.zeros_like(self.analysed_layer.data, dtype=bool)
                    for label in selected_labels:
                        mask = mask | (self.analysed_layer.data == label)
                    
                    # Create a temporary layer with the selected labels
                    if temp_layer_name in self.viewer.layers:
                        # Update existing layer
                        self.viewer.layers[temp_layer_name].data = mask
                    else:
                        # Create new layer
                        self.viewer.add_image(
                            mask.astype(float),
                            name=temp_layer_name,
                            colormap='yellow',
                            blending='additive',
                            opacity=0.5
                        )
                    
                    # Schedule removal of the temporary layer
                    if hasattr(self, '_selection_timer') and self._selection_timer is not None:
                        self._selection_timer.stop()
                    
                    self._selection_timer = QTimer()
                    self._selection_timer.setSingleShot(True)
                    self._selection_timer.timeout.connect(
                        lambda: self._remove_temp_layer(temp_layer_name)
                    )
                    self._selection_timer.start(2000)  # Remove after 2 seconds
                    
                    print(f"Highlighted {len(selected_labels)} labels in napari")
            except Exception as e:
                print(f"Error highlighting labels: {e}")
                
        # Update the class layers in napari
        self._update_class_layers()

    def on_point_clicked(self, point_idx):
        """Handle point click events from the Vispy canvas.
        
        Args:
            point_idx: Index of the clicked point.
        """
        if self.analysed_layer is None or not hasattr(self, 'data_x') or not hasattr(self, 'data_y'):
            return
        
        print(f"Point {point_idx} clicked in Vispy plotter")
        
        # Get the corresponding index in the napari layer
        napari_idx = None
        
        # Try to map using the data_indices if available
        if hasattr(self, 'data_indices') and self.data_indices is not None:
            if point_idx < len(self.data_indices):
                napari_idx = self.data_indices[point_idx]
                print(f"Mapped to napari index {napari_idx} using data_indices")
        else:
            # If no mapping is available, use the point index directly
            napari_idx = point_idx
            print(f"Using direct index mapping: {napari_idx}")
        
        # If we have a valid index, highlight the point in napari
        if napari_idx is not None:
            # For Points layer
            if isinstance(self.analysed_layer, Points):
                # Store original colors
                if not hasattr(self, 'original_point_colors'):
                    self.original_point_colors = self.analysed_layer.face_color.copy()
                
                # Clear current selection
                self.analysed_layer.selected_data = {}
                
                # Select the point
                if napari_idx < len(self.analysed_layer.data):
                    # Create a temporary bright color for the selected point
                    face_colors = self.original_point_colors.copy()
                    
                    # Check if we have a single color or per-point colors
                    if face_colors.ndim == 1:  # Single color for all points
                        # Convert to per-point colors
                        face_colors = np.tile(face_colors, (len(self.analysed_layer.data), 1))
                    
                    # Set the selected point to bright green
                    face_colors[napari_idx] = [0, 1, 0, 1]  # Bright green with full opacity
                    
                    # Update the layer colors
                    self.analysed_layer.face_color = face_colors
                    
                    # Also select the point to show its border
                    self.analysed_layer.selected_data = {napari_idx}
                    
                    print(f"Selected point {napari_idx} in napari layer {self.analysed_layer.name}")
                    
                    # Get the coordinates of the selected point
                    point_coords = self.analysed_layer.data[napari_idx]
                    print(f"Napari coordinates: {point_coords}")
                    
                    # Update selection info
                    self.selection_info_label.setText(f"Point {napari_idx} selected from Vispy plotter")
                    
                    # Set a timer to reset the color after 2 seconds
                    if hasattr(self, 'color_reset_timer'):
                        self.color_reset_timer.stop()
                    else:
                        self.color_reset_timer = QTimer()
                        self.color_reset_timer.setSingleShot(True)
                        self.color_reset_timer.timeout.connect(self._reset_point_colors)
                    
                    self.color_reset_timer.start(2000)  # 2 seconds
            
            # For Labels layer
            elif isinstance(self.analysed_layer, Labels):
                # For Labels, we need to find the label with the matching value
                # Try to get features_df if it's None
                if not hasattr(self, 'features_df') or self.features_df is None:
                    self.features_df = self.get_layer_tabular_data(self.analysed_layer)
                
                # Check if we have valid features data
                if self.features_df is not None and 'label' in self.features_df.columns:
                    label_value = napari_idx
                    print(f"Highlighting label {label_value} in napari")
                    
                    # Create a temporary layer to highlight the selected label
                    highlight_data = np.zeros_like(self.analysed_layer.data)
                    highlight_data[self.analysed_layer.data == label_value] = 1
                    
                    # Store the original visibility state
                    original_visible = self.analysed_layer.visible
                    
                    # Check if highlight layer already exists
                    highlight_layer_name = f"{self.analysed_layer.name}_highlight"
                    if highlight_layer_name in self.viewer.layers:
                        # Update existing layer
                        self.viewer.layers[highlight_layer_name].data = highlight_data
                    else:
                        # Create new layer
                        self.viewer.add_labels(
                            highlight_data,
                            name=highlight_layer_name,
                            colormap={1: 'lime'},  # Bright green
                            opacity=0.7
                        )
                    
                    # Make sure the original layer is still visible
                    self.analysed_layer.visible = original_visible
                    
                    # Update selection info
                    self.selection_info_label.setText(f"Label {label_value} selected from Vispy plotter")

    def _on_napari_mouse_click(self, viewer, event):
        """Handle mouse click events in napari.
        
        Args:
            viewer: The napari viewer.
            event: The mouse event.
        """
        # Only process click events (not drag events)
        if event.type == 'mouse_press':
            # Get the active layer
            active_layer = self.viewer.layers.selection.active
            
            # Only process Points and Labels layers
            if active_layer is None or not isinstance(active_layer, (Points, Labels)):
                return
            
            # Get the value under the cursor
            value = active_layer.get_value(
                position=viewer.cursor.position,
                view_direction=viewer.camera.view_direction,
                dims_displayed=viewer.dims.displayed,
                world=True
            )
            
            # For Points layers, value is the index of the clicked point
            if isinstance(active_layer, Points) and value is not None:
                print(f"Clicked on point with index {value} in layer {active_layer.name}")
                
                # Get the coordinates of the clicked point
                if value < len(active_layer.data):
                    point_coords = active_layer.data[value]
                    print(f"Point coordinates in napari: {point_coords}")
                    
                    # Highlight the point in the plot
                    if self.analysed_layer == active_layer and hasattr(self, 'data_x') and hasattr(self, 'data_y'):
                        # Create a mask for the selected point
                        mask = np.zeros(len(self.data_x), dtype=bool)
                        
                        # Try direct indexing first
                        if value < len(self.data_x):
                            mask[value] = True
                            print(f"Mapped to plot using direct indexing")
                            print(f"Plot coordinates: ({self.data_x[value]}, {self.data_y[value]})")
                        # If we have data_indices, try to map using them
                        elif hasattr(self, 'data_indices') and self.data_indices is not None:
                            idx = np.where(self.data_indices == value)[0]
                            if len(idx) > 0:
                                mask[idx[0]] = True
                                print(f"Mapped to plot index {idx[0]} using data_indices")
                                print(f"Plot coordinates: ({self.data_x[idx[0]]}, {self.data_y[idx[0]]})")
                        
                        # Highlight the selected point in the plot
                        if np.any(mask):
                            self.canvas._highlight_selected_points(mask)
                            
                            # Update selection info
                            self.selection_info_label.setText(f"Point {value} selected from napari viewer")
            
            # For Labels layers, value is the label value at the clicked position
            elif isinstance(active_layer, Labels) and value > 0:  # 0 is background
                print(f"Clicked on label with value {value} in layer {active_layer.name}")
                
                # Get the position of the clicked point
                position = viewer.cursor.position
                print(f"Clicked at position: {position}")
                
                # Highlight the label in the plot
                if hasattr(self, 'data_x') and hasattr(self, 'data_y'):
                    # Create a mask for the selected label
                    mask = np.zeros(len(self.data_x), dtype=bool)
                    
                    # Try to map using label value if available
                    if hasattr(self, 'features_df') and self.analysed_layer == active_layer:
                        if 'label' in self.features_df.columns:
                            # Find rows with matching label value
                            matches = self.features_df['label'] == value
                            if np.any(matches):
                                idx = np.where(matches)[0]
                                mask[idx] = True
                                print(f"Mapped to plot indices {idx} using label value")
                                if len(idx) > 0:
                                    print(f"Plot coordinates: ({self.data_x[idx[0]]}, {self.data_y[idx[0]]})")
                        else:
                            # If no exact match, find the closest centroid
                            if all(col in self.features_df.columns for col in ['centroid_x', 'centroid_y', 'centroid_z']):
                                distances = np.sqrt(
                                    (self.features_df['centroid_x'] - position[0])**2 +
                                    (self.features_df['centroid_y'] - position[1])**2 +
                                    (self.features_df['centroid_z'] - position[2])**2
                                )
                                closest_idx = np.argmin(distances)
                                mask[closest_idx] = True
                                print(f"Mapped to plot index {closest_idx} using closest centroid")
                                print(f"Plot coordinates: ({self.data_x[closest_idx]}, {self.data_y[closest_idx]})")
                    else:
                        # Fall back to index mapping
                        if hasattr(self, 'data_indices') and self.data_indices is not None:
                            # Find which indices in data_indices match the selected label
                            idx = np.where(self.data_indices == value)[0]
                            if len(idx) > 0:
                                mask[idx] = True
                                print(f"Mapped to plot indices {idx} using data_indices")
                                if len(idx) > 0:
                                    print(f"Plot coordinates: ({self.data_x[idx[0]]}, {self.data_y[idx[0]]})")
                
                    # Highlight the selected label
                    if np.any(mask):
                        self.canvas._highlight_selected_points(mask)
                        
                        # Update selection info
                        self.selection_info_label.setText(f"Label {value} selected from napari viewer")

    def _update_point_highlighting(self):
        """Update point highlighting based on current slice."""
        if self.analysed_layer is None or not hasattr(self.canvas, 'scatter') or self.canvas.scatter is None:
            return
        
        # Get the current frame
        current_frame = self.frame
        print(f"Updating point highlighting for frame {current_frame}")
        
        # Get the layer data
        features = self.get_layer_tabular_data(self.analysed_layer)
        if features is None or len(features) == 0:
            print("No features available for highlighting")
            return
        
        # Check if we have a z or time dimension to use for highlighting
        z_columns = [col for col in features.columns if col.lower() in ['z', 'depth', 'slice', 'frame', 'time', 't', 'centroid_z']]
        if not z_columns:
            print("No z/time dimension found for highlighting")
            return
        
        # Use the first matching column
        z_column = z_columns[0]
        if z_column not in features.columns:
            print(f"Column {z_column} not found in features")
            return
        
        print(f"Using column {z_column} for slice-based highlighting")
        
        # Get z values
        z_values = features[z_column].to_numpy()
        self.z_values = z_values  # Store for later use
        
        # Calculate weights based on distance from current frame
        weights = points_in_current_slice(z_values, current_frame)
        
        # Create opacity values (1.0 for current slice, decreasing with distance)
        opacities = weights
        
        # Create size values (larger for current slice, smaller with distance)
        base_size = 10  # Base point size
        self.sizes = base_size * (0.5 + 0.5 * weights)  # Scale between 50% and 100% of base size
        
        # Update colors with new opacities
        if hasattr(self, 'original_colors') and self.original_colors is not None:
            colors = self.original_colors.copy()
        elif hasattr(self, 'colors') and self.colors is not None:
            colors = self.colors.copy()
        else:
            print("No colors available for highlighting")
            return
            
        if colors.shape[1] >= 4:  # If alpha channel exists
            for i in range(len(colors)):
                colors[i, 3] = max(opacities[i], 0.1)  # Ensure minimum opacity of 0.1
        
        # Store the updated colors
        self.colors = colors.copy()
        
        # Update scatter plot
        try:
            self.canvas.scatter.set_data(
                pos=np.column_stack([self.data_x, self.data_y]),
                face_color=colors,
                size=self.sizes,
                edge_width=0
            )
            print(f"Updated scatter plot with {len(colors)} points")
        except Exception as e:
            print(f"Error updating scatter plot: {e}")
        
        # If we have selected points, update their highlighting too
        if hasattr(self.canvas, 'selected_mask') and self.canvas.selected_mask is not None:
            mask = self.canvas.selected_mask
            if np.any(mask):
                highlight_colors = colors.copy()
                # Apply yellow color to selected points while preserving opacity
                for i in np.where(mask)[0]:
                    highlight_colors[i, 0:3] = [1.0, 1.0, 0.0]  # Yellow
                
                try:
                    self.canvas.scatter.set_data(
                        pos=np.column_stack([self.data_x, self.data_y]),
                        face_color=highlight_colors,
                        size=self.sizes,
                        edge_width=0
                    )
                    print(f"Updated scatter plot with {np.sum(mask)} selected points")
                except Exception as e:
                    print(f"Error updating scatter plot with selections: {e}")
                    
        # If we have class assignments, update the class visualization
        if hasattr(self, 'class_assignments') and self.class_assignments is not None and np.any(self.class_assignments > 0):
            self._update_class_visualization()

    def extract_features_from_selected_layer(self):
        """Extract features from the selected layer."""
        # Get the selected layer
        layer_name = self.layer_combo.currentText()
        if not layer_name or layer_name not in self.viewer.layers:
            self.selection_info_label.setText("No layer selected")
            return
            
        layer = self.viewer.layers[layer_name]
        
        # Check if the layer already has features
        if hasattr(layer, 'features') and layer.features is not None and len(layer.features) > 0:
            # Ask for confirmation
            from qtpy.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, 
                'Confirm Feature Extraction',
                f"Layer '{layer_name}' already has features. Do you want to replace them?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
        
        # Extract features based on the layer type
        from napari.layers import Image, Points, Labels, Shapes, Vectors, Surface
        
        if isinstance(layer, Image):
            # Extract features from image layer
            success = self.extract_image_features(layer)
            if not success:
                return
                
        elif isinstance(layer, Points):
            # For Points layers, create features from coordinates
            import pandas as pd
            import numpy as np
            
            # Get the coordinates
            coords = layer.data
            
            # Create a DataFrame with the coordinates
            if coords.shape[1] == 2:
                features = pd.DataFrame({
                    'x': coords[:, 0],
                    'y': coords[:, 1]
                })
            elif coords.shape[1] == 3:
                features = pd.DataFrame({
                    'x': coords[:, 0],
                    'y': coords[:, 1],
                    'z': coords[:, 2]
                })
            else:
                features = pd.DataFrame({
                    f'dim_{i}': coords[:, i] for i in range(coords.shape[1])
                })
                
            # Add size and color as features if available
            if hasattr(layer, 'size') and layer.size is not None:
                if isinstance(layer.size, (int, float)):
                    features['size'] = layer.size
                elif isinstance(layer.size, np.ndarray) and len(layer.size) == len(coords):
                    features['size'] = layer.size
                    
            if hasattr(layer, 'face_color') and layer.face_color is not None:
                if isinstance(layer.face_color, np.ndarray) and layer.face_color.shape[0] == len(coords):
                    # Convert RGBA to hex for easier visualization
                    hex_colors = []
                    for color in layer.face_color:
                        r, g, b = [int(c * 255) for c in color[:3]]
                        hex_colors.append(f'#{r:02x}{g:02x}{b:02x}')
                    features['color'] = hex_colors
            
            # Set the features on the layer
            layer.features = features
            
            # Update the UI
            self.selection_info_label.setText(
                f"Created {len(features)} features from Points layer '{layer_name}'"
            )
            
        elif isinstance(layer, Labels):
            # For Labels layers, create features from region properties
            import pandas as pd
            from skimage import measure
            
            # Get the labels data
            data = layer.data
            
            # Calculate region properties
            try:
                props = measure.regionprops_table(
                    data, 
                    properties=('label', 'area', 'centroid', 'bbox', 'mean_intensity')
                )
                features = pd.DataFrame(props)
                
                # Rename columns for better readability
                if 'centroid-0' in features.columns:
                    features.rename(columns={
                        'centroid-0': 'centroid_y',
                        'centroid-1': 'centroid_x'
                    }, inplace=True)
                    
                if data.ndim == 3 and 'centroid-2' in features.columns:
                    features.rename(columns={
                        'centroid-0': 'centroid_z',
                        'centroid-1': 'centroid_y',
                        'centroid-2': 'centroid_x'
                    }, inplace=True)
                
                # Set the features on the layer
                layer.features = features
                
                # Update the UI
                self.selection_info_label.setText(
                    f"Extracted {len(features)} features from Labels layer '{layer_name}'"
                )
            except Exception as e:
                self.selection_info_label.setText(
                    f"Error extracting features from Labels layer '{layer_name}': {str(e)}"
                )
                return
                
        else:
            # Other layer types not supported yet
            self.selection_info_label.setText(
                f"Feature extraction not supported for layer type '{type(layer).__name__}'"
            )
            return
            
        # Update the axes lists
        self.update_axes_and_clustering_id_lists()

    def _remove_temp_layer(self, layer_name):
        """Remove a temporary layer from napari.
        
        Args:
            layer_name: Name of the layer to remove.
        """
        if self.viewer is not None and layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)
            print(f"Removed temporary layer: {layer_name}")
            
        # Reset the selection timer
        if hasattr(self, '_selection_timer'):
            self._selection_timer = None
