"""
Vispy-based plotter for napari-iohub.

This module provides a high-performance plotter using Vispy for visualizing
large datasets within napari.
"""

import numpy as np
from enum import Enum, auto

from vispy import scene, app
from vispy.scene import visuals
from vispy.color import get_colormap, Color

from napari.layers import Image, Labels, Layer, Points, Surface
from napari.utils.colormaps import ALL_COLORMAPS

from qtpy import QtWidgets, QtCore
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class PlottingType(Enum):
    SCATTER = auto()
    HISTOGRAM = auto()


class VispyCanvas(QWidget):
    """Canvas for Vispy visualization with lasso selection capability"""
    
    # Signal emitted when points are selected
    selection_changed = Signal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create a scene canvas with a transparent background
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='transparent')
        self.canvas.create_native()
        self.canvas.native.setParent(self)
        
        # Create a view for the canvas
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.view.camera.aspect = 1
        
        # Fix the camera orientation
        self.view.camera.flip = (False, True, False)
        
        # Set up grid lines
        self.grid = scene.visuals.GridLines(parent=self.view.scene)
        
        # Set up axis
        self.axis = scene.visuals.Axis(pos=[[0, 0], [1, 0]], 
                                      domain=(0, 1),
                                      tick_direction=(0, -1),
                                      font_size=8,
                                      axis_color='white',
                                      tick_color='white',
                                      text_color='white',
                                      parent=self.view.scene)
        
        # Set up axis labels
        self.x_label = scene.visuals.Text("X", pos=(0.5, -0.05), color='white', 
                                         anchor_x='center', anchor_y='top',
                                         font_size=10, parent=self.view.scene)
        self.y_label = scene.visuals.Text("Y", pos=(-0.05, 0.5), color='white',
                                         anchor_x='right', anchor_y='center',
                                         font_size=10, rotation=-90, parent=self.view.scene)
        
        # Set up scatter plot
        self.scatter = None
        self.histogram = None
        self.x_data = None
        self.y_data = None
        self.colors = None
        self.sizes = None
        
        # Set up selection
        self.is_selecting = False
        self.selection_path = []
        self.selection_line = None
        self.selected_mask = None
        
        # Connect events
        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.canvas.events.mouse_release.connect(self._on_mouse_release)
        self.canvas.events.key_press.connect(self._on_key_press)
        
        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas.native)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
    def reset(self):
        """Reset the canvas"""
        # Clear all visuals
        if self.scatter is not None:
            self.scatter.parent = None
            self.scatter = None
        
        if self.histogram is not None:
            self.histogram.parent = None
            self.histogram = None
            
        if self.selection_line is not None:
            self.selection_line.parent = None
            self.selection_line = None
            
        # Reset selection
        self.is_selecting = False
        self.selection_path = []
        self.selected_mask = None
        
        # Reset data
        self.x_data = None
        self.y_data = None
        self.colors = None
        self.sizes = None
        
        # Ensure camera orientation is preserved
        self.view.camera.flip = (False, True, False)
        
    def set_labels(self, x_label, y_label):
        """Set axis labels"""
        # Swap labels to correct the orientation
        self.x_label.text = x_label
        self.y_label.text = y_label
        
    def _on_mouse_press(self, event):
        """Handle mouse press events"""
        if event.button == 1 and 'Shift' in event.modifiers:  # Left click with Shift key
            # Start lasso selection
            self.is_selecting = True
            self.selection_path = []
            
            # Remove any existing selection line
            if self.selection_line is not None:
                self.selection_line.parent = None
                self.selection_line = None
            
            # Get position in data coordinates
            tr = self.view.scene.transform
            pos = tr.imap(event.pos)[:2]  # Only take x, y coordinates
            self.selection_path.append(pos)
            
            # Create line visual for selection path
            self.selection_line = scene.visuals.Line(
                pos=np.array([pos]),
                color='yellow',
                width=2,
                connect='strip',
                parent=self.view.scene
            )
            
            # Prevent event from propagating to other handlers (like panning)
            event.handled = True
        
    def _on_mouse_move(self, event):
        """Handle mouse move events"""
        if self.is_selecting and event.button == 1 and 'Shift' in event.modifiers:
            # Add point to selection path
            tr = self.view.scene.transform
            pos = tr.imap(event.pos)[:2]  # Only take x, y coordinates
            self.selection_path.append(pos)
            
            # Update selection line
            self.selection_line.set_data(pos=np.array(self.selection_path))
            
            # Prevent event from propagating to other handlers
            event.handled = True
            
    def _on_mouse_release(self, event):
        """Handle mouse release events"""
        if self.is_selecting and event.button == 1:
            # End selection
            self.is_selecting = False
            
            # Close the path
            if len(self.selection_path) > 2:
                self.selection_path.append(self.selection_path[0])
                self.selection_line.set_data(pos=np.array(self.selection_path))
                
                # Create a polygon from the selection path
                from matplotlib.path import Path
                path = Path(self.selection_path)
                
                # Check which points are inside the polygon
                if self.scatter is not None and self.x_data is not None and self.y_data is not None:
                    points = np.column_stack([self.x_data, self.y_data])
                    mask = path.contains_points(points)
                    
                    # Highlight selected points
                    self._highlight_selected_points(mask)
                    
                    # Emit signal with selected indices
                    self.selection_changed.emit(np.where(mask)[0])
            
            # Prevent event from propagating to other handlers
            event.handled = True
            
    def _on_key_press(self, event):
        """Handle key press events"""
        if event.key == 'Escape':
            # Cancel selection
            if self.selection_line is not None:
                self.selection_line.parent = None
                self.selection_line = None
            
            # Reset selection
            self.is_selecting = False
            self.selection_path = []
            
            # Clear selection highlight
            if self.selected_mask is not None:
                self._highlight_selected_points(np.zeros_like(self.selected_mask, dtype=bool))
                self.selection_changed.emit(np.array([]))
                
            # Prevent event from propagating to other handlers
            event.handled = True
        
    def _highlight_selected_points(self, mask):
        """Highlight selected points"""
        if self.scatter is None or not hasattr(self, 'colors') or self.colors is None:
            return
            
        # Store selected mask
        self.selected_mask = mask
        
        # Get original colors
        colors = self.colors.copy()
        
        # Highlight selected points
        if np.any(mask):
            # Make a copy of the colors
            highlight_colors = colors.copy()
            
            # Set selected points to yellow
            highlight_colors[mask] = [1.0, 1.0, 0.0, 1.0]  # Bright yellow
            
            # Update scatter plot colors
            self.scatter.set_data(
                pos=np.column_stack([self.x_data, self.y_data]),
                size=self.sizes,
                face_color=highlight_colors
            )
        else:
            # Reset to original colors
            self.scatter.set_data(
                pos=np.column_stack([self.x_data, self.y_data]),
                size=self.sizes,
                face_color=colors
            )
        
    def make_scatter_plot(self, x_data, y_data, colors, sizes, alpha=0.7):
        """Create a scatter plot"""
        self.reset()
        
        # Store data for selection
        self.x_data = x_data
        self.y_data = y_data
        self.colors = colors
        self.sizes = sizes
        
        # Create scatter visual - swap x and y to correct orientation
        pos = np.column_stack([x_data, y_data])
        
        # Set alpha for all colors
        face_color = colors.copy()
        if face_color.shape[1] >= 4:  # If alpha channel exists
            face_color[:, 3] = alpha
        
        # Create scatter plot
        self.scatter = visuals.Markers()
        self.scatter.set_data(
            pos=pos, 
            face_color=face_color, 
            size=sizes,
            edge_width=0
        )
        self.view.add(self.scatter)
        
        # Auto-scale the view
        self.view.camera.set_range()
        
    def make_2d_histogram(self, x_data, y_data, bin_number=100, log_scale=False):
        """Create a 2D histogram"""
        self.reset()
        
        # Store the original data for selection
        self.x_data = x_data
        self.y_data = y_data
        
        # Compute 2D histogram
        H, xedges, yedges = np.histogram2d(x_data, y_data, bins=bin_number)
        
        # Apply log scale if requested
        if log_scale:
            H = np.log1p(H)
        
        # Normalize for colormap
        H_normalized = H / np.max(H) if np.max(H) > 0 else H
        
        # Get colormap
        cmap = get_colormap('viridis')
        
        # Create image with histogram data - use H directly, not transposed
        self.histogram = visuals.Image(
            H_normalized, 
            cmap=cmap,
            clim=(0, 1)
        )
        
        # Position the image correctly
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        
        self.histogram.transform = scene.transforms.STTransform(
            scale=(
                (x_max - x_min) / H.shape[0],
                (y_max - y_min) / H.shape[1],
                1
            ),
            translate=(x_min, y_min, 0)
        )
        
        self.view.add(self.histogram)
        
        # Auto-scale the view
        self.view.camera.set_range()
        
    def make_1d_histogram(self, data, bin_number=100, log_scale=False):
        """Create a 1D histogram"""
        self.reset()
        
        # Store the original data for selection
        self.x_data = data
        self.y_data = np.zeros_like(data)  # Dummy Y data for selection
        
        # Compute histogram
        hist, bin_edges = np.histogram(data, bins=bin_number)
        
        # Apply log scale if requested
        if log_scale:
            hist = np.log1p(hist)
        
        # Create line plot for histogram
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        pos = np.column_stack([bin_centers, hist])
        
        # Create line visual
        line = scene.visuals.Line(
            pos=pos, 
            color='white',
            width=2,
            connect='strip'
        )
        
        # Create bars
        bar_width = (bin_edges[1] - bin_edges[0]) * 0.8
        for i, (x, y) in enumerate(zip(bin_centers, hist)):
            rect = scene.visuals.Rectangle(
                center=(x, y/2),
                width=bar_width,
                height=y,
                color='white'
            )
            self.view.add(rect)
        
        self.view.add(line)
        
        # Auto-scale the view
        self.view.camera.set_range()


def vispy_plotter_widget(napari_viewer=None):
    """Create a Vispy-based widget for plotting data from napari layers.
    
    This widget provides high-performance interactive visualization of data 
    from napari layers, including scatter plots and histograms. It's optimized
    for large datasets with thousands or millions of points.
    
    Parameters
    ----------
    napari_viewer : napari.Viewer, optional
        The napari viewer instance. If not provided, will try to get the current viewer.
        
    Returns
    -------
    VispyPlotterWidget
        The plotter widget instance
    """
    if napari_viewer is None:
        import napari
        try:
            napari_viewer = napari.current_viewer()
        except:
            raise ValueError("No napari viewer found. Please provide a viewer instance.")
    
    return VispyPlotterWidget(napari_viewer)


class VispyPlotterWidget(QWidget):
    """Widget for plotting data from napari layers using Vispy"""
    
    def __init__(self, napari_viewer):
        super().__init__()
        
        self.viewer = napari_viewer
        self.analysed_layer = None
        self.visualized_layer = None
        self.cluster_ids = None
        self.data_x = None
        self.data_y = None
        self.plot_x_axis_name = None
        self.plot_y_axis_name = None
        self.plot_cluster_name = None
        self.old_frame = None
        self.frame = self.viewer.dims.current_step[0] if self.viewer.dims.ndim > 0 else 0
        
        # Set transparent background
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        
        # Create main layout
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        
        # Create canvas widget with Vispy
        self.graphics_widget = VispyCanvas()
        self.graphics_widget.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        
        # Connect selection signal
        self.graphics_widget.selection_changed.connect(self.on_selection_changed)
        
        # Create graph container
        graph_container = QWidget()
        graph_container.setMinimumHeight(300)
        graph_container.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        graph_container.setLayout(QVBoxLayout())
        graph_container.layout().setContentsMargins(0, 0, 0, 0)  # Remove margins
        graph_container.layout().addWidget(self.graphics_widget)
        
        self.layout.addWidget(graph_container, alignment=Qt.AlignTop)
        
        # Create layer selection
        layer_container = QWidget()
        layer_container.setLayout(QVBoxLayout())
        layer_container.layout().setContentsMargins(5, 5, 5, 5)
        layer_container.setStyleSheet("background-color: rgba(40, 40, 40, 100); border-radius: 5px;")
        layer_container_label = QLabel("<b>Layer Selection</b>")
        layer_container_label.setStyleSheet("color: white;")
        layer_container.layout().addWidget(layer_container_label)
        
        self.layer_select = QComboBox()
        self.layer_select.setStyleSheet("background-color: rgba(60, 60, 60, 150); color: white; selection-background-color: rgba(80, 80, 80, 200);")
        self.update_layer_list()
        layer_container.layout().addWidget(self.layer_select)
        
        # Create axes selection
        axes_container = QWidget()
        axes_container.setLayout(QHBoxLayout())
        axes_container.layout().setContentsMargins(5, 5, 5, 5)
        axes_container.setStyleSheet("background-color: rgba(40, 40, 40, 100); border-radius: 5px;")
        axes_label = QLabel("Axes")
        axes_label.setStyleSheet("color: white;")
        axes_container.layout().addWidget(axes_label)
        self.plot_x_axis = QComboBox()
        self.plot_x_axis.setStyleSheet("background-color: rgba(60, 60, 60, 150); color: white; selection-background-color: rgba(80, 80, 80, 200);")
        self.plot_y_axis = QComboBox()
        self.plot_y_axis.setStyleSheet("background-color: rgba(60, 60, 60, 150); color: white; selection-background-color: rgba(80, 80, 80, 200);")
        axes_container.layout().addWidget(self.plot_x_axis)
        axes_container.layout().addWidget(self.plot_y_axis)
        
        # Create selection info container
        self.selection_info = QLabel("Hold Shift + drag to select points with lasso")
        self.selection_info.setStyleSheet("color: white; background-color: rgba(40, 40, 40, 100); border-radius: 3px; padding: 5px;")
        
        # Create advanced options
        advanced_container = QWidget()
        advanced_container.setLayout(QVBoxLayout())
        advanced_container.layout().setContentsMargins(5, 5, 5, 5)
        advanced_container.setStyleSheet("background-color: rgba(40, 40, 40, 100); border-radius: 5px;")
        advanced_label = QLabel("<b>Advanced Options</b>")
        advanced_label.setStyleSheet("color: white;")
        advanced_container.layout().addWidget(advanced_label)
        
        # Plotting type selection
        plot_type_container = QWidget()
        plot_type_container.setLayout(QHBoxLayout())
        plot_type_container.setStyleSheet("background-color: rgba(50, 50, 50, 80);")
        plot_type_label = QLabel("Plotting type")
        plot_type_label.setStyleSheet("color: white;")
        plot_type_container.layout().addWidget(plot_type_label)
        self.plotting_type = QComboBox()
        self.plotting_type.setStyleSheet("background-color: rgba(60, 60, 60, 150); color: white; selection-background-color: rgba(80, 80, 80, 200);")
        self.plotting_type.addItems([PlottingType.SCATTER.name, PlottingType.HISTOGRAM.name])
        plot_type_container.layout().addWidget(self.plotting_type)
        advanced_container.layout().addWidget(plot_type_container)
        
        # Bin number for histograms
        self.bin_number_container = QWidget()
        self.bin_number_container.setLayout(QHBoxLayout())
        self.bin_number_container.setStyleSheet("background-color: rgba(50, 50, 50, 80);")
        bin_number_label = QLabel("Number of bins")
        bin_number_label.setStyleSheet("color: white;")
        self.bin_number_container.layout().addWidget(bin_number_label)
        self.bin_number_spinner = QSpinBox()
        self.bin_number_spinner.setStyleSheet("background-color: rgba(60, 60, 60, 150); color: white;")
        self.bin_number_spinner.setMinimum(1)
        self.bin_number_spinner.setMaximum(1000)
        self.bin_number_spinner.setValue(100)
        self.bin_number_container.layout().addWidget(self.bin_number_spinner)
        self.bin_auto = QCheckBox("Auto")
        self.bin_auto.setStyleSheet("color: white;")
        self.bin_auto.setChecked(True)
        self.bin_number_container.layout().addWidget(self.bin_auto)
        advanced_container.layout().addWidget(self.bin_number_container)
        self.bin_number_container.setVisible(False)
        
        # Log scale option
        self.log_scale_container = QWidget()
        self.log_scale_container.setLayout(QHBoxLayout())
        self.log_scale_container.setStyleSheet("background-color: rgba(50, 50, 50, 80);")
        log_scale_label = QLabel("Log scale")
        log_scale_label.setStyleSheet("color: white;")
        self.log_scale_container.layout().addWidget(log_scale_label)
        self.log_scale = QCheckBox("")
        self.log_scale.setStyleSheet("color: white;")
        self.log_scale.setChecked(False)
        self.log_scale_container.layout().addWidget(self.log_scale)
        advanced_container.layout().addWidget(self.log_scale_container)
        self.log_scale_container.setVisible(False)
        
        # Colormap selection
        self.colormap_container = QWidget()
        self.colormap_container.setLayout(QHBoxLayout())
        self.colormap_container.setStyleSheet("background-color: rgba(50, 50, 50, 80);")
        colormap_label = QLabel("Colormap")
        colormap_label.setStyleSheet("color: white;")
        self.colormap_container.layout().addWidget(colormap_label)
        self.colormap_dropdown = QComboBox()
        self.colormap_dropdown.setStyleSheet("background-color: rgba(60, 60, 60, 150); color: white; selection-background-color: rgba(80, 80, 80, 200);")
        # Use napari's available colormaps
        self.colormap_dropdown.addItems(list(ALL_COLORMAPS.keys()))
        self.colormap_dropdown.setCurrentText("viridis")
        self.colormap_container.layout().addWidget(self.colormap_dropdown)
        advanced_container.layout().addWidget(self.colormap_container)
        self.colormap_container.setVisible(False)
        
        # Create buttons
        run_container = QWidget()
        run_container.setLayout(QHBoxLayout())
        run_container.layout().setContentsMargins(5, 5, 5, 5)
        run_container.setStyleSheet("background-color: rgba(40, 40, 40, 100); border-radius: 5px;")
        self.run_button = QPushButton("Plot")
        self.run_button.setStyleSheet("background-color: rgba(70, 70, 70, 200); color: white; border-radius: 3px; padding: 5px;")
        run_container.layout().addWidget(self.run_button)
        
        update_container = QWidget()
        update_container.setLayout(QHBoxLayout())
        update_container.layout().setContentsMargins(5, 5, 5, 5)
        update_container.setStyleSheet("background-color: rgba(40, 40, 40, 100); border-radius: 5px;")
        self.update_button = QPushButton("Update Axes Options")
        self.update_button.setStyleSheet("background-color: rgba(70, 70, 70, 200); color: white; border-radius: 3px; padding: 5px;")
        update_container.layout().addWidget(self.update_button)
        
        # Add all widgets to layout
        self.layout.addWidget(layer_container, alignment=Qt.AlignTop)
        self.layout.addWidget(axes_container, alignment=Qt.AlignTop)
        self.layout.addWidget(self.selection_info, alignment=Qt.AlignTop)
        self.layout.addWidget(advanced_container, alignment=Qt.AlignTop)
        self.layout.addWidget(update_container, alignment=Qt.AlignTop)
        self.layout.addWidget(run_container, alignment=Qt.AlignTop)
        
        # Connect signals
        self.run_button.clicked.connect(self.run_clicked)
        self.update_button.clicked.connect(self.update_axes_and_clustering_id_lists)
        self.plotting_type.currentIndexChanged.connect(self.plotting_type_changed)
        self.bin_auto.stateChanged.connect(self.bin_auto_changed)
        self.layer_select.currentIndexChanged.connect(self.update_axes_and_clustering_id_lists)
        self.viewer.layers.events.inserted.connect(self.update_layer_list)
        self.viewer.layers.events.removed.connect(self.update_layer_list)
        self.viewer.dims.events.current_step.connect(self.frame_changed)
        self.colormap_dropdown.currentIndexChanged.connect(self.update_colormap)
        
    def on_selection_changed(self, selected_indices):
        """Handle selection change events"""
        if len(selected_indices) > 0:
            # Update selection info
            self.selection_info.setText(f"{len(selected_indices)} points selected (Esc to cancel)")
            
            # Highlight selected points in napari
            if self.visualized_layer is not None and isinstance(self.visualized_layer, (Points, Labels)):
                # Get the data for the selected points
                if isinstance(self.visualized_layer, Points):
                    # For Points layer, get the coordinates
                    selected_data = self.visualized_layer.data[selected_indices]
                    
                    # Create or update a selection layer
                    selection_layer_name = f"{self.visualized_layer.name}_selection"
                    
                    # Check if selection layer already exists
                    if selection_layer_name in self.viewer.layers:
                        # Update existing layer
                        self.viewer.layers[selection_layer_name].data = selected_data
                    else:
                        # Create new layer
                        self.viewer.add_points(
                            selected_data,
                            name=selection_layer_name,
                            size=10,
                            face_color='yellow',
                            edge_color='white',
                            edge_width=1,
                            opacity=0.7
                        )
                
                elif isinstance(self.visualized_layer, Labels):
                    # For Labels layer, create a binary mask of selected labels
                    labels_data = self.visualized_layer.data
                    unique_labels = np.unique(labels_data)
                    
                    # Get the feature data for this layer
                    features = self.get_layer_tabular_data(self.visualized_layer)
                    
                    # Map selected indices to label values
                    if features is not None and len(features) > 0:
                        # Get label values for selected indices
                        selected_labels = features.index.values[selected_indices]
                        
                        # Create a binary mask for selected labels
                        mask = np.zeros_like(labels_data, dtype=bool)
                        for label in selected_labels:
                            if label in unique_labels:
                                mask = mask | (labels_data == label)
                        
                        # Create or update a selection layer
                        selection_layer_name = f"{self.visualized_layer.name}_selection"
                        
                        # Check if selection layer already exists
                        if selection_layer_name in self.viewer.layers:
                            # Update existing layer
                            self.viewer.layers[selection_layer_name].data = mask
                        else:
                            # Create new layer
                            self.viewer.add_labels(
                                mask.astype(np.uint8),
                                name=selection_layer_name,
                                color={1: 'yellow'},
                                opacity=0.5
                            )
        else:
            # Reset selection info
            self.selection_info.setText("Hold Shift + drag to select points with lasso")
            
            # Remove selection layers
            if self.visualized_layer is not None:
                selection_layer_name = f"{self.visualized_layer.name}_selection"
                if selection_layer_name in self.viewer.layers:
                    self.viewer.layers.remove(selection_layer_name)
    
    def update_layer_list(self, event=None):
        """Update the list of available layers"""
        self.layer_select.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, (Labels, Points, Surface)):
                self.layer_select.addItem(layer.name)
    
    def get_selected_layer(self):
        """Get the currently selected layer"""
        layer_name = self.layer_select.currentText()
        if layer_name:
            return self.viewer.layers[layer_name]
        return None
    
    def update_axes_and_clustering_id_lists(self):
        """Update the available axes and clustering IDs based on the selected layer"""
        selected_layer = self.get_selected_layer()
        
        if selected_layer is None:
            return
            
        features = self.get_layer_tabular_data(selected_layer)
        if features is None:
            return
            
        self.plot_x_axis.clear()
        self.plot_y_axis.clear()
        
        self.plot_x_axis.addItems(list(features.keys()))
        self.plot_y_axis.addItems(list(features.keys()))
    
    def get_layer_tabular_data(self, layer):
        """Get tabular data from a layer"""
        if hasattr(layer, 'features') and layer.features is not None:
            return layer.features
        return None
    
    def plotting_type_changed(self):
        """Handle changes to the plotting type"""
        if self.plotting_type.currentText() == PlottingType.HISTOGRAM.name:
            self.bin_number_container.setVisible(True)
            self.log_scale_container.setVisible(True)
            self.colormap_container.setVisible(True)
        else:
            self.bin_number_container.setVisible(False)
            self.log_scale_container.setVisible(False)
            self.colormap_container.setVisible(False)
    
    def bin_auto_changed(self):
        """Handle changes to the bin auto checkbox"""
        self.bin_number_spinner.setEnabled(not self.bin_auto.isChecked())
    
    def frame_changed(self, event):
        """Handle changes to the current frame"""
        if self.viewer.dims.ndim <= 3:
            return
            
        frame = event.value[0]
        if (not self.old_frame) or (self.old_frame != frame):
            self.frame = frame
            if self.analysed_layer is not None:
                self.run()
                
        self.old_frame = frame
    
    def run_clicked(self):
        """Handle the run button click"""
        selected_layer = self.get_selected_layer()
        
        if selected_layer is None:
            print("Please select a layer!")
            return
            
        features = self.get_layer_tabular_data(selected_layer)
        if features is None:
            print("No features/properties found in the selected layer!")
            return
            
        if self.plot_x_axis.currentText() == "" or self.plot_y_axis.currentText() == "":
            print("Please select axes for plotting!")
            return
            
        self.run()
    
    def run(self):
        """Run the plotting with the current settings"""
        selected_layer = self.get_selected_layer()
        if selected_layer is None:
            return
        
        # Get features from layer
        features = self.get_layer_tabular_data(selected_layer)
        if features is None or len(features) == 0:
            return
        
        # Get axis names
        plot_x_axis_name = self.plot_x_axis.currentText()
        plot_y_axis_name = self.plot_y_axis.currentText()
        
        # Check if axes exist in features
        if plot_x_axis_name not in features.columns or plot_y_axis_name not in features.columns:
            return
        
        # Store current settings
        self.data_x = features[plot_x_axis_name].to_numpy()
        self.data_y = features[plot_y_axis_name].to_numpy()
        self.plot_x_axis_name = plot_x_axis_name
        self.plot_y_axis_name = plot_y_axis_name
        self.analysed_layer = selected_layer
        
        # Get plotting type
        plotting_type = PlottingType[self.plotting_type.currentText()]
        
        # Get bin number for histograms
        bin_number = self.bin_number_spinner.value()
        if self.bin_auto.isChecked():
            bin_number = self.estimate_bin_number()
        
        # Get log scale option
        log_scale = self.log_scale.isChecked()
        
        # Determine if we're using clustering
        self.cluster_ids = None
        colors = np.ones((len(self.data_x), 4), dtype=np.float32) * 0.7  # Default gray
        
        # Auto-detect clustering columns
        cluster_columns = [col for col in features.columns if 'CLUSTER' in col.upper() or 'LABEL' in col.upper()]
        if cluster_columns:
            # Use the first clustering column found
            cluster_column = cluster_columns[0]
            self.cluster_ids = features[cluster_column].fillna(-1).to_numpy()
            colors = self.get_cluster_colors(self.cluster_ids)
        
        # Create sizes array (all same size for now)
        sizes = np.ones(len(self.data_x)) * 10
        
        # Create the plot based on the plotting type
        if plotting_type == PlottingType.SCATTER:
            self.graphics_widget.make_scatter_plot(
                self.data_x, self.data_y, colors, sizes
            )
            self.graphics_widget.set_labels(plot_x_axis_name, plot_y_axis_name)
            
            # Show colormap option for scatter plots
            self.colormap_container.setVisible(True)
            
        elif plotting_type == PlottingType.HISTOGRAM:
            if plot_x_axis_name == plot_y_axis_name:
                # 1D histogram
                self.graphics_widget.make_1d_histogram(
                    self.data_x, bin_number=bin_number, log_scale=log_scale
                )
                self.graphics_widget.set_labels(plot_x_axis_name, "Count")
            else:
                # 2D histogram
                self.graphics_widget.make_2d_histogram(
                    self.data_x, self.data_y, bin_number=bin_number, log_scale=log_scale
                )
                self.graphics_widget.set_labels(plot_x_axis_name, plot_y_axis_name)
            
            # Hide colormap option for histograms
            self.colormap_container.setVisible(False)
        
        # Update the visualized layer
        self.visualized_layer = selected_layer
    
    def estimate_bin_number(self):
        """Estimate a reasonable number of bins for histograms"""
        if self.plot_x_axis_name == self.plot_y_axis_name:
            # For 1D histogram
            return min(int(np.sqrt(len(self.data_x))), 100)
        else:
            # For 2D histogram
            return min(int(np.sqrt(len(self.data_x))), 50)
    
    def get_cluster_colors(self, cluster_ids):
        """Get colors for clusters"""
        unique_ids = np.unique(cluster_ids)
        n_clusters = len(unique_ids)
        
        # Get a colormap from vispy
        cmap_name = self.colormap_dropdown.currentText()
        cmap = get_colormap(cmap_name)
        
        # Map cluster IDs to colors
        colors = np.zeros((len(cluster_ids), 4))
        for i, cluster_id in enumerate(unique_ids):
            mask = cluster_ids == cluster_id
            if cluster_id == -1:  # Noise points
                colors[mask] = [0.5, 0.5, 0.5, 0.5]  # Gray, semi-transparent
            else:
                # Normalize color index to 0-1 range
                color_idx = i / max(1, n_clusters - 1)
                colors[mask] = cmap.map(np.array([color_idx]))[0]
        
        return colors
    
    def update_colormap(self):
        """Update the colormap when selection changes"""
        self.graphics_widget.colormap_name = self.colormap_dropdown.currentText()
        if self.data_x is not None and self.data_y is not None:
            self.run()
    
    def update_axes_and_clustering_id_lists(self):
        """Update the available axes and clustering IDs based on the selected layer"""
        selected_layer = self.get_selected_layer()
        
        if selected_layer is None:
            return
            
        features = self.get_layer_tabular_data(selected_layer)
        if features is None:
            return
            
        self.plot_x_axis.clear()
        self.plot_y_axis.clear()
        
        self.plot_x_axis.addItems(list(features.keys()))
        self.plot_y_axis.addItems(list(features.keys()))
    
    def get_layer_tabular_data(self, layer):
        """Get tabular data from a layer"""
        if hasattr(layer, 'features') and layer.features is not None:
            return layer.features
        return None
    
    def plotting_type_changed(self):
        """Handle changes to the plotting type"""
        if self.plotting_type.currentText() == PlottingType.HISTOGRAM.name:
            self.bin_number_container.setVisible(True)
            self.log_scale_container.setVisible(True)
            self.colormap_container.setVisible(True)
        else:
            self.bin_number_container.setVisible(False)
            self.log_scale_container.setVisible(False)
            self.colormap_container.setVisible(False)
    
    def bin_auto_changed(self):
        """Handle changes to the bin auto checkbox"""
        self.bin_number_spinner.setEnabled(not self.bin_auto.isChecked())
    
    def frame_changed(self, event):
        """Handle changes to the current frame"""
        if self.viewer.dims.ndim <= 3:
            return
            
        frame = event.value[0]
        if (not self.old_frame) or (self.old_frame != frame):
            self.frame = frame
            if self.analysed_layer is not None:
                self.run()
                
        self.old_frame = frame
    
    def run_clicked(self):
        """Handle the run button click"""
        selected_layer = self.get_selected_layer()
        
        if selected_layer is None:
            print("Please select a layer!")
            return
            
        features = self.get_layer_tabular_data(selected_layer)
        if features is None:
            print("No features/properties found in the selected layer!")
            return
            
        if self.plot_x_axis.currentText() == "" or self.plot_y_axis.currentText() == "":
            print("Please select axes for plotting!")
            return
            
        self.run()
    
    def run(self):
        """Run the plotting with the current settings"""
        selected_layer = self.get_selected_layer()
        if selected_layer is None:
            return
        
        # Get features from layer
        features = self.get_layer_tabular_data(selected_layer)
        if features is None or len(features) == 0:
            return
        
        # Get axis names
        plot_x_axis_name = self.plot_x_axis.currentText()
        plot_y_axis_name = self.plot_y_axis.currentText()
        
        # Check if axes exist in features
        if plot_x_axis_name not in features.columns or plot_y_axis_name not in features.columns:
            return
        
        # Store current settings
        self.data_x = features[plot_x_axis_name].to_numpy()
        self.data_y = features[plot_y_axis_name].to_numpy()
        self.plot_x_axis_name = plot_x_axis_name
        self.plot_y_axis_name = plot_y_axis_name
        self.analysed_layer = selected_layer
        
        # Get plotting type
        plotting_type = PlottingType[self.plotting_type.currentText()]
        
        # Get bin number for histograms
        bin_number = self.bin_number_spinner.value()
        if self.bin_auto.isChecked():
            bin_number = self.estimate_bin_number()
        
        # Get log scale option
        log_scale = self.log_scale.isChecked()
        
        # Determine if we're using clustering
        self.cluster_ids = None
        colors = np.ones((len(self.data_x), 4), dtype=np.float32) * 0.7  # Default gray
        
        # Auto-detect clustering columns
        cluster_columns = [col for col in features.columns if 'CLUSTER' in col.upper() or 'LABEL' in col.upper()]
        if cluster_columns:
            # Use the first clustering column found
            cluster_column = cluster_columns[0]
            self.cluster_ids = features[cluster_column].fillna(-1).to_numpy()
            colors = self.get_cluster_colors(self.cluster_ids)
        
        # Create sizes array (all same size for now)
        sizes = np.ones(len(self.data_x)) * 10
        
        # Create the plot based on the plotting type
        if plotting_type == PlottingType.SCATTER:
            self.graphics_widget.make_scatter_plot(
                self.data_x, self.data_y, colors, sizes
            )
            self.graphics_widget.set_labels(plot_x_axis_name, plot_y_axis_name)
            
            # Show colormap option for scatter plots
            self.colormap_container.setVisible(True)
            
        elif plotting_type == PlottingType.HISTOGRAM:
            if plot_x_axis_name == plot_y_axis_name:
                # 1D histogram
                self.graphics_widget.make_1d_histogram(
                    self.data_x, bin_number=bin_number, log_scale=log_scale
                )
                self.graphics_widget.set_labels(plot_x_axis_name, "Count")
            else:
                # 2D histogram
                self.graphics_widget.make_2d_histogram(
                    self.data_x, self.data_y, bin_number=bin_number, log_scale=log_scale
                )
                self.graphics_widget.set_labels(plot_x_axis_name, plot_y_axis_name)
            
            # Hide colormap option for histograms
            self.colormap_container.setVisible(False)
        
        # Update the visualized layer
        self.visualized_layer = selected_layer 