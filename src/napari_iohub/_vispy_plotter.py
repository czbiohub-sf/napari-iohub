"""
Vispy-based plotter for napari-iohub.

This module provides a high-performance plotter using Vispy for visualizing
large datasets within napari.
"""

import numpy as np
import pandas as pd
from enum import Enum, auto
from matplotlib.path import Path
from vispy import scene, visuals
from vispy.scene import visuals
from vispy.color import get_colormap

from napari.layers import Image, Labels, Layer, Points, Surface
from napari.utils.colormaps import ALL_COLORMAPS, color_dict_to_colormap, label_colormap

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
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
        
        # Store parent for accessing class colors
        self.parent_widget = parent
        
        # Default class colors if parent doesn't provide them
        self.class_colors = [
            [1.0, 0.0, 0.0, 1.0],  # Red
            [0.0, 1.0, 0.0, 1.0],  # Green
            [0.0, 0.0, 1.0, 1.0],  # Blue
            [1.0, 1.0, 0.0, 1.0],  # Yellow
            [1.0, 0.0, 1.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0, 1.0],  # Cyan
        ]
        self.current_class = 0
        
        # Create a scene canvas with a transparent background
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='transparent')
        self.canvas.create_native()
        self.canvas.native.setParent(self)
        
        # Create a view for the canvas
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.view.camera.aspect = 1
        
        # Set camera orientation for standard image coordinates
        # (0,0) at top left, positive x to right, positive y downward
        # In Vispy, we need to flip the y-axis to match this convention
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
            
            # Get class colors and current class from parent if available
            if hasattr(self.parent_widget, 'class_colors') and hasattr(self.parent_widget, 'current_class'):
                class_colors = self.parent_widget.class_colors
                current_class = self.parent_widget.current_class
            else:
                class_colors = self.class_colors
                current_class = self.current_class
                
            # Get the next class color that would be assigned
            next_class_idx = (current_class + 1) % len(class_colors)
            next_class_color = class_colors[next_class_idx]
            
            # Set selected points to the next class color (instead of yellow)
            for i in np.where(mask)[0]:
                highlight_colors[i, :3] = next_class_color[:3]
                # Keep original opacity or set to 1.0 for visibility
                highlight_colors[i, 3] = max(colors[i, 3], 0.8)
            
            # Update scatter plot
            self.scatter.set_data(
                pos=np.column_stack([self.x_data, self.y_data]),
                face_color=highlight_colors,
                size=self.sizes,
                edge_width=0
            )
        else:
            # Reset to original colors
            self.scatter.set_data(
                pos=np.column_stack([self.x_data, self.y_data]),
                face_color=colors,
                size=self.sizes,
                edge_width=0
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
        self.colors = None
        self.plot_x_axis_name = None
        self.plot_y_axis_name = None
        self.old_frame = None
        self.frame = self.viewer.dims.current_step[0] if self.viewer.dims.ndim > 0 else 0
        
        # Classification system
        self.class_colors = [
            [1.0, 0.0, 0.0, 1.0],  # Red
            [0.0, 1.0, 0.0, 1.0],  # Green
            [0.0, 0.0, 1.0, 1.0],  # Blue
            [1.0, 1.0, 0.0, 1.0],  # Yellow
            [1.0, 0.0, 1.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0, 1.0],  # Orange
            [0.5, 0.0, 1.0, 1.0],  # Purple
            [0.0, 0.5, 0.0, 1.0],  # Dark Green
            [0.5, 0.5, 0.5, 1.0],  # Gray
        ]
        self.current_class = 0  # Index to cycle through colors
        self.class_assignments = None  # Will store class assignments for each point
        self.class_layers = {}  # Will store class layers in napari
        
        # Set transparent background
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0);")
        
        # Create main layout
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignTop)
        self.layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        
        # Create canvas widget with Vispy
        self.graphics_widget = VispyCanvas(parent=self)  # Pass self as parent
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
        
        # Add instruction label right below the plot
        instruction_label = QLabel("Hold Shift + drag to select points. Each selection creates a new class.\nEmpty selection clears all classes. Press Escape to cancel selection.")
        instruction_label.setStyleSheet("color: white; background-color: rgba(40, 40, 40, 100); border-radius: 3px; padding: 5px;")
        instruction_label.setAlignment(Qt.AlignCenter)
        graph_container.layout().addWidget(instruction_label)
        
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
        self.selection_info = QLabel("Hold Shift + drag to select points with lasso (each selection creates a new class)")
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
        self.viewer.dims.events.ndisplay.connect(self._on_ndisplay_change)
        
    def on_selection_changed(self, selected_indices):
        """Handle selection change events"""
        if len(selected_indices) > 0:
            # Update selection info
            self.selection_info.setText(f"{len(selected_indices)} points selected (Esc to cancel, empty selection to clear all classes)")
            
            # Automatically assign a class to the selected points
            if self.class_assignments is None:
                self.class_assignments = np.zeros(len(self.data_x), dtype=int)
            
            # Increment class ID and cycle through colors
            self.current_class = (self.current_class + 1) % len(self.class_colors)
            class_id = self.current_class + 1  # Add 1 to avoid 0 (unassigned)
            
            # Assign selected points to the current class
            self.class_assignments[selected_indices] = class_id
            
            # Print debug info
            print(f"Selected {len(selected_indices)} points, assigned to class {class_id}")
            print(f"Layer type: {type(self.analysed_layer).__name__}")
            
            # If this is a Labels layer, provide additional feedback
            if isinstance(self.analysed_layer, Labels):
                features = self.get_layer_tabular_data(self.analysed_layer)
                if features is not None and 'label' in features.columns:
                    # Get the label values for the selected points
                    selected_labels = features.iloc[selected_indices]['label'].values
                    print(f"Selected labels: {selected_labels[:10]}{'...' if len(selected_labels) > 10 else ''}")
                    self.selection_info.setText(f"{len(selected_indices)} points selected ({len(np.unique(selected_labels))} unique labels)")
            
            # Update visualization in the plot
            self._update_class_visualization()
            
            # Update class layer in napari
            self._update_class_layers()
            
        else:
            # Empty selection - clear all classes
            if self.class_assignments is not None and np.any(self.class_assignments > 0):
                print("Empty selection detected - clearing all classes")
                self.clear_classes()
                self.selection_info.setText("All classes cleared. Hold Shift + drag to select points with lasso.")
            else:
                # No classes to clear, just reset selection info
                self.selection_info.setText("Hold Shift + drag to select points with lasso (empty selection clears all classes)")
    
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
            
        # Debug: Compare coordinates
        self._debug_compare_coordinates(selected_layer)
            
        # Clear existing items
        self.plot_x_axis.clear()
        self.plot_y_axis.clear()
        
        # Get features from the layer
        features = self.get_layer_tabular_data(selected_layer)
        
        if features is not None and len(features.columns) > 0:
            # Add feature columns to the dropdown menus
            self.plot_x_axis.addItems(list(features.columns))
            self.plot_y_axis.addItems(list(features.columns))
            
            # Select default axes if available
            default_axes = ['UMAP_1', 'UMAP_2', 'PCA_1', 'PCA_2', 'x', 'y', 'z']
            for axis in default_axes:
                if axis in features.columns:
                    idx = self.plot_x_axis.findText(axis)
                    if idx >= 0:
                        self.plot_x_axis.setCurrentIndex(idx)
                        break
                        
            for axis in default_axes[::-1]:  # Reverse order for y-axis
                if axis in features.columns:
                    idx = self.plot_y_axis.findText(axis)
                    if idx >= 0:
                        self.plot_y_axis.setCurrentIndex(idx)
                        break
        else:
            # No features available, add a message
            self.plot_x_axis.addItem("No features available")
            self.plot_y_axis.addItem("No features available")
            
            # Disable the run button
            if hasattr(self, 'run_button'):
                self.run_button.setEnabled(False)
    
    def get_layer_tabular_data(self, layer):
        """Get tabular data from a layer"""
        if hasattr(layer, 'features') and layer.features is not None:
            if len(layer.features) > 0:
                return layer.features
            else:
                print(f"Warning: Layer '{layer.name}' has empty features table")
        else:
            print(f"Warning: Layer '{layer.name}' does not have features")
        return None
    
    def _debug_compare_coordinates(self, layer):
        """Debug function to compare coordinate values between layers"""
        if layer is None:
            return
            
        features = self.get_layer_tabular_data(layer)
        if features is None or len(features) == 0:
            return
            
        # Check if this is a Labels or Points layer
        layer_type = type(layer).__name__
        
        # Check for coordinate columns
        coord_columns = {
            'x': None, 'y': None, 'z': None,
            'centroid_x': None, 'centroid_y': None, 'centroid_z': None
        }
        
        for col in coord_columns.keys():
            if col in features.columns:
                # Store min, max, mean values
                values = features[col].to_numpy()
                coord_columns[col] = {
                    'min': np.min(values),
                    'max': np.max(values),
                    'mean': np.mean(values)
                }
        
        # Print debug info
        print(f"\nCoordinate debug info for {layer_type} layer '{layer.name}':")
        for col, stats in coord_columns.items():
            if stats is not None:
                print(f"  {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
        
        # Compare centroid_x/y with x/y if both exist
        if coord_columns['x'] is not None and coord_columns['centroid_x'] is not None:
            x_diff = abs(coord_columns['x']['mean'] - coord_columns['centroid_x']['mean'])
            print(f"  x vs centroid_x mean difference: {x_diff:.2f}")
            
        if coord_columns['y'] is not None and coord_columns['centroid_y'] is not None:
            y_diff = abs(coord_columns['y']['mean'] - coord_columns['centroid_y']['mean'])
            print(f"  y vs centroid_y mean difference: {y_diff:.2f}")
    
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
        """Handle frame change events"""
        if self.viewer.dims.ndim > 0:
            new_frame = self.viewer.dims.current_step[0]
            if new_frame != self.frame:
                self.frame = new_frame
                # Update the plot if we have data and the frame changed
                if self.analysed_layer is not None and self.data_x is not None and self.data_y is not None:
                    self._update_point_highlighting()
    
    def _update_point_highlighting(self):
        """Update point highlighting based on current slice"""
        if self.analysed_layer is None or not hasattr(self.graphics_widget, 'scatter') or self.graphics_widget.scatter is None:
            return
            
        # Get the current frame
        current_frame = self.frame
        
        # Get the layer data
        features = self.get_layer_tabular_data(self.analysed_layer)
        if features is None or len(features) == 0:
            return
            
        # Check if we have a z or time dimension to use for highlighting
        z_columns = [col for col in features.columns if col.lower() in ['z', 'depth', 'slice', 'frame', 'time', 't']]
        if not z_columns:
            return
            
        # Use the first matching column
        z_column = z_columns[0]
        if z_column not in features.columns:
            return
            
        # Get z values
        z_values = features[z_column].to_numpy()
        
        # Calculate distance from current frame
        distances = np.abs(z_values - current_frame)
        
        # Normalize distances to 0-1 range for opacity
        max_dist = np.max(distances) if np.max(distances) > 0 else 1
        normalized_distances = distances / max_dist
        
        # Create opacity values (1.0 for current slice, decreasing with distance)
        opacities = np.clip(1.0 - normalized_distances, 0.1, 1.0)
        
        # Create size values (larger for current slice, smaller with distance)
        base_size = 10  # Base point size
        sizes = base_size * (0.5 + 0.5 * opacities)  # Scale between 50% and 100% of base size
        
        # Update colors with new opacities
        colors = self.colors.copy()
        if colors.shape[1] >= 4:  # If alpha channel exists
            for i in range(len(colors)):
                colors[i, 3] = opacities[i]
            
        # Update scatter plot
        self.graphics_widget.scatter.set_data(
            pos=np.column_stack([self.data_x, self.data_y]),
            face_color=colors,
            size=sizes,
            edge_width=0
        )
            
        # If we have selected points, update their highlighting too
        if hasattr(self.graphics_widget, 'selected_mask') and self.graphics_widget.selected_mask is not None:
            mask = self.graphics_widget.selected_mask
            if np.any(mask):
                highlight_colors = colors.copy()
                # Apply yellow color to selected points while preserving opacity
                for i in np.where(mask)[0]:
                    highlight_colors[i, 0:3] = [1.0, 1.0, 0.0]  # Yellow
                
                self.graphics_widget.scatter.set_data(
                    pos=np.column_stack([self.data_x, self.data_y]),
                    face_color=highlight_colors,
                    size=sizes,
                    edge_width=0
                )
    
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
            
        if self.plot_x_axis.currentText() == "No features available" or self.plot_y_axis.currentText() == "No features available":
            print("No features available for plotting!")
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
        
        # Apply coordinate transformation based on layer type
        # For Labels layer, we need to ensure proper orientation for spatial coordinates
        if isinstance(selected_layer, Labels):
            # Check if we're plotting spatial coordinates that need adjustment
            spatial_coords_y = ['centroid_y', 'y']
            if plot_y_axis_name in spatial_coords_y:
                print(f"Applying y-axis flip for Labels layer spatial coordinate: {plot_y_axis_name}")
                # Negate y to make it increase downward (standard image coordinates)
                self.data_y = -self.data_y
        
        self.plot_x_axis_name = plot_x_axis_name
        self.plot_y_axis_name = plot_y_axis_name
        self.analysed_layer = selected_layer
        
        # Initialize class assignments if needed
        if self.class_assignments is None or len(self.class_assignments) != len(self.data_x):
            self.class_assignments = np.zeros(len(self.data_x), dtype=int)
        
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
        
        # Store colors for later use
        self.colors = colors
        
        # Create the plot based on the plotting type
        if plotting_type == PlottingType.SCATTER:
            self.graphics_widget.make_scatter_plot(
                self.data_x, self.data_y, colors, sizes
            )
            self.graphics_widget.set_labels(plot_x_axis_name, plot_y_axis_name)
            
            # Show colormap option for scatter plots
            self.colormap_container.setVisible(True)
            
            # Update point highlighting based on current slice
            self._update_point_highlighting()
            
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
        """Update the colormap when the dropdown changes"""
        if self.cluster_ids is not None and hasattr(self.graphics_widget, 'scatter') and self.graphics_widget.scatter is not None:
            colors = self.get_cluster_colors(self.cluster_ids)
            self.colors = colors
            self.graphics_widget.scatter.set_data(
                pos=np.column_stack([self.data_x, self.data_y]),
                face_color=colors,
                size=np.ones(len(self.data_x)) * 10,
                edge_width=0
            )
            # Update point highlighting
            self._update_point_highlighting() 

    def clear_classes(self):
        """Handle clearing all class assignments"""
        if self.analysed_layer is None:
            return
            
        # Reset class assignments
        if self.class_assignments is not None:
            self.class_assignments.fill(0)
        
        self.current_class = 0
        
        # Update visualization in the plot
        self._update_class_visualization()
        
        # Remove class layer from napari
        if self.viewer is not None and hasattr(self.viewer, 'layers'):
            class_layer_name = f"{self.analysed_layer.name}_classes"
            if class_layer_name in self.viewer.layers:
                self.viewer.layers.remove(class_layer_name)
    
    def _update_class_visualization(self):
        """Update the visualization of classes in the plot"""
        if self.analysed_layer is None or not hasattr(self.graphics_widget, 'scatter') or self.graphics_widget.scatter is None:
            return
            
        if self.class_assignments is None or len(self.class_assignments) == 0 or np.all(self.class_assignments == 0):
            # If no class assignments, use original colors
            if hasattr(self, 'colors') and self.colors is not None:
                colors = self.colors.copy()
            else:
                # Default gray if no colors
                colors = np.ones((len(self.data_x), 4), dtype=np.float32) * 0.7
        else:
            # Create a color array based on class assignments
            colors = np.ones((len(self.data_x), 4), dtype=np.float32) * 0.7  # Default gray
            
            # Apply class colors
            for i in range(len(self.data_x)):
                class_id = self.class_assignments[i]
                if class_id > 0:  # If assigned to a class
                    color_idx = (class_id - 1) % len(self.class_colors)  # Convert to 0-based index and handle overflow
                    colors[i, :] = self.class_colors[color_idx]
        
        # Store colors for later use
        self.colors = colors
        
        # Update scatter plot
        self.graphics_widget.scatter.set_data(
            pos=np.column_stack([self.data_x, self.data_y]),
            face_color=colors,
            size=np.ones(len(self.data_x)) * 10,
            edge_width=0
        )
        
        # Update point highlighting based on current slice
        self._update_point_highlighting()
    
    def _update_class_layers(self):
        """Update the class layers in napari"""
        if self.analysed_layer is None or self.class_assignments is None:
            return
            
        # Get unique class IDs (excluding 0)
        unique_classes = np.unique(self.class_assignments)
        unique_classes = unique_classes[unique_classes > 0]
        
        if len(unique_classes) == 0:
            # Remove any existing class layers
            if isinstance(self.analysed_layer, Points):
                class_layer_name = f"{self.analysed_layer.name}_classes"
                if class_layer_name in self.viewer.layers:
                    self.viewer.layers.remove(class_layer_name)
            elif isinstance(self.analysed_layer, Labels):
                class_layer_name = f"{self.analysed_layer.name}_classes"
                if class_layer_name in self.viewer.layers:
                    self.viewer.layers.remove(class_layer_name)
            return
            
        # Create a single layer for all classes
        if isinstance(self.analysed_layer, Points):
            # For Points layer, create a single points layer with all classified points
            # Get all points that have a class assignment
            classified_mask = self.class_assignments > 0
            if not np.any(classified_mask):
                return
                
            # Get the coordinates of classified points
            classified_points = self.analysed_layer.data[classified_mask]
            
            # Create a color array for the classified points
            point_colors = []
            for i in np.where(classified_mask)[0]:
                class_id = self.class_assignments[i]
                color_idx = (class_id - 1) % len(self.class_colors)
                # Use the exact same RGB values as in the Vispy plotter
                point_colors.append(self.class_colors[color_idx][:3])
            
            # Create or update the class layer
            class_layer_name = f"{self.analysed_layer.name}_classes"
            
            if class_layer_name in self.viewer.layers:
                # Update existing layer
                self.viewer.layers[class_layer_name].data = classified_points
                self.viewer.layers[class_layer_name].face_color = point_colors
                # Ensure it's visible in 3D
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
        
        elif isinstance(self.analysed_layer, Labels):
            # For Labels layer, create a single labels layer with all classified labels
            labels_data = self.analysed_layer.data
            
            # Get the feature data for this layer
            features = self.get_layer_tabular_data(self.analysed_layer)
            
            if features is not None and len(features) > 0:
                # Create a mask for all classified labels
                class_mask = np.zeros_like(labels_data, dtype=np.uint8)
                
                # Create a color dictionary for the labels
                custom_colormap = {}
                
                # Process each class
                for class_id in unique_classes:
                    # Get mask for this class
                    mask = self.class_assignments == class_id
                    
                    # Get label values for this class
                    if 'label' in features.columns:
                        # If there's a 'label' column, use that for the label IDs
                        # First get the indices of the selected points
                        selected_indices = np.where(mask)[0]
                        # Then get the corresponding labels from the features dataframe
                        class_labels = features.iloc[selected_indices]['label'].values
                        print(f"Using 'label' column for class {class_id}: {class_labels}")
                    elif isinstance(features.index, pd.MultiIndex):
                        # Handle MultiIndex case (common in some napari layers)
                        class_labels = [idx[0] for idx in features.index[mask]]
                    else:
                        # Standard index case
                        class_labels = features.index.values[mask]
                    
                    # Print debug info
                    print(f"Class {class_id}: Found {len(class_labels)} labels to highlight")
                    
                    # Get color for this class - use the exact same RGB values as in the Vispy plotter
                    color_idx = (class_id - 1) % len(self.class_colors)
                    class_color = self.class_colors[color_idx]
                    # Convert to 0-255 RGB for napari labels layer
                    rgb_color = [int(c * 255) for c in class_color[:3]]
                    
                    # Add to colormap dictionary
                    custom_colormap[class_id] = rgb_color
                    
                    # Add to class mask - ensure we're using the correct label values
                    unique_labels = np.unique(labels_data)
                    print(f"Available labels in layer: {unique_labels[:10]}{'...' if len(unique_labels) > 10 else ''}")
                    
                    for label in class_labels:
                        # Ensure label is an integer
                        try:
                            label_int = int(label)
                            # Create a mask for this label
                            label_mask = labels_data == label_int
                            if np.any(label_mask):
                                class_mask[label_mask] = class_id
                                print(f"  - Label {label_int}: Added to class mask (found in labels data)")
                            else:
                                print(f"  - Label {label_int}: Not found in labels data")
                        except (ValueError, TypeError):
                            print(f"  - Label {label}: Could not convert to integer")
                
                # Create or update the class layer
                class_layer_name = f"{self.analysed_layer.name}_classes"
                
                # Check if the class mask has any non-zero values
                if np.any(class_mask > 0):
                    print(f"Class mask has {np.sum(class_mask > 0)} non-zero values")
                    print(f"Using colors: {custom_colormap}")
                    
                    # Create a proper label colormap for napari
                    # First create a base label colormap
                    from napari.utils.colormaps import label_colormap
                    
                    # Get the maximum class ID to determine colormap size
                    max_class_id = max(custom_colormap.keys())
                    
                    # Create a label colormap with a fixed seed for consistency
                    colormap = label_colormap(num_colors=max_class_id+1, seed=0.5)
                    
                    # Now manually set the colors for our class IDs to match the ones used in the plot
                    for class_id, rgb_color in custom_colormap.items():
                        # Convert RGB to normalized values (0-1)
                        normalized_color = [c/255 for c in rgb_color]
                        # Set the color in the colormap
                        # We need to convert the class_id to a 0-1 range for the colormap
                        colormap.colors[int(class_id)] = normalized_color + [1.0]  # Add alpha=1.0
                    
                    if class_layer_name in self.viewer.layers:
                        # Update existing layer
                        self.viewer.layers[class_layer_name].data = class_mask
                        # Update colormap
                        self.viewer.layers[class_layer_name].colormap = colormap
                        # Ensure it's visible in 3D
                        self.viewer.layers[class_layer_name].n_dimensional = True
                        self.viewer.layers[class_layer_name].depiction = 'volume'
                    else:
                        # Create new layer with the colormap
                        self.viewer.add_labels(
                            class_mask,
                            name=class_layer_name,
                            colormap=colormap,
                            opacity=0.5,
                            depiction='volume'
                        )
                else:
                    print("Warning: Class mask is empty, no labels were found for the selected points")

    def _on_ndisplay_change(self, event=None):
        """Handle changes in the viewer's display dimensions (2D/3D)"""
        # When switching to 3D view, ensure class layers are visible
        if self.viewer.dims.ndisplay == 3:
            # Check for any class layers and ensure they're visible in 3D
            for layer in self.viewer.layers:
                if isinstance(layer, Points) and layer.name.endswith('_classes'):
                    layer.n_dimensional = True
                elif isinstance(layer, Labels) and layer.name.endswith('_classes'):
                    layer.depiction = 'volume' 