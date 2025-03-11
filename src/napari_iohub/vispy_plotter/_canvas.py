"""
Canvas for Vispy visualization with lasso selection capability.

This module provides a Vispy canvas with interactive selection capabilities
for high-performance visualization of large datasets.
"""

from enum import Enum, auto
import numpy as np
from matplotlib.path import Path
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import QWidget, QVBoxLayout
from vispy import scene, keys
from vispy.scene.visuals import Markers, Line, Text, Mesh
import vispy.visuals as visuals

# Import local utilities
from ._utils import estimate_bin_number


class PlottingType(Enum):
    """Enum for different plotting types."""
    SCATTER = auto()
    HISTOGRAM = auto()


class VispyCanvas(QWidget):
    """Canvas for Vispy visualization with lasso selection capability.
    
    This class provides a Vispy canvas with interactive selection capabilities
    for high-performance visualization of large datasets.
    
    Attributes:
        selection_changed: Signal emitted when points are selected.
        point_clicked: Signal emitted when a point is clicked.
    """
    
    # Signal emitted when points are selected
    selection_changed = Signal(object)
    
    # Signal emitted when a point is clicked
    point_clicked = Signal(int)
    
    def __init__(self, parent=None):
        """Initialize the canvas.
        
        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        
        # Create a layout for the canvas
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        # Create a scene canvas with a dark background
        self.canvas = scene.SceneCanvas(keys='interactive', bgcolor='black')
        self.canvas.create_native()
        self.canvas.native.setParent(self)
        layout.addWidget(self.canvas.native)
        
        # Create a view for the canvas
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'panzoom'
        self.view.camera.aspect = 1
        
        # Set camera orientation for standard image coordinates
        # (0,0) at top left, positive x to right, positive y downward
        # In Vispy, we need to flip the y-axis to match this convention
        self.view.camera.flip = (False, True, False)
        
        # Initialize grid lines (will be updated in _add_axis_with_ticks)
        self.grid = None
        
        # Set up axis labels
        self.x_label = scene.visuals.Text("X", pos=(0.5, -0.05), color='white', 
                                         anchor_x='center', anchor_y='top',
                                         font_size=10, parent=self.view.scene)
        self.y_label = scene.visuals.Text("Y", pos=(-0.05, 0.5), color='white',
                                         anchor_x='right', anchor_y='center',
                                         font_size=10, rotation=-90, parent=self.view.scene)
        
        # Initialize variables for selection
        self.is_selecting = False
        self.selection_path = []
        self.selection_line = None
        self.scatter = None
        self.selected_mask = None
        self.original_colors = None
        
        # Store the last clicked point
        self.clicked_point_idx = None
        
        # Connect events
        self.canvas.events.mouse_press.connect(self._on_mouse_press)
        self.canvas.events.mouse_move.connect(self._on_mouse_move)
        self.canvas.events.mouse_release.connect(self._on_mouse_release)
        self.canvas.events.key_press.connect(self._on_key_press)
        
        # Initialize histogram
        self.histogram = None
    
    def reset(self):
        """Reset the canvas by removing all visuals."""
        # Remove all visuals from the scene
        if hasattr(self, 'scatter') and self.scatter is not None:
            self.scatter.parent = None
            self.scatter = None
        
        if hasattr(self, 'histogram') and self.histogram is not None:
            self.histogram.parent = None
            self.histogram = None
        
        if hasattr(self, 'selection_line') and self.selection_line is not None:
            self.selection_line.parent = None
            self.selection_line = None
        
        # Clear the view
        for child in list(self.view.scene.children):
            if child is not self.grid and child is not self.x_label and child is not self.y_label:
                child.parent = None
        
        # Reset selection variables
        self.selection_path = []
        self.selected_mask = None
    
    def set_labels(self, x_label, y_label):
        """Set axis labels.
        
        Args:
            x_label: X-axis label.
            y_label: Y-axis label.
        """
        self.x_label_text = x_label
        self.y_label_text = y_label
        
        # Update labels if they exist
        if hasattr(self, 'x_label') and hasattr(self, 'y_label'):
            self.x_label.text = x_label
            self.y_label.text = y_label
    
    def _on_mouse_press(self, event):
        """Handle mouse press events.
        
        Args:
            event: Mouse press event.
        """
        # Check if Shift key is pressed for lasso selection
        modifiers = event.modifiers
        if modifiers and keys.SHIFT in modifiers:
            # Start lasso selection
            self.is_selecting = True
            self.selection_path = []
            
            # Clear any existing selection line
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
        elif event.button == 1 and self.scatter is not None and hasattr(self, 'x_data') and hasattr(self, 'y_data'):
            # Handle point click (left mouse button)
            # Get position in data coordinates
            tr = self.view.scene.transform
            pos = tr.imap(event.pos)[:2]  # Only take x, y coordinates
            
            # Find the closest point
            distances = np.sqrt((self.x_data - pos[0])**2 + (self.y_data - pos[1])**2)
            closest_idx = np.argmin(distances)
            min_distance = distances[closest_idx]
            
            # Check if the click is close enough to a point (within 10 pixels)
            # Convert 10 pixels to data coordinates
            pixel_size = self.view.camera.transform.scale[0]  # Size of a pixel in data coordinates
            threshold = 10 * pixel_size
            
            if min_distance < threshold:
                print(f"Clicked on point {closest_idx}")
                # Emit signal with the index of the clicked point
                self.point_clicked.emit(closest_idx)
                
                # Highlight the clicked point
                mask = np.zeros_like(self.x_data, dtype=bool)
                mask[closest_idx] = True
                self._highlight_selected_points(mask)
                
                # Prevent event from propagating
                event.handled = True
    
    def _on_mouse_move(self, event):
        """Handle mouse move events.
        
        Args:
            event: Mouse move event.
        """
        if self.is_selecting and self.selection_line is not None:
            # Get position in data coordinates
            tr = self.view.scene.transform
            pos = tr.imap(event.pos)[:2]  # Only take x, y coordinates
            self.selection_path.append(pos)
            
            # Update selection line
            self.selection_line.set_data(pos=np.array(self.selection_path))
            
            # Prevent event from propagating to other handlers (like panning)
            event.handled = True
    
    def _on_mouse_release(self, event):
        """Handle mouse release events.
        
        Args:
            event: Mouse release event.
        """
        if self.is_selecting and self.selection_line is not None:
            # End selection
            self.is_selecting = False
            
            # Get position in data coordinates
            tr = self.view.scene.transform
            pos = tr.imap(event.pos)[:2]  # Only take x, y coordinates
            self.selection_path.append(pos)
            
            # Update selection line
            self.selection_line.set_data(pos=np.array(self.selection_path))
            
            # Check if we have a scatter plot
            if hasattr(self, 'scatter') and self.scatter is not None and hasattr(self, 'x_data') and hasattr(self, 'y_data'):
                # Create a mask for points inside the lasso
                mask = self._points_in_lasso(self.x_data, self.y_data, self.selection_path)
                
                # Highlight selected points
                self._highlight_selected_points(mask)
                
                # Emit selection changed signal with indices of selected points
                selected_indices = np.where(mask)[0]
                self.selection_changed.emit(selected_indices)
            
            # Prevent event from propagating to other handlers
            event.handled = True
    
    def _on_key_press(self, event):
        """Handle key press events.
        
        Args:
            event: Key press event.
        """
        if event.key == 'Escape' and self.is_selecting:
            # Cancel selection
            self.is_selecting = False
            self.selection_path = []
            
            # Remove selection line
            if self.selection_line is not None:
                self.selection_line.parent = None
                self.selection_line = None
            
            # Clear any highlighted points
            if self.selected_mask is not None:
                self._highlight_selected_points(np.zeros_like(self.selected_mask, dtype=bool))
                self.selection_changed.emit(np.array([]))
            
            # Prevent event from propagating to other handlers
            event.handled = True
    
    def _highlight_selected_points(self, mask):
        """Highlight selected points.
        
        Args:
            mask: Boolean mask of selected points.
        """
        if self.scatter is None or not hasattr(self, 'x_data') or not hasattr(self, 'y_data'):
            return
        
        # Store the mask
        self.selected_mask = mask
        
        # Get the original colors from the widget or use default
        if not hasattr(self, 'original_colors') or self.original_colors is None:
            # If we don't have original colors, use the current colors
            if hasattr(self, 'x_data') and hasattr(self, 'y_data'):
                # Get colors from the parent widget
                parent = self.parent()
                if hasattr(parent, 'original_colors') and parent.original_colors is not None:
                    self.original_colors = parent.original_colors.copy()
                else:
                    # Default to blue
                    self.original_colors = np.ones((len(self.x_data), 4)) * np.array([0.5, 0.5, 1.0, 0.7])
        
        # Create a copy of the original colors
        colors = self.original_colors.copy() if hasattr(self, 'original_colors') else np.ones((len(mask), 4)) * np.array([0.5, 0.5, 1.0, 0.7])
        
        # Ensure all points have some opacity (at least 0.5)
        colors[:, 3] = np.maximum(colors[:, 3], 0.5)
        
        # Set selected points to the class color with full opacity
        if np.any(mask):
            # Get the parent widget to access class colors
            parent = self.parent()
            
            # Default highlight color (yellow)
            highlight_color = np.array([1.0, 1.0, 0.0])
            
            # Try to get the next class color from the parent
            if hasattr(parent, 'class_colors') and len(parent.class_colors) > 0:
                # Get the next class ID
                next_class = 1
                if hasattr(parent, 'class_assignments') and parent.class_assignments is not None and np.any(parent.class_assignments > 0):
                    next_class = np.max(parent.class_assignments) + 1
                
                # Limit to the number of available colors
                color_idx = (next_class - 1) % len(parent.class_colors)
                highlight_color = parent.class_colors[color_idx][:3]
            
            # Apply the highlight color
            colors[mask, 0:3] = highlight_color
            colors[mask, 3] = 1.0  # Full opacity for selected points
        
        # Get sizes from parent or use default
        sizes = getattr(self.parent(), 'sizes', np.ones(len(self.x_data)) * 10)
        
        # Update scatter plot colors
        self.scatter.set_data(
            pos=np.column_stack([self.x_data, self.y_data]),
            face_color=colors,
            size=sizes,
            edge_width=0
        )
        
        # Update the canvas
        self.canvas.update()
    
    def _points_in_lasso(self, x, y, lasso_path):
        """Check which points are inside the lasso.
        
        Args:
            x: X coordinates of points.
            y: Y coordinates of points.
            lasso_path: List of (x, y) coordinates defining the lasso path.
        
        Returns:
            Boolean mask of points inside the lasso.
        """
        # Create a Path object from the lasso path
        path = Path(lasso_path)
        
        # Check which points are inside the path
        points = np.column_stack([x, y])
        mask = path.contains_points(points)
        
        return mask
    
    def make_scatter_plot(self, x_data, y_data, colors, sizes, alpha=0.7):
        """Create a scatter plot with the given data.
        
        Args:
            x_data: X coordinates.
            y_data: Y coordinates.
            colors: Colors for each point.
            sizes: Sizes for each point.
            alpha: Alpha value for points.
        """
        # Clear any existing plot
        self.reset()
        
        # Store data for selection
        self.x_data = x_data
        self.y_data = y_data
        self.colors = colors.copy()  # Store original colors
        self.sizes = sizes
        
        # Ensure colors have alpha
        face_color = colors.copy()
        if face_color.shape[1] == 3:
            face_color = np.column_stack([face_color, np.ones(len(face_color)) * alpha])
        else:
            face_color[:, 3] = alpha
        
        # Store original colors for selection highlighting
        self.original_colors = face_color.copy()
        
        # Create scatter plot
        self.scatter = Markers()
        self.scatter.set_data(
            pos=np.column_stack([x_data, y_data]),
            face_color=face_color,
            size=sizes,
            edge_width=0
        )
        
        # Add scatter plot to the scene
        self.view.add(self.scatter)
        
        # Calculate axis limits with some padding
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_padding = x_range * 0.05
        y_padding = y_range * 0.05
        
        # Set axis limits
        self.view.camera.set_range(
            x=(x_min - x_padding, x_max + x_padding),
            y=(y_min - y_padding, y_max + y_padding),
            margin=0
        )
        
        # Add axis with ticks and labels
        self._add_axis_with_ticks(x_min, x_max, y_min, y_max)
        
        # Reset zoom to fit all data
        self.view.camera.set_range()
    
    def _add_axis_with_ticks(self, x_min, x_max, y_min, y_max):
        """Add axis with ticks and labels to the plot.
        
        Args:
            x_min: Minimum x value.
            x_max: Maximum x value.
            y_min: Minimum y value.
            y_max: Maximum y value.
        """
        # Create axis visuals
        x_axis = scene.visuals.Line(pos=np.array([[x_min, y_min], [x_max, y_min]]), color='white', width=2)
        y_axis = scene.visuals.Line(pos=np.array([[x_min, y_min], [x_min, y_max]]), color='white', width=2)
        
        # Add axes to the scene
        self.view.add(x_axis)
        self.view.add(y_axis)
        
        # Create tick marks and labels
        # X-axis ticks
        x_ticks = np.linspace(x_min, x_max, 5)
        for x in x_ticks:
            # Tick mark
            tick = scene.visuals.Line(pos=np.array([[x, y_min], [x, y_min - (y_max - y_min) * 0.02]]), color='white', width=1)
            self.view.add(tick)
            
            # Tick label
            label = scene.visuals.Text(text=f"{x:.2f}", pos=[x, y_min - (y_max - y_min) * 0.05], color='white', font_size=8)
            self.view.add(label)
            
            # Vertical grid line
            grid_line = scene.visuals.Line(pos=np.array([[x, y_min], [x, y_max]]), color='white', width=0.5)
            grid_line.set_gl_state(blend=True)
            grid_line.opacity = 0.2
            self.view.add(grid_line)
        
        # Y-axis ticks
        y_ticks = np.linspace(y_min, y_max, 5)
        for y in y_ticks:
            # Tick mark
            tick = scene.visuals.Line(pos=np.array([[x_min, y], [x_min - (x_max - x_min) * 0.02, y]]), color='white', width=1)
            self.view.add(tick)
            
            # Tick label
            label = scene.visuals.Text(text=f"{y:.2f}", pos=[x_min - (x_max - x_min) * 0.1, y], color='white', font_size=8)
            self.view.add(label)
            
            # Horizontal grid line
            grid_line = scene.visuals.Line(pos=np.array([[x_min, y], [x_max, y]]), color='white', width=0.5)
            grid_line.set_gl_state(blend=True)
            grid_line.opacity = 0.2
            self.view.add(grid_line)
        
        # Update axis labels
        self.x_label.text = self.x_label_text
        self.y_label.text = self.y_label_text
    
    def make_2d_histogram(self, x_data, y_data, bin_number=100, log_scale=False):
        """Create a 2D histogram.
        
        Args:
            x_data: X coordinates of points.
            y_data: Y coordinates of points.
            bin_number: Number of bins.
            log_scale: Whether to use log scale.
        """
        # Clear any existing plot
        self.reset()
        
        # Store data for reference
        self.x_data = x_data
        self.y_data = y_data
        
        # Calculate histogram
        H, xedges, yedges = np.histogram2d(x_data, y_data, bins=bin_number)
        
        # Apply log scale if requested
        if log_scale:
            H = np.log1p(H)
        
        # Normalize histogram
        H = H / np.max(H)
        
        # Create image visual
        self.histogram = visuals.Image(
            H.T,  # Transpose to match orientation
            cmap='viridis',
            parent=self.view.scene
        )
        
        # Set image position and scale
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        self.histogram.transform = scene.transforms.STTransform(
            scale=(
                (x_max - x_min) / bin_number,
                (y_max - y_min) / bin_number
            ),
            translate=(x_min, y_min)
        )
        
        # Add axis with ticks and labels
        self._add_axis_with_ticks(x_min, x_max, y_min, y_max)
        
        # Reset zoom to fit all data
        self.view.camera.set_range()
    
    def make_1d_histogram(self, data, bin_number=100, log_scale=False):
        """Create a 1D histogram.
        
        Args:
            data: Data to plot.
            bin_number: Number of bins.
            log_scale: Whether to use log scale.
        """
        # Clear any existing plot
        self.reset()
        
        # Store data for reference
        self.x_data = data
        
        # Calculate histogram
        hist, bin_edges = np.histogram(data, bins=bin_number)
        
        # Apply log scale if requested
        if log_scale:
            hist = np.log1p(hist)
        
        # Normalize histogram
        hist = hist / np.max(hist)
        
        # Create bar visual
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Create line visual for histogram
        pos = []
        for i, (x, h) in enumerate(zip(bin_centers, hist)):
            pos.append([x, 0])
            pos.append([x, h])
        
        self.histogram = visuals.Line(
            pos=np.array(pos),
            color='white',
            width=2,
            connect='segments',
            parent=self.view.scene
        )
        
        # Add axis with ticks and labels
        x_min, x_max = np.min(data), np.max(data)
        y_min, y_max = 0, 1
        self._add_axis_with_ticks(x_min, x_max, y_min, y_max)
        
        # Reset zoom to fit all data
        self.view.camera.set_range() 