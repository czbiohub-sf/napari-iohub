"""
Utility functions for the Vispy plotter.

This module provides utility functions for the Vispy plotter, such as
color mapping, data processing, and coordinate transformations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# Predefined colors for clusters
CLUSTER_COLORS = [
    (1.0, 0.0, 0.0, 1.0),  # Red
    (0.0, 1.0, 0.0, 1.0),  # Green
    (0.0, 0.0, 1.0, 1.0),  # Blue
    (1.0, 1.0, 0.0, 1.0),  # Yellow
    (1.0, 0.0, 1.0, 1.0),  # Magenta
    (0.0, 1.0, 1.0, 1.0),  # Cyan
    (1.0, 0.5, 0.0, 1.0),  # Orange
    (0.5, 0.0, 1.0, 1.0),  # Purple
    (0.0, 0.5, 0.0, 1.0),  # Dark green
    (0.5, 0.5, 0.0, 1.0),  # Olive
    (0.5, 0.0, 0.5, 1.0),  # Dark purple
    (0.0, 0.5, 0.5, 1.0),  # Teal
]

def get_cluster_colors(cluster_ids, colormap_name='viridis'):
    """Get colors for cluster IDs.
    
    Args:
        cluster_ids: Array of cluster IDs.
        colormap_name: Name of the matplotlib colormap to use.
    
    Returns:
        Array of RGBA colors for each cluster ID.
    """
    # Get unique cluster IDs
    unique_ids = np.unique(cluster_ids)
    n_clusters = len(unique_ids)
    
    # Create colors array
    colors = np.ones((len(cluster_ids), 4))
    
    try:
        # Get matplotlib colormap
        cmap = plt.get_cmap(colormap_name)
        
        # Assign colors to each cluster
        for i, cluster_id in enumerate(unique_ids):
            # Create mask for this cluster
            mask = cluster_ids == cluster_id
            
            # Assign color
            if cluster_id == -1:  # Noise points
                colors[mask] = [0.5, 0.5, 0.5, 0.5]  # Gray, semi-transparent
            else:
                # Normalize color index to 0-1 range
                color_idx = i / max(1, n_clusters - 1)
                colors[mask] = cmap(color_idx)
    except:
        # Fallback to predefined colors if colormap fails
        print(f"Error using colormap '{colormap_name}'. Falling back to predefined colors.")
        for i, cluster_id in enumerate(unique_ids):
            mask = cluster_ids == cluster_id
            if cluster_id == -1:
                colors[mask] = [0.5, 0.5, 0.5, 0.5]  # Gray, semi-transparent
            else:
                # Use predefined colors and cycle through them
                color_idx = i % len(CLUSTER_COLORS)
                colors[mask] = CLUSTER_COLORS[color_idx]
    
    return colors


def estimate_bin_number(data, max_bins=100):
    """Estimate a reasonable number of bins for histograms.
    
    Args:
        data: Data to bin.
        max_bins: Maximum number of bins.
    
    Returns:
        Estimated number of bins.
    """
    # Use Freedman-Diaconis rule
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr * len(data)**(-1/3)
    
    if bin_width == 0:
        return max_bins
    
    data_range = np.max(data) - np.min(data)
    n_bins = int(data_range / bin_width)
    
    # Limit to max_bins
    return min(n_bins, max_bins)


def points_in_current_slice(z_values, current_slice, slice_thickness=1.0):
    """Calculate weights for points based on their distance from the current slice.
    
    Args:
        z_values: Z coordinates of points.
        current_slice: Current slice position.
        slice_thickness: Thickness of the slice.
    
    Returns:
        Array of weights (0-1) for each point.
    """
    # Calculate distance from current slice
    distances = np.abs(z_values - current_slice)
    
    # Normalize distances to 0-1 range
    weights = np.clip(1.0 - distances / slice_thickness, 0.0, 1.0)
    
    return weights 