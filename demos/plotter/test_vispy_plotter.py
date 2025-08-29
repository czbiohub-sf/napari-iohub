#%%
import numpy as np
import napari
import pandas as pd
from napari.utils import nbscreenshot
from napari_iohub.vispy_plotter import vispy_plotter_widget

# Create synthetic data
def create_synthetic_data():
    # Create a 3D volume with some objects
    volume = np.zeros((30, 100, 100), dtype=np.uint16)
    
    # Create some objects at different frames
    for frame in range(30):
        # Add 5 objects per frame
        for i in range(5):
            # Random position
            z = frame
            y = np.random.randint(20, 80)
            x = np.random.randint(20, 80)
            
            # Create a small sphere
            radius = np.random.randint(3, 8)
            z_min, z_max = max(0, z-radius//2), min(30, z+radius//2)
            y_min, y_max = max(0, y-radius), min(100, y+radius)
            x_min, x_max = max(0, x-radius), min(100, x+radius)
            
            # Set the object label (unique ID)
            label_id = frame * 10 + i + 1
            
            # Create a sphere
            for zz in range(z_min, z_max):
                for yy in range(y_min, y_max):
                    for xx in range(x_min, x_max):
                        if ((zz-z)**2 + (yy-y)**2 + (xx-x)**2) <= radius**2:
                            volume[zz, yy, xx] = label_id
    
    # Create features for the objects
    features_list = []
    for frame in range(30):
        for i in range(5):
            label_id = frame * 10 + i + 1
            
            # Get object properties
            z_indices, y_indices, x_indices = np.where(volume == label_id)
            if len(z_indices) == 0:
                continue
                
            # Calculate centroid
            centroid_z = np.mean(z_indices)
            centroid_y = np.mean(y_indices)
            centroid_x = np.mean(x_indices)
            
            # Calculate other features
            volume_feature = len(z_indices)
            intensity = np.random.uniform(50, 200)
            
            # Add to features list
            features_list.append({
                'label': label_id,
                'frame': frame,
                'centroid_z': centroid_z,
                'centroid_y': centroid_y,
                'centroid_x': centroid_x,
                'volume': volume_feature,
                'intensity': intensity,
                'cluster_id': np.random.randint(1, 4)  # Random cluster ID
            })
    
    # Create DataFrame
    features_df = pd.DataFrame(features_list)
    
    return volume, features_df

# Create synthetic data
volume, features = create_synthetic_data()

# Create a viewer
viewer = napari.Viewer(ndisplay=3)

# Add the labels layer with features
labels_layer = viewer.add_labels(volume, name='synthetic_labels')
labels_layer.features = features

# Set the camera angles
viewer.camera.angles = (0, 10, 150)

# Add the Vispy plotter widget
plotter = vispy_plotter_widget(viewer)
viewer.window.add_dock_widget(plotter, name='Vispy Plotter')

# Run napari
napari.run()
# %%
