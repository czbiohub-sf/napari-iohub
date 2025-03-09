"""
Demo for the high-performance Vispy plotter in napari-iohub.

This example creates synthetic data with UMAP and PCA dimensions, along with other
features that can be visualized using the Vispy plotter widget. It demonstrates
how to create points and labels layers with rich feature data for visualization.
"""

import numpy as np
from skimage import data, feature, filters, morphology
from skimage.measure import regionprops
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

import napari
from napari_iohub._vispy_plotter import vispy_plotter_widget


def create_demo_data():
    """Create demo data with cells, labels, and points with UMAP/PCA features"""
    # Get sample 3D image data
    print("Loading 3D cell data...")
    cells3d = data.cells3d()
    nuclei = cells3d[:, 1]  # Nuclei channel

    # Process the nuclei to get labels
    print("Processing nuclei data...")
    nuclei_smoothed = filters.gaussian(nuclei, sigma=5)
    nuclei_thresholded = nuclei_smoothed > filters.threshold_otsu(nuclei_smoothed)
    nuclei_labels = morphology.label(nuclei_thresholded)
    
    # Find maxima points
    print("Finding nuclei maxima...")
    nuclei_points = feature.peak_local_max(nuclei_smoothed, min_distance=20)
    
    # Create features for the labels
    print("Creating features for visualization...")
    props = regionprops(nuclei_labels)
    
    # Extract basic features
    label_ids = [p.label for p in props]
    areas = [p.area for p in props]
    centroids = np.array([p.centroid for p in props])
    
    # Extract intensity features
    intensity_mean = [np.mean(nuclei_smoothed[nuclei_labels == label]) for label in label_ids]
    intensity_max = [np.max(nuclei_smoothed[nuclei_labels == label]) for label in label_ids]
    intensity_min = [np.min(nuclei_smoothed[nuclei_labels == label]) for label in label_ids]
    intensity_std = [np.std(nuclei_smoothed[nuclei_labels == label]) for label in label_ids]
    
    # Create a feature matrix for dimensionality reduction
    # Only use properties that are available for 3D images
    feature_matrix = np.column_stack([
        areas,
        centroids[:, 0],  # z
        centroids[:, 1],  # y
        centroids[:, 2],  # x
        intensity_mean,
        intensity_max,
        intensity_min,
        intensity_std
    ])
    
    # Handle any NaN values
    feature_matrix = np.nan_to_num(feature_matrix)
    
    # Standardize the features
    print("Standardizing features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    # Perform PCA
    print("Performing PCA...")
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_features)
    
    # Perform UMAP
    print("Performing UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_result = reducer.fit_transform(scaled_features)
    
    # Create a DataFrame with all features
    features_df = pd.DataFrame({
        'label': label_ids,
        'area': areas,
        'centroid_z': centroids[:, 0],
        'centroid_y': centroids[:, 1],
        'centroid_x': centroids[:, 2],
        'intensity_mean': intensity_mean,
        'intensity_max': intensity_max,
        'intensity_min': intensity_min,
        'intensity_std': intensity_std,
        'PCA_1': pca_result[:, 0],
        'PCA_2': pca_result[:, 1],
        'PCA_3': pca_result[:, 2],
        'UMAP_1': umap_result[:, 0],
        'UMAP_2': umap_result[:, 1]
    })
    
    # Add a cluster ID for demonstration
    n_clusters = 3
    features_df['CLUSTER_ID'] = np.random.randint(0, n_clusters, size=len(features_df))
    
    # Create point features based on nuclei points
    point_features = pd.DataFrame({
        'index': np.arange(len(nuclei_points)),
        'z': nuclei_points[:, 0],
        'y': nuclei_points[:, 1],
        'x': nuclei_points[:, 2],
        'intensity': [nuclei[tuple(p)] for p in nuclei_points],
    })
    
    # Add synthetic UMAP and PCA for points
    # (In a real scenario, you would compute these from actual features)
    n_points = len(point_features)
    point_pca = np.random.randn(n_points, 3)
    point_umap = np.random.randn(n_points, 2)
    
    point_features['PCA_1'] = point_pca[:, 0]
    point_features['PCA_2'] = point_pca[:, 1]
    point_features['PCA_3'] = point_pca[:, 2]
    point_features['UMAP_1'] = point_umap[:, 0]
    point_features['UMAP_2'] = point_umap[:, 1]
    point_features['CLUSTER_ID'] = np.random.randint(0, n_clusters, size=n_points)
    
    
    return {
        'nuclei': nuclei,
        'nuclei_labels': nuclei_labels,
        'nuclei_points': nuclei_points,
        'features_df': features_df,
        'point_features': point_features
    }



def main():
    """Run the demo"""
    # Create the data
    data_dict = create_demo_data()
    nuclei = data_dict['nuclei']
    nuclei_labels = data_dict['nuclei_labels']
    nuclei_points = data_dict['nuclei_points']
    features_df = data_dict['features_df']
    point_features = data_dict['point_features']
    
    # Create the viewer and add the layers
    viewer = napari.Viewer()
    
    # Add the image layer
    image_layer = viewer.add_image(
        nuclei, 
        name='nuclei', 
        contrast_limits=(10000, 65355)
    )
    
    # Add the labels layer with features
    labels_layer = viewer.add_labels(
        nuclei_labels, 
        name='nuclei_labels'
    )
    labels_layer.features = features_df
    
    # Add the points layer with features
    points_layer = viewer.add_points(
        nuclei_points, 
        name='nuclei_points', 
        size=10,
        face_color='yellow',
        blending='additive', 
        opacity=0.7
    )
    points_layer.features = point_features
    
    # Add bounding boxes for better visualization
    for layer in viewer.layers:
        if hasattr(layer, 'bounding_box'):
            layer.bounding_box.visible = True
    
    # Set the view to 3D and rotate camera
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (2, 15, 150)
    
    # Add the Vispy plotter widget
    plotter = vispy_plotter_widget(viewer)
    viewer.window.add_dock_widget(
        plotter, 
        name="High-Performance Plotter",
        area="right"
    )
    
    # Print instructions
    print("\nDemo loaded successfully!")
    print("\nTry these plotting options:")
    print("\nFor nuclei_labels layer:")
    print("1. Plot 'UMAP_1' vs 'UMAP_2' - clustering will be automatically detected")
    print("2. Plot 'PCA_1' vs 'PCA_2' - clustering will be automatically detected")
    print("3. Plot 'area' vs 'intensity_mean'")
    
    print("\nFor nuclei_points layer:")
    print("1. Plot 'UMAP_1' vs 'UMAP_2' - clustering will be automatically detected")
    print("2. Plot 'PCA_1' vs 'PCA_2' - clustering will be automatically detected")
    print("3. Plot 'z' vs 'intensity'")
    
    print("\nTry both scatter plot and histogram visualization modes")
    
    print("\nNote on Vispy Plotter:")
    print("- The Vispy plotter is optimized for performance with large datasets")
    print("- It uses hardware acceleration via OpenGL for rendering")
    print("- Scatter plots support millions of points with interactive performance")
    print("- Histograms provide efficient visualization of data distributions")
    print("- Clustering columns are automatically detected and used for coloring points")
    
    print("\nInteractive Selection Features:")
    print("- Hold Shift + drag to draw a lasso around points to select them")
    print("- Selected points are highlighted in both the plot and the napari viewer")
    print("- Press Escape to cancel a selection")
    print("- Regular mouse drag (without Shift) pans the plot")
    
    # Keep the viewer open
    napari.run()


if __name__ == "__main__":
    main() 