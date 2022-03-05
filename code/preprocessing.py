"""
PREPROCCESSING

DEVELOPED BY:
Job de Vogel
"""

"""
THIS SCRIPT PREPROCESSES THE .XYZ DATA
USING
- FEATURES
- NORMALIZATION
- CLIPPING EXTREME VALUES
"""

import os
import numpy as np
import pandas as pd
from functions import loading, color_plt
from features.zheight import z_height
from features.shaperatio import shapeRatio
from features.convexhull import getConvexHullArea
from features.pointcloudbb import pointcloud_bounding_box

class pointCloudObject:
    def __str__(self):
        return 'Point Class'

    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates

    # Feature 1: calculate the max z coordinate
    def max_z(self):
        return z_height(self.coordinates)

    # Feature 2: convex hull area
    def convex_hull_area(self):
        """
        Precision factor indicates precision of area        
        """
        prec_factor = 1
        return getConvexHullArea(self.coordinates, prec_factor)
    
    # Feature 3: minimal bounding box volume
    def bounding_box_volume(self):
        return pointcloud_bounding_box(self.coordinates)
    
    # Feature 4: y_range x_range ratio
    def shape_ratio(self):
        return shapeRatio(self.coordinates, 1)

def xyz_to_df(directory, filename):
    filename = os.path.join(directory, filename)

    pointCloud = pd.read_table(filename, skiprows=0, delim_whitespace=True,
                            names=['x', 'y', 'z'])
    
    return pointCloud

def df_to_csv(dataframe, csv_name, index_bool=False):
    dataframe.to_csv(csv_name, index=index_bool)

# Convert all the 1D feature arrays to ndarray
def generate_feature_array(*args: np.array):
    return np.stack((args), axis=0).T

def normalize_by_column(dataframe):
    # Normalize each column from 0 to 1
    return (dataframe-dataframe.min())/(dataframe.max()-dataframe.min())

# final function
def preprocess(directory):
    amount_of_files = len(os.listdir(directory))

    z_heights = np.empty([amount_of_files], dtype=np.float64)
    shape_ratios = np.empty([amount_of_files], dtype=np.float64)
    convex_hull_areas = np.empty([amount_of_files], dtype=np.float64)
    bounding_box_volumes = np.empty([amount_of_files], dtype=np.float64)
    
    # For each file in data directory
    for i, filename in enumerate(os.listdir(directory)):
        loading('Preprocessing', i, amount_of_files)

        # Convert file data to dataframe
        data = xyz_to_df(directory, filename)
        name = 'Point_Cloud_{}'.format(i)

        # Create a pointCloud object
        pointCloud = pointCloudObject(name, data)

        # Add all features of pointCloud to their feature containers
        z_heights[i] = pointCloud.max_z()
        shape_ratios[i] = pointCloud.shape_ratio()
        convex_hull_areas[i] = pointCloud.convex_hull_area()
        bounding_box_volumes[i] = pointCloud.bounding_box_volume()
        
    # Clip the values
    z_heights = np.clip(z_heights, 0, 15)
    shape_ratios = np.clip(shape_ratios, 0, 2)
    convex_hull_areas = np.clip(convex_hull_areas, 0, 100)
    bounding_box_volumes = np.clip(bounding_box_volumes, 0, 3000)

    # Format the features
    feature_array = generate_feature_array(z_heights, convex_hull_areas, bounding_box_volumes)
    feature_array_df = pd.DataFrame(feature_array, columns=['z_height', 'convex_hull_areas', 'bounding_box_volumes'])

    # Normalize the features
    feature_array_norm = normalize_by_column(feature_array_df)
    
    # Specify if you want features_array_df or features_array_norm:
    return feature_array_norm

# PRINT THE OBSERVED CLUSTERS TO A PLOT
# FEATURES = ['z_height', 'convex_hull_areas', 'bounding_box_volumes']

# colors = ['green', 'yellow', 'red', 'blue', 'orange', 'black']
# color_selection = []

# for i in range(5):
#     color = colors[i]
#     for i in range(100):
#         color_selection.append(color)

# features = preprocess('./data')

# # Print and plot result
# color_plt(features, color_selection, *FEATURES)