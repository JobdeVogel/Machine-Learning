import os
import numpy as np
import pandas as pd
from functions import loading
from options import set_options
from features.convexhull import getConvexHullArea

class pointCloudObject:
    def __str__(self):
        return 'Point Class'

    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates

    # Feature 1: calculate the max z coordinate
    def max_z(self):
        return self.coordinates['z'].max(0)

    # Feature 2: convex hull area
    def convex_hull_area(self):
        return getConvexHullArea(self.coordinates, 1)
    
    # Feature 3: convec hull volume
    def convex_hull_volume(self):
        return 1

def xyz_to_df(directory, filename):
    filename = os.path.join(directory, filename)

    pointCloud = pd.read_table(filename, skiprows=0, delim_whitespace=True,
                            names=['x', 'y', 'z'])
    
    return pointCloud

def df_to_csv(dataframe, csv_name):
    dataframe.to_csv(csv_name, index=False)

# Convert all the 1D feature arrays to ndarray
def generate_feature_array(*args: np.array):
    return np.stack((args), axis=0).T

def normalize_by_column(dataframe):
    # Normalize each column from 0 to 1
    return (dataframe-dataframe.min())/(dataframe.max()-dataframe.min())

# final function
def preprocess(directory):
    amount_of_files = len(os.listdir(directory))

    z_indices = np.empty([amount_of_files], dtype=np.float64)
    convex_hull_volumes = np.empty([amount_of_files], dtype=np.float64)
    convex_hull_areas = np.empty([amount_of_files], dtype=np.float64)

    # For each file in data directory
    for i, filename in enumerate(os.listdir(directory)):

        loading(i, amount_of_files)

        # Convert file data to dataframe
        data = xyz_to_df(directory, filename)
        name = 'Point_Cloud_{}'.format(i)

        # Create a pointCloud object
        pointCloud = pointCloudObject(name, data)

        # Add all features of pointCloud to their feature containers
        z_indices[i] = pointCloud.max_z()
        convex_hull_areas[i] = pointCloud.convex_hull_area()
        convex_hull_volumes[i] = pointCloud.convex_hull_volume()

    print('\n')
    feature_array = generate_feature_array(z_indices, convex_hull_areas, convex_hull_volumes)
    feature_array_df = pd.DataFrame(feature_array, columns=['z_indices', 'convex_hull_areas', 'convex_hull_volumes'])

    feature_array_norm = normalize_by_column(feature_array_df)
    
    return feature_array_norm

set_options()

print(preprocess('./data'))