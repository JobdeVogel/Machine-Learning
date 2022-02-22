"""
THIS METHOD CALCULATES EXPECTED CLUSTERS
BASED ON HIERARCHICAL CLUSTERING

WE ASSUME 5 CLUSTERS:
- Houses
- Cars
- Fences
- Traffic Lights
- Trees
"""

"""
Pseudocode
https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/

- Create a proximity matrix for the data
    format: 500 x 500 matrix
            499 x 499 matrix
            498 x 498 matrix

- Find the smallest value in the proximity matrix
- Update proximity matrix (combine the previous values)
            Use dataframe for the naming?
            So if point 1 and 2 closest, naming will be 1-2
            Store in another matrix?

            Step 1 remove column point
            Step 2 remove row point
            Step 3 insert new column
            Step 4 insert new row

- Based on k, select len(pointcloud) - k
"""

import numpy as np
import pandas as pd
import time

data = pd.read_csv('csv_data.csv')

"""
RETURNS A 500X500 PROXIMITY MATRIX
"""
def minkowski(p_norm, df):
    # Convert df to np array
    data = df.values
    rows, columns = data.shape

    # Create an empty container with shape
    # (amount of features, amount of points, amount of points)
    distances = np.empty((columns, rows, rows), dtype=np.float32)
    
    # For each feature, create a 500x500 distance matrix
    for i in range(columns):
        feature = data[:, i]
        feature_x, feature_y = np.meshgrid(feature, feature.T)

        feature_dif = np.abs(feature_x - feature_y)
        distances[i] = feature_dif

    # Calculate minkowski distances
    p_pow_feature_distances = distances**p_norm
    minkowski_distances = np.sum(p_pow_feature_distances, axis=0)**(1/p_norm)
    
    return minkowski_distances



