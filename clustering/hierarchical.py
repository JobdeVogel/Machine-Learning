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

from msilib.schema import IniFile
from turtle import pos
import numpy as np
import pandas as pd
import time

data = pd.read_csv('csv_data.csv')[:20]
INFINITY = np.Inf

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
    
    np.fill_diagonal(minkowski_distances, INFINITY)
    
    # Dataframe
    # proximity_matrix = pd.DataFrame(minkowski_distances, 
    #                                     index=[str(i) for i in range(rows)], columns=[str(i) for i in range(rows)])

    return minkowski_distances

# Delete the highest highest value of result, both column and row
# Change the name of lowest value of result to {result}
# Add result to some kind of library
# Recalculate the values of the column/row with lowest value result

def minimum_proximity_pos(proximity_matrix: pd.DataFrame):
    # Convert df to np
    # np_matrix = proximity_matrix.values

    # Find the position of lowest value without zeros
    min_value = np.amin(proximity_matrix[proximity_matrix < INFINITY])
    position = np.where(proximity_matrix == min_value)
    
    if len(position[0]) > 2:
        position = np.array([position[0][0], position[1][0]])
    else:
        position = position[0]
    
    return position

def hierarchical_clustering(proximity_matrix: pd.DataFrame, type='nearest'):
    pointCount = len(proximity_matrix)

    # Create an initial array to store all clusters
    clusters = np.arange(pointCount, dtype=np.uint)
    clusters = np.meshgrid(clusters, clusters.T)[1]

    # Updated proximity matrix stores all intermediate distances
    # for each iteration. Consider point x and y being closest,
    # distances to point y will be overwritte to 0, and discarded
    # in following iterations. Point x will be considered to be a
    # group of point x and y, using the max, min or avg distance.
    updated_proximity_matrix = proximity_matrix.copy()

    # Start loop...
    for i in range(1, pointCount):
        # What if a distance already is 0 ????????????
        point_x, point_y = minimum_proximity_pos(updated_proximity_matrix)

        # Assign point_y to (group of) point_x
        clusters[point_y, i:] = point_x
        
        # If point_y is a cluster, merge the cluster with point_x
        connected_points = np.where(clusters[:, i] == point_y)
        clusters[connected_points, i:] = point_x 

        # Find the members of cluster x
        cluster_members = np.where(clusters[:, i] == point_x)[0]

        # Find to which points or clusters 
        # point_x or point_y are not connected to yet
        non_zeros_x = updated_proximity_matrix[point_x]
        non_zeros_x = np.where(non_zeros_x < INFINITY)
        non_zeros_y = updated_proximity_matrix[point_y]
        non_zeros_y = np.where(non_zeros_y < INFINITY)

        possible_dis_indices = np.intersect1d(non_zeros_x, non_zeros_y)
        zeros = np.setxor1d(non_zeros_x, non_zeros_y)

        # New distances to be assigned to group of point_x, at possible_dis_indices
        ########## GOES WORNG HERE #####################
        new_distances = proximity_matrix[:, cluster_members][possible_dis_indices]

        # Find the distances to calculate minimum, maximum or average from
        if type == 'nearest':
            new_distances = new_distances.min(axis=1)
        elif type == 'farthest':
            new_distances = new_distances.max(axis=1)
        elif type == 'average':
            new_distances = new_distances.mean(axis=1)
        else:
            print('Type not possible for hierarchical clustering, choose between: \'nearest\', \'farthest\' or \'average\'')
            return

        # Set point_x connections that are assigned to 0        
        updated_proximity_matrix[point_x, zeros] = INFINITY
        updated_proximity_matrix[zeros, point_x] = INFINITY

        # Set point_x relations that are not assigned to new_distances
        updated_proximity_matrix[point_x, possible_dis_indices] = new_distances
        updated_proximity_matrix[possible_dis_indices, point_x] = new_distances
        
        # Point_y is now properly assigned to a group, set distance to 0
        updated_proximity_matrix[point_y] = INFINITY
        updated_proximity_matrix[:, point_y] = INFINITY

    return clusters

"""
CALCULATE THE MINIMUM, MAXIMUM OR AVERAGE
MINKOWSKI DISTANCE

USE THE ORIGINAL PROXIMITY MATRIX!!!

Type can be: 
    - 'nearest'
    - 'farthest'
    - 'average'
"""


proximity_matrix = minkowski(2, data)
# pd.DataFrame(proximity_matrix).to_csv('proximity_matrix.csv')

res = hierarchical_clustering(proximity_matrix, 'nearest')
print(res)