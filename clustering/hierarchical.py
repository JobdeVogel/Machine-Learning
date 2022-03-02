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
HIERARCHICAL CLUSTERING

PSEUDECODE:
    -   Calculate initial proximity matrix using
        minkowski distance. Distances from points
        to themselves are assigned with distance
        equals infinity;
    -   


Pseudocode
https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/

"""

import numpy as np
import pandas as pd
import time

INFINITY = np.Inf

"""
MINKOWSKI DISTANCE
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

"""
RETURNS A 2X1 MATRIX CONTAINING CLOSEST PAIR OF POINT INDICES
"""
def minimum_proximity_pos(proximity_matrix):
    # Find the position of lowest value without zeros
    min_value = np.amin(proximity_matrix[proximity_matrix < INFINITY])
    position = np.where(proximity_matrix == min_value)
    
    if len(position[0]) > 2:
        position = np.array([position[0][0], position[1][0]])
    else:
        position = position[0]
    
    return position

"""
PRINTS THE LOADING PERCENTAGE OF THE ALGORITHM
"""
def loading(iteration, count):
    percentage = round((iteration / (count - 1)) * 100, 2)

    if percentage < 100:
        print('Hierarchical Clustering ' + str(percentage) + '% completed', end="\r")
    else:
        print('Hierarchical Clustering ' + str(percentage) + '% completed')
    
    return

"""
RETURNS A 500X500 MATRIX IN WHICH EACH COLUMN
STANDS FOR ONE ITERATION. EVERY ROW STANDS FOR
A POINT INDEX WITH A VALUE INDICATING THE CLUSTER
IT BELONGS TO.

TO EXTRACT K CLUSTERS, TAKE CLUSTERS[LEN(CLUSTERS) - K]
"""
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
        loading(i, pointCount)

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
        non_inf_x = updated_proximity_matrix[point_x]
        non_inf_x = np.where(non_inf_x < INFINITY)
        non_inf_y = updated_proximity_matrix[point_y]
        non_inf_y = np.where(non_inf_y < INFINITY)

        possible_dis_indices = np.intersect1d(non_inf_x, non_inf_y)
        zeros = np.setxor1d(non_inf_x, non_inf_y)

        # New distances to be assigned to group of point_x, at possible_dis_indices
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
RETURNS:
- [0] AN NP ARRAY WITH ALL INDICES ORDERED BY CLUSTER
- [1] THE SIZES OF THE CLUSTERS

TO BE PLOTTED WITH MATPLOTLIB
"""

def extract_cluster(clusters, k):
    cluster = clusters[:, len(clusters) - k]

    # All unique origins of the clusters
    cluster_origins = np.unique(cluster)
    
    # Initialize containers for cluster sizes and indices
    # Add the first cluster to the cluster indices and sizes
    cluster_assignment = np.zeros([0], dtype=np.int32)
    cluster_indices = np.zeros([0])
    
    # For each other cluster, add the indices and sizes to the
    # corresponding containers
    for i, origin in enumerate(cluster_origins):
        indices = np.where(cluster == origin)
        
        cluster_indices = np.append(cluster_indices, indices)
        cluster_assignment = np.append(cluster_assignment, np.repeat(i, indices[0].size))

    sorting_ints = cluster_indices.argsort()
    cluster_assignment = cluster_assignment[sorting_ints]

    return cluster_assignment

def main(p_norm, k, type, data):
    start = time.time()
    # Calculate the proximity matrix for the data
    proximity_matrix = minkowski(p_norm, data)

    # Save the proximity matrix to a csv
    # pd.DataFrame(proximity_matrix).to_csv('proximity_matrix.csv')

    # Calculate the clusters
    clusters = hierarchical_clustering(proximity_matrix, type)
    result = extract_cluster(clusters, k)

    end = time.time()

    print('Hierarchical clustering computed in {} seconds'.format(round(end-start, 2)))
    return result