"""
PSEUDOCODE

- Define eps and minPts = 4
- Define corepoints by points with at least minpts in neighborhood eps
- Define non-core points if 0 < pts < 4
- Define outliers if pts == 0
"""

import numpy as np
from matplotlib import pyplot as plt
import time

INFINITY = np.Inf

#https://www.section.io/engineering-education/dbscan-clustering-in-python/
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

# Specify the eps value using the min distance graph
# Please input value after closing the graph
def calc_eps(proximity_matrix):
    min_distances = proximity_matrix.min(axis=1)
    sorted_dis = np.sort(min_distances)

    plt.title("Nearest Neighbour Distance") 
    plt.xlabel("Point Indices") 
    plt.ylabel("Minimum Distance") 

    x = np.arange(proximity_matrix.shape[0])
    y = sorted_dis
    print("Please enter maximum curvature (default value is 0.12): ", end="\r")
    
    plt.plot(x,y) 
    plt.show()

    eps = input("Please enter maximum curvature (default value is 0.12): ")
    eps = 0.12
    
    # If user does not select eps, use 0.1
    if not eps:
        eps = 0.1   

    return float(eps)

# Recursive algorithm to assign clusters to points
def assign_neighbours(clusters, neighbour_bools, core_points, index, to_assign):
    # Find all the neighbours
    neighbours = np.where(neighbour_bools[:, index] == True)[0]

    # All neighbours are now assigned to index
    clusters[neighbours] = to_assign

    neighbour_bools[index] = False
    neighbour_bools[neighbours] = False

    for neighbour in neighbours:
        if neighbour in core_points:
            assign_neighbours(clusters, neighbour_bools, core_points, neighbour, to_assign)
    
    return

def main(p_norm, data):
    start_0 = time.time()
    # Container for the clusters
    clusters = np.arange(data.shape[0])
    minpts = data.shape[1] * 2
    minpts = 7

    # Calculate or import proximity matrix
    proximity_matrix = minkowski(p_norm, data)
    #proximity_matrix = pd.read_csv('proximity_matrix.csv')

    end_0 = time.time()
    # Visualize graph to specify eps value
    eps = calc_eps(proximity_matrix)
    
    start_1 = time.time()
    # Bool that indicates if neighbour closer than eps distance
    neighbour_bools = proximity_matrix < eps

    # Amount of neighbours closer than eps distance
    count = np.count_nonzero(neighbour_bools, axis=1)

    # Containers for core points, non core points and outliers
    core_points = np.where(count >= minpts)[0]

    # non_core_points = np.where((count > 0) & (count < minpts))[0]
    outliers = np.where(count == 0)[0]

    assigned_labels = []

    # Recursively assign clusters
    # If neighbour is a non-core point, do not search for new neighbours
    for core in core_points:
        # If the core point is not already assigned
        if clusters[core] == core:
            assigned_labels.append(core)
            # Recursively go through all neighbours
            assign_neighbours(clusters, neighbour_bools, core_points, core, core)

    # Find which indices are used as cluster
    cluster_idxs = np.where(np.bincount(clusters) > minpts - 1)[0]

    # Change cluster_idxs to cluster 0, 1, 2, 3 ... 5
    # instead of 207, 306, 307, 309 ... 410
    for i, cluster in enumerate(cluster_idxs):
        clusters[np.where(clusters == cluster)] = i

    # assign all that are not assigned    
    for cluster in clusters:
        if cluster not in assigned_labels:
            clusters[cluster] = i+1

    end_1 = time.time()
    print('DBSCAN clustering computed in {} seconds'.format(round((end_1-start_1) + (end_0-start_0), 2)))

    return clusters