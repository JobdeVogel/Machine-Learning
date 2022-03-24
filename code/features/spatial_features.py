"""
THIS CODE USES THE COVARIANCE MATRIX AND EIGENVALUES
TO CALCULATE SPATIAL FEATURES:
- Linearity
- Planarity
- Sphericity
- Omnivariance
- Anisotropy
- Eigentropy
- Sum of eigenvalues
- Verticality
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eig
import matplotlib.pyplot as plt
import open3d as o3d
import random
import math

# Calculate covariance weights based on gaussian distance to median
def gaussian_distance_weights(pointcloud):
    # Calucalate median
    median = np.median(pointcloud, axis=0)

    # Calculate euclidian distance to median
    median_dis = np.sum((pointcloud - median) ** 2, axis=1) ** (1/2)
    
    # Standard deviation = 0.5 (empirical observation)
    std_dev = 0.5

    # Calculate gaussian distance
    gaussian_dis = (1 / (math.sqrt(std_dev) * math.sqrt(2 * math.pi))) ** ((median_dis ** 2) / 2 * std_dev)

    # Calculate normalized weights
    weights = gaussian_dis / np.linalg.norm(gaussian_dis)

    # Returns an integer between 0 and 100 as weight for each point
    return np.uint8(weights * 100)

# Calculate the eigenvalues and eigenvalues based on weighted covariance matrix
def eigenvalues_vectors(pointcloud, weights):
    # Transform data to origin
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pointcloud)

    """"
    Convariance matrix accurater te maken
    """

    cov_matrix = np.cov(scaled_data.T, fweights=weights)

    eigenvalues, eigenvectors = eig(cov_matrix)
    eigenvalues[::-1].sort()
    eigenvectors[::-1].sort()

    return eigenvalues, eigenvectors

# Features based on eigenvalues
def linearity(lambda1, lambda2, lambda3):
    return (lambda1 - lambda2) / lambda1

def planarity(lambda1, lambda2, lambda3):
    return (lambda2 - lambda3) / lambda1

def sphericity(lambda1, lambda2, lambda3):
    return lambda3 / lambda1

def omnivariance(lambda1, lambda2, lambda3):
    return (lambda1 * lambda2 * lambda3) ** (1/3)

def anisotropy(lambda1, lambda2, lambda3):
    return (lambda1 - lambda3) / lambda1

def eigentropy(lambda1, lambda2, lambda3):
    eigentropy = np.sum(lambda1 * np.log(lambda1) + lambda2 * np.log(lambda2) + lambda3 * np.log(lambda3))
    return eigentropy

def eigenvalue_sum(lambda1, lambda2, lambda3):
    return lambda1 + lambda2 + lambda3

def verticality(vector1):
    z = vector1[2]
    return 1 - z

# Find the k nearest neighbors of point in data
def neirest_neighborhood(data, p, treshhold):
    x_point, y_point, z_point = data[p]
    x, y, z = data.T

    dis = ((x - x_point)**2 + (y - y_point)**2 + (z - z_point)**2)**(1/2)

    idxs = np.where(dis < treshhold)

    return idxs

# Pick a random point and find neighbours
# then calculate the specified feature of that point with neighbours
def random_sampling(data, feature, k, treshhold, visualize=False):
    data = data.values

    if visualize:
        colors = np.zeros((len(data), 3))

    if k > len(data):
        k = len(data)

    results = []
    for _ in range(k):
        point = random.randint(0, len(data) - 1)
        idxs = neirest_neighborhood(data, point, treshhold)
        
        if visualize:
            r = random.uniform(0, 1)
            g = random.uniform(0, 1)
            b = random.uniform(0, 1)

            colors[idxs] = [r, g, b]

        if len(idxs[0]) > 1:
            neigh_data = data[idxs]
            weights = gaussian_distance_weights(neigh_data)

            values, vectors = eigenvalues_vectors(neigh_data, weights)
            lambda1, lambda2, lambda3 = values
            
            if feature == 'linearity':
                results.append(linearity(lambda1, lambda2, lambda3))
            elif feature == 'planarity':
                results.append(planarity(lambda1, lambda2, lambda3))
            elif feature == 'sphericity':
                results.append(sphericity(lambda1, lambda2, lambda3))
            elif feature == 'omnivariance':
                results.append(omnivariance(lambda1, lambda2, lambda3))
            elif feature == 'anisotropy':
                results.append(anisotropy(lambda1, lambda2, lambda3))
            elif feature == 'eigentropy':
                results.append(eigentropy(lambda1, lambda2, lambda3))
            elif feature == 'eigenvalue_sum':
                results.append(eigenvalue_sum(lambda1, lambda2, lambda3))
            elif feature == 'verticality':
                results.append(verticality(vectors[0]))

    if visualize:
        open3d_pointcloud = o3d.geometry.PointCloud()
        open3d_pointcloud.points = o3d.utility.Vector3dVector(data)
        open3d_pointcloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([open3d_pointcloud])

    avg_result = sum(results) / len(results)

    return avg_result

# Visualize a specific feature on the data
if __name__ == '__main__':
    directory = './code/data'

    results = []

    # 000 - 099: building;
    # 100 - 199: car;
    # 200 - 299: fence;
    # 300 - 399: pole;
    # 400 - 499: tree.

    x = 0
    y = 500
    feature = 'linearity'

    for i, file in enumerate(os.listdir(directory)[x:y]):
        filename = os.path.join(directory, file)
        data = pd.read_table(filename, skiprows=0, delim_whitespace=True,
                                        names=['x', 'y', 'z'])
    
        results.append(random_sampling(data, feature, 15, .5, False))

    frame = np.array([range(y-x), results])

    res = pd.DataFrame(frame.T, columns=['Object_idx', feature])
    
    colors = ['green', 'yellow', 'red', 'blue', 'orange', 'black']
    color_selection = []

    for i in range((y-x) // 100):
        color = colors[i]
        for i in range(100):
            color_selection.append(color)

    res.plot(kind='scatter', x='Object_idx', y=feature, color=color_selection)
    plt.show()