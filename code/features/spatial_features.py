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
"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eig
import matplotlib.pyplot as plt

# https://vitalflux.com/eigenvalues-eigenvectors-python-examples/

# filename = os.path.join('./code/data', '003.xyz')
# data = pd.read_table(filename, skiprows=0, delim_whitespace=True, names=['x', 'y', 'z'])

def eigenvalues_vectors(pointcloud):
    # Transform data to origin
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pointcloud)

    cov_matrix = np.cov(scaled_data.T)

    eigenvalues, eigenvectors = eig(cov_matrix)
    eigenvalues[::-1].sort()
    eigenvectors[::-1].sort()

    return eigenvalues, eigenvectors

def linearity(lambda1, lambda2):
    return (lambda1 - lambda2) / lambda1

def planarity(lambda1, lambda2, lambda3):
    return (lambda2 - lambda3) / lambda1

def sphericity(lambda1, lambda3):
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

# Since we are calculating object-based and not point-based, it makes more sense
# to use the amount of points in cloud
def local_point_density(pointcloud):
    return len(pointcloud)

if __name__ == '__main__':
    directory = './code/data'
   
    linearities = []
    sphericalities = []

    # 000 - 099: building;
    # 100 - 199: car;
    # 200 - 299: fence;
    # 300 - 399: pole;
    # 400 - 499: tree.

    x = 300
    y = 400

    for i, filename in enumerate(os.listdir(directory)[x:y]):
        filename = os.path.join(directory, filename)
        data = pd.read_table(filename, skiprows=0, delim_whitespace=True,
                                names=['x', 'y', 'z'])

        values, vectors = eigenvalues_vectors(data)
        
        lambda1, lambda2, lambda3 = values
        vector1, vector2, vector3 = vectors

        linearities.append(anisotropy(lambda1, lambda2, lambda3))
        sphericalities.append(eigentropy(lambda1, lambda2, lambda3))
        

    features = np.array([linearities, sphericalities]).T
    df = pd.DataFrame(features, columns=['anisotropy', 'eigentropy'])
    
    colors = ['green', 'yellow', 'red', 'blue', 'orange', 'black']
    color_selection = []

    for i in range((y-x)//100):
        color = colors[i]
        for i in range(100):
            color_selection.append(color)
    
    df.plot(kind='scatter', x='anisotropy', y='eigentropy', color=color_selection)
    plt.show()