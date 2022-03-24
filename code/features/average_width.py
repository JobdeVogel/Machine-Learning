import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eig
import math

# Calculate the eigenvalues and eigenvalues based on weighted covariance matrix
def eigenvalues_vectors(pointcloud):
    # Transform data to origin
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pointcloud)

    cov_matrix = np.cov(scaled_data.T)

    eigenvalues, eigenvectors = eig(cov_matrix)
    eigenvalues[::-1].sort()
    eigenvectors[::-1].sort()

    return eigenvalues, eigenvectors

def average_width(data, k_split):
    min = np.min(data['z'])
    height_diff = np.max(data['z']) - min

    norm = height_diff / k_split

    data = data.values

    results = np.array([])

    z_values = data[:, 2]

    for i in range(k_split):
        temp_data = data[(z_values < min + (i+1) * norm)  & (z_values > min + i * norm)]

        if len(temp_data) > 2:
            # Extract largest eigenvalue without weights
            res = eigenvalues_vectors(temp_data)[0][0]

            results = np.append(results, res)
    
    if len(results) == 0:
        print('k_split is too low to calculate average')
        
        return

    # Return average
    return np.mean(results)

# Visualize a specific feature on the data
if __name__ == '__main__':
    directory = './code/data'

    # 000 - 099: building;
    # 100 - 199: car;
    # 200 - 299: fence;
    # 300 - 399: pole;
    # 400 - 499: tree.

    x = 0
    y = 500
    feature = 'average_width'

    results = []
    for i, file in enumerate(os.listdir(directory)[x:y]):
        filename = os.path.join(directory, file)
        data = pd.read_table(filename, skiprows=0, delim_whitespace=True,
                                        names=['x', 'y', 'z'])
    
        results.append(average_width(data, 40))

    results = np.clip(np.array(results), 0, 40)
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