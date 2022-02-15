import numpy as np

pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])

x_range, y_range, z_range = np.max(pts, 0) - np.min(pts, 0)

rotation_matrix = np.array([])

# https://stackoverflow.com/questions/12148351/efficiently-rotate-a-set-of-points-with-a-rotation-matrix-in-numpy
rotated_pointcloud = np.dot(pts, rotation_matrix.T)

"""
Step 1: calculate the x_range, y_range, z_range
Step 2: calculate the volume
Step 3: rotate the pointcloud by n degrees
Step 4: check if volume is lower
Step 5: if that is the case, overwrite volume
"""