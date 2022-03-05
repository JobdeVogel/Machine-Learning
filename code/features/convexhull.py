"""
CONVEX HULL AREA POINT CLOUD

DEVELOPED BY:
Jirri van den Bos
"""

from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint
import numpy as np

"""
This feature calculates the projected convex hull area
Using the precision factor prec_factor, the amount of
points in the cloud is reduced to make the algorithm less
computationally expensive.
"""

def getConvexHullArea(dataframe, prec_factor):
    pointcloud_np_array = dataframe.values

    # Reduce the cloud size with a precision_factor
    cloud_size = len(pointcloud_np_array)
    random_indices = np.random.choice(cloud_size, size=int(prec_factor * cloud_size), replace=False)
    reduced_pointcloud = pointcloud_np_array[random_indices]

    # If not enough points available, return area = 0
    if len(reduced_pointcloud) < 4:
        print('Not enough points available to calculate area convex Hull')
        return 0

    # Calculate the convex hull
    convex_hull = ConvexHull(reduced_pointcloud)
    return MultiPoint(pointcloud_np_array).convex_hull.area