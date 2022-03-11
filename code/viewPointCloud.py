import os
import pandas as pd
import open3d as o3d

"""
VIEW POINTCLOUD

DEVELOPED BY:
Job de Vogel
"""

""""
THIS SCRIPT VISUALIZES A SPECIFIC POINT CLOUD
"""

filename = os.path.join('./code/data', '007.xyz')

"""
0-99: Houses
100-199: Cars
200-299: Fences
300-399: Traffic Lights
400-499: Trees
"""

data = pd.read_table(filename, skiprows=0, delim_whitespace=True, names=['x', 'y', 'z'])
print((data['y'].max(0) - data['y'].min(0)) / (data['x'].max(0) - data['x'].min(0)))
print(data['z'].max(0))
print(data['z'].min(0))


data = data.values

open3d_pointcloud = o3d.geometry.PointCloud()
open3d_pointcloud.points = o3d.utility.Vector3dVector(data)
o3d.visualization.draw_geometries([open3d_pointcloud])