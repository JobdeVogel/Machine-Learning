"""
MINIMUM BOUNDING BOX POINT CLOUD

DEVELOPED BY:
Job de Vogel
"""

import open3d as o3d

"""
CALCULATE THE SMALLEST POSSIBLE BOUDING BOX
AROUND THE POINTCLOUD USING THE OPEN3D PACKAGE
"""

def pointcloud_bounding_box(dataframe):
    pointcloud = dataframe.values

    open3d_pointcloud = o3d.geometry.PointCloud()
    open3d_pointcloud.points = o3d.utility.Vector3dVector(pointcloud)

    oriented_bouding_box = open3d_pointcloud.get_oriented_bounding_box()

    """
    VISUALIZE THE POINTCLOUD IN 3D

    oriented_bouding_box.color = (1, 0, 0)
    o3d.visualization.draw_geometries([open3d_pointcloud, oriented_bouding_box])
    """

    return oriented_bouding_box.volume()