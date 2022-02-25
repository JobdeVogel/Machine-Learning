# Due date: 07 March, 2022 23:59
import numpy as np
import pandas as pd
# from preprocessing import preprocess
from clustering import hierarchical
from functions import color_plt
from options import set_options

"""
CHOOSE BETWEEN EXTRACTING DATA FROM CSV FILE
OR USE THE PREPROCESSING FUNCTION.
"""

#data = preprocess('./data')
data = pd.read_csv('csv_data.csv')

def main(data):
    FEATURES = ['z_height', 'convex_hull_areas', 'bounding_box_volumes']

    p_norm = 1
    k = 5
    type = 'nearest' #Choose between 'nearest' 'average' 'farthest'

    """
    CHOOSE BETWEEN
    - K-means clustering
    - Hierarchical clustering
    - DBSCAN
    """

    hierarchical_clusters = hierarchical.main(p_norm, k, type, data)
    
    color_selection = []
    colors = ['yellow', 'green', 'red', 'blue', 'orange']
    
    for color in hierarchical_clusters:
        color_selection.append(colors[color])

    color_plt(data, color_selection, *FEATURES)

main(data)