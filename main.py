# Due date: 07 March, 2022 23:59
import numpy as np
import pandas as pd
from preprocessing import preprocess
from clustering import hierarchical
from functions import color_plt
from options import set_options
from evaluation import evaluation

"""
CHOOSE BETWEEN EXTRACTING DATA FROM CSV FILE
OR USE THE PREPROCESSING FUNCTION.
"""

data = preprocess('./data')
#data = pd.read_csv('csv_data.csv')

def main(data):
    FEATURES = ['z_height', 'convex_hull_areas', 'bounding_box_volumes']

    p_norm = 1
    k = 5
    type = 'farthest' #Choose between 'nearest' 'average' 'farthest'

    print("features: " + str(FEATURES))
    print("p_norm: " + str(p_norm))
    print("type: " + str(type))

    """
    CHOOSE BETWEEN
    - K-means clustering
    - Hierarchical clustering
    - DBSCAN
    """

    hierarchical_clusters = hierarchical.main(p_norm, k, type, data)
    evaluation(hierarchical_clusters)
    
    color_selection = []
    colors = ['yellow', 'green', 'red', 'blue', 'orange']
    
    for color in hierarchical_clusters:
        color_selection.append(colors[color])

    color_plt(data, color_selection, *FEATURES)

main(data)
# Best result currently with p_norm = 1 and type = 'farthest'
# Perhaps more trustworthy with p_norm = 4 and type = 'average'
# Preprocessing settings: 'z_height', 'convex_hull_areas', 'bounding_box_volumes'
# Clipping values: 15, 2, 100, 3000
