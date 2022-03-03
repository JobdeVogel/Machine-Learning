# Due date: 07 March, 2022 23:59
import numpy as np
import pandas as pd
from preprocessing import preprocess
from clustering import kmeans, hierarchical, dbscan
from functions import color_plt
from options import set_options
from evaluation import evaluation

"""
# Best result currently with p_norm = 1 and type = 'farthest'
# Perhaps more trustworthy with p_norm = 4 and type = 'average'
# Preprocessing settings: 'z_height', 'convex_hull_areas', 'bounding_box_volumes'
# Clipping values: 15, 2, 100, 3000
"""

def info():
    pass

def main(data):
    FEATURES = ['z_height', 'convex_hull_areas', 'bounding_box_volumes']

    print('Available cluster algorithms \'kmeans\' \'hierarchical\' \'dbscan\'')
    cluster_type = input('Please select cluster type: ')
    print('\n')

    if cluster_type == 'kmeans':
        return
    elif cluster_type == 'hierarchical':
        clusters = hierarchical.main(P_NORM, K, TYPE, data)
    elif cluster_type == 'dbscan':
        clusters = dbscan.main(P_NORM, data)
    else:
        print('This cluster algorithm is not available, please re-run and choose between kmeans, hierarchical and dbscan.')
        return

    cluster_labels = evaluation(clusters)
    
    color_selection = []

    for cluster_idx in clusters:
        idx = cluster_labels[cluster_idx]
        try:
            color_selection.append(COLORS[idx])
        except:
            print('\nWARNING: Not enough colors available for all clusters, please add more colors to COLORS container to visualize plot.')
            return

    color_plt(data, color_selection, *FEATURES)

    return

# GENERAL SETTINGS
FEATURES = ['z_height', 'convex_hull_areas', 'bounding_box_volumes']
COLORS = ['green', 'red', 'blue', 'yellow', 'orange', 'black']

# DISTANCE SETTINGS
P_NORM = 1

# K-MEANS AND HIERARCHICAL SETTINGS
K = 5
TYPE = 'farthest' #Choose between 'nearest' 'average' 'farthest'

print('Preprocessing may take approximately 30 seconds.')
#data = preprocess('./data')
data = pd.read_csv('csv_data.csv')

main(data)