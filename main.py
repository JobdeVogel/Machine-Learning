# Due date: 07 March, 2022 23:59
"""
THIS IS THE MAIN SCRIPT FOR CLUSTERING ALGORITHMS.
RUN AND FOLLOW PROMPT TO VIEW RESULTS.

AVAILABLE OPTIONS FOR ALGORITHMS FROM LINE 57

DEVELOPED BY: 
Job de Vogel
Jirri van den Bos
"""

print('Loading packages...')
import pandas as pd
from preprocessing import preprocess
from clustering import kmeans, hierarchical, dbscan
from functions import color_plt
from options import set_options
from evaluation import evaluation

# SPECIFY IF PREPROCESSING REQUIRED
def process_data():
    process = input('Would you like to preprocess data? Default value NO [y][n] ')
    if process == 'y':
        print('Preprocessing may take approximately 30 seconds.')
        data = preprocess('./data')
    else:
        data = pd.read_csv('csv_data.csv')
    
    return data

def main(data):
    # SELECT CLUSTER TYPE
    print('Available cluster algorithms \'kmeans\' \'hierarchical\' \'dbscan\'')
    cluster_type = input('Please select cluster type: ')

    if cluster_type == 'kmeans':
        clusters = kmeans.main(K, data)
    elif cluster_type == 'hierarchical':
        clusters = hierarchical.main(P_NORM, K, TYPE, data)
    elif cluster_type == 'dbscan':
        clusters = dbscan.main(P_NORM, data)
    else:
        print('This cluster algorithm is not available, please re-run and choose between kmeans, hierarchical and dbscan.')
        return

    # EVALUATE FOR CLUSTER LABEL AND ACCURACY
    color_selection = evaluation(clusters, COLORS)

    # PLOT THE RESULT
    color_plt(data, color_selection, *FEATURES)

    return

# GENERAL SETTINGS
FEATURES = ['z_height', 'convex_hull_areas', 'bounding_box_volumes']
COLORS = ['green', 'yellow', 'red', 'blue', 'orange', 'black']

# DISTANCE SETTINGS
P_NORM = 1

# K-MEANS AND HIERARCHICAL SETTINGS
K = 5
TYPE = 'farthest' #Choose between 'nearest' 'average' 'farthest'

set_options()
data = process_data()
main(data)