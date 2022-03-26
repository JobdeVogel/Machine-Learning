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
from clustering import kmeans, hierarchical, dbscan, SVM, RF
from functions import color_plt
from options import set_options
from evaluation import evaluation

# SPECIFY IF PREPROCESSING REQUIRED
def process_data():
    process = input('Would you like to preprocess data? Default value NO [y][n] ')
    if process == 'y':
        print('Preprocessing may take approximately 30 seconds.')
        data = preprocess('./code/data', FEATURES, AVAILABLE_FEATURES)
    else:
        data = pd.read_csv('./code/csv_data.csv')
    
    return data

def main(data):
    # SELECT CLUSTER TYPE
    print('Available cluster algorithms \'kmeans\' \'hierarchical\' \'dbscan\', \'SVM\', \'RF\'')
    cluster_type = input('Please select cluster type (default type Random Forest): ')

    if cluster_type == 'kmeans':
        clusters = kmeans.main(K, data)
    elif cluster_type == 'hierarchical':
        clusters = hierarchical.main(P_NORM, K, TYPE, data)
    elif cluster_type == 'dbscan':
        clusters = dbscan.main(P_NORM, data)
    elif cluster_type == 'SVM':
        clusters = SVM.main(data)
    elif cluster_type == 'RF' or cluster_type == '':
        RF.main(data)
        return
    else:
        print('This cluster algorithm is not available, please re-run and choose between kmeans, hierarchical, dbscan, SVM and RF.')
        return

    # EVALUATE FOR CLUSTER LABEL AND ACCURACY
    color_selection = evaluation(clusters, COLORS)

    # PLOT THE RESULT
    color_plt(data, color_selection, *FEATURES)

    return

AVAILABLE_FEATURES = ['z_height', 'shape_ratios', 'convex_hull_areas', 'bounding_box_volumes', 'linearity', 'planarity', 'sphericity', 'anisotropy', 'eigentropy', 'omnivariance', 'eigenvalue_sum', 'varticality', 'average_width']

# Please select features to preprocess, use same order as AVAILABLE_FEATURES
FEATURES = ['z_height', 'shape_ratios', 'convex_hull_areas', 'bounding_box_volumes', 'linearity', 'planarity', 'sphericity', 'anisotropy', 'eigentropy', 'omnivariance', 'eigenvalue_sum', 'varticality', 'average_width']
COLORS = ['green', 'yellow', 'red', 'blue', 'orange', 'black', 'purple']

# HIERARCHICAL AND DENSITY DISTANCE SETTINGS
P_NORM = 1

# K-MEANS AND HIERARCHICAL SETTINGS
K = 5
TYPE = 'farthest' #Choose between 'nearest' 'average' 'farthest'

if __name__ == '__main__':
    set_options()

    data = process_data()
    
    # Save data to csv
    data.to_csv('code/csv_data.csv', index=False)
    print('Data saved to csv_data.csv')

    main(data)


    
