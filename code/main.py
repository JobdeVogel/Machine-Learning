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

import numpy as np

# SPECIFY IF PREPROCESSING REQUIRED
def process_data(manual=True):
    if manual:
        process = input('Would you like to preprocess data? Default value NO [y][n] ')
    else:
        process = 'y'

    if process == 'y':
        print('Preprocessing may take approximately 30 seconds.')
        data = preprocess('./code/data', FEATURES, AVAILABLE_FEATURES)
    else:
        data = pd.read_csv('./code/csv_data.csv')
    
    return data[FEATURES]

# Exhaustive method to find best feature
def featureScoring():
    sphericity = 0
    planarity = 0
    anisotropy = 0
    omnivariance = 0
    linearity = 0

    for j in range(50):
        data = process_data(manual=False)

        import warnings
        warnings.filterwarnings("ignore")

        best_features = []
        final_score = 0
        for i in range(200):
            temp_features = np.random.choice(FEATURES, 3, replace=False)
            temp_data = data[temp_features]

            targets = np.zeros(len(temp_data), dtype=np.int16)

            for i in range(len(temp_data)):
                targets[i] = int(i // 100)

            temp_data['labels'] = targets

            preselected_param = [{
                'kernel' : 'sigmoid', 
                'C' : 100,
                'gamma' : 'auto',
                'degree' : 2
                }]

            score = SVM.supportVector(temp_data, preselected_param, 0.3, to_print = False)

            if score > final_score:
                final_score = score
                best_features = temp_features
                print(temp_features)
                print(score)
                
        print(final_score)
        print(best_features)

        if 'linearity' in best_features:
            linearity +=1
        if 'planarity' in best_features:
            planarity +=1
        if 'sphericity' in best_features:
            sphericity +=1
        if 'anisotropy' in best_features:
            anisotropy +=1
        if 'omnivariance' in best_features:
            omnivariance +=1
    
        print('Intermidiate results:')
        print('linearity: ' + str(linearity))
        print('planarity: ' + str(planarity))
        print('sphericity: ' + str(sphericity))
        print('anisotropy: ' + str(anisotropy))
        print('omnivariance: ' + str(omnivariance))
        print('\n')
    
    return

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
        clusters = SVM.main(data, True, FEATURES)
        return
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
FEATURES = ['z_height', 'planarity', 'average_width']
COLORS = ['green', 'yellow', 'red', 'blue', 'orange', 'black', 'purple', 'pink', 'white', 'silver']

# HIERARCHICAL AND DENSITY DISTANCE SETTINGS
P_NORM = 1

# K-MEANS AND HIERARCHICAL SETTINGS
K = 5
TYPE = 'farthest' #Choose between 'nearest' 'average' 'farthest'

if __name__ == '__main__':
    set_options()
    
    #featureScoring()

    data = process_data()

    # Save data to csv
    # data.to_csv('code/csv_data.csv', index=False)
    # print('Data saved to csv_data.csv')

    main(data)