"""
EVALUATION

DEVELOPED BY:
Job de Vogel
"""

"""
CALCULATE THE HIGHEST ACCURACY BASED ON ALL
POSSIBLE CLUSTER LABEL ASSIGNMENT VARIATIONS
"""

import numpy as np
import itertools

# Evaluate the result for accuracy and assign labels to the group
def evaluation(predicted_labels, colors):
    print('Predicting labels of clusters...')
    k = len(np.unique(predicted_labels))

    available_labels = np.arange(k)
    observed_options = list(itertools.permutations(available_labels))

    highest_percentage = 0
    cluster_labels = []

    # For each possible cluster label order, calculate the accuracy
    for observed in observed_options:
        observed = np.array(list(observed))
        observed_labels = np.repeat(observed, 100)

        correct = 0

        # If point correctly labeled add one to correct
        for obs_label, pred_label in zip(observed_labels, predicted_labels):
            if obs_label == pred_label:
                correct += 1

        # Calculate the accuracy
        accuracy = (correct/500) * 100

        # If the accuracy is higer with this label order
        # Set the new label order
        if accuracy > highest_percentage:
            highest_percentage = round(accuracy, 3)
            cluster_labels = observed
    
    print('Optimized cluster label assignment: ' + str(cluster_labels))
    print(str(highest_percentage) + '% accuracy')

    # Unoptimized color assignment
    # If large amount of clusters, do not use color assignment!
    color_selection = []

    for idx in predicted_labels:
        for i in range(k):
            if idx == cluster_labels[i]:
                try:
                    color_selection.append(colors[i])
                except: 
                    print('WARNING: NOT ENOUGH COLORS AVAILABLE, ADD COLORS IN main.py line 57')
                    return

    return color_selection
