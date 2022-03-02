import numpy as np
import itertools

def evaluation(predicted_labels):
    k = len(np.unique(predicted_labels))

    available_labels = np.arange(k)
    observed_options = list(itertools.permutations(available_labels))

    highest_percentage = 0
    cluster_labels = []

    for observed in observed_options:
        observed = np.array(list(observed))
        observed_labels = np.repeat(observed, 100)

        correct = 0

        for obs_label, pred_label in zip(observed_labels, predicted_labels):
            if obs_label == pred_label:
                correct += 1

        accuracy = (correct/500) * 100

        if accuracy > highest_percentage:
            highest_percentage = round(accuracy, 3)
            cluster_labels = observed
    
    print('Optimized cluster labels: ' + str(cluster_labels))
    print(str(highest_percentage) + '% accuracy')
    
    return cluster_labels
