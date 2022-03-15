import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn

# Add the labels to the proximity matrix dataframe
def add_labels(df):
    targets = np.zeros(len(df), dtype=np.int16)

    for i in range(len(df)):
        targets[i] = int(i // 100)

    df['labels'] = targets

    return df

# Calculate accuracy with random forest classifier
def random_forest(data, n_estimators=100, criterion='gini', max_features='auto', bootstrap=True, max_samples=None, test_size=0.3, to_print=False):
    x = data.loc[:, data.columns != 'labels']
    y = data['labels']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    clf=RFC(n_estimators=n_estimators, criterion=criterion, max_features=max_features, bootstrap=bootstrap, max_samples=max_samples)
    clf.fit(x_train, y_train)

    y_pred=clf.predict(x_test)

    # Overall accuracy
    oA = metrics.accuracy_score(y_test, y_pred)

    # Mean per class accuracy
    mcA = metrics.balanced_accuracy_score(y_test, y_pred)

    # Confusion Matrix
    cM = metrics.confusion_matrix(y_test, y_pred)

    if to_print:
        print('Overall accuracy acore: {}'.format(round(oA, 2)))
        print('Mean per class accuracy: {}'.format(round(mcA, 2)))

        # Plot confusion matrix with seaborn
        plt.figure(figsize=(5,5))
        plt.grid(False)
        sn.heatmap(cM, annot = True, cmap="Greens")
        plt.xlabel('Predicted label')
        plt.ylabel('Real label')
        plt.title('Confusion Matrix (Test Size {}%)'.format(test_size * 100))
        plt.show()

    return oA

# Plot the learning curve, based on certain parameter of Random Forest classifier
def learning_curve(data, n_estimators):
    train_sizes = []
    accuracies = []

    for i in range(1, 100):

        test_size = i / 100

        score = random_forest(data, test_size = test_size, n_estimators=n_estimators)

        train_sizes.append(100 - (test_size * 100))
        accuracies.append(score * 100)

        print('Finished learning curve {}%, n_estimators {}'.format(round(((i - 1) / 98) * 100), n_estimators), end='\r')

    plt.plot(train_sizes, accuracies, label="n = {}".format(n_estimators))
    print('')

def main(data):
    data = add_labels(data)
    
    # Execute for different training sizes and n_estimators
    n_estimators = [10]

    for n in n_estimators:
        learning_curve(data, n)

    plt.xlabel('Training size (%)')
    plt.ylabel('Overall Accuracy (%)')
    plt.title('Learning Curve')
    plt.grid()
    plt.legend()
    plt.show()

    # Execute only once for 30% test data
    random_forest(data, test_size=0.5, to_print=True)

    return

if __name__ == "__main__":
    # Set the seed for classifier
    np.random.seed(1) 

    # Use csv data
    data = pd.read_csv('./proximity_matrix.csv')
    
    main(data)
