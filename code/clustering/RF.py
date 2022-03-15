import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv('./proximity_matrix.csv', skiprows=0)

def add_labels(df):
    targets = np.zeros(len(df), dtype=np.int16)

    for i in range(len(df)):
        targets[i] = int(i // 100)

    df['labels'] = targets

    return df

def random_forest(data, n_estimators=100, criterion='gini', max_features='auto', bootstrap=True, max_samples=None, test_size=0.3):
    x = data.loc[:, data.columns != 'labels']
    y = data['labels']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    clf=RFC(n_estimators=n_estimators, criterion=criterion, max_features=max_features, bootstrap=bootstrap, max_samples=max_samples)
    clf.fit(x_train, y_train)

    y_pred=clf.predict(x_test)

    # Overall accuracy
    oA = metrics.accuracy_score(y_test, y_pred)
    # print('Overall accuracy acore: {}'.format(oA))

    # Mean per class accuracy
    mcA = metrics.accuracy_score(y_test, y_pred)
    # print('Mean per class accuracy: {}'.format(mcA))

    # Confusion Matrix
    cM = metrics.confusion_matrix(y_test, y_pred)
    # print(cM)

    return oA

def learning_curve(data):
    test_sizes = []
    accuracies = []

    for i in range(1, 100):
        test_size = i / 100
        score = random_forest(data, test_size = test_size)

        test_sizes.append(test_size * 100)
        accuracies.append(score * 100)

        print('Finished learning curve {}%'.format(round(((i - 1) / 98) * 100), 2), end='\r')

    plt.plot(test_sizes, accuracies)

    plt.xlabel('Test size (%)')
    plt.ylabel('Overall Accuracy (%)')

    plt.title('Learning Curve')
    plt.show()

data = add_labels(data)
learning_curve(data)
