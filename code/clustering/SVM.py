from asyncio import constants
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from seaborn import heatmap

#Add labels to dataset
def addLabels(data):
    targets = np.zeros(len(data), dtype=np.int16)

    for i in range(len(data)):
        targets[i] = int(i // 100)
    
    data['labels'] = targets

    return data

#Find the best hyperparameters using GridSearchCV
def findParameters(param_dict, check):
    results = []
    x_train, x_test, y_train, y_test = train_test_split(data.drop('labels', axis=1), data['labels'], test_size=0.5)
    iterate = param_dict[check]

    #Saving only the found parameters to use in a seperately built model
    for i in iterate:
        param_dict[check] = [i]
        results.append(GridSearchCV(SVC(), param_dict, refit=True, verbose=1).fit(x_train, y_train).best_params_)

    return results

#Main function for svm classification
def supportVector(data, param, test_size, to_print = False):
    x_train, x_test, y_train, y_test = train_test_split(data.drop('labels', axis=1), data['labels'], test_size=test_size)
    clf = SVC(**param)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Overall accuracy
    oA = accuracy_score(y_test, y_pred)

    # Mean per class accuracy
    mcA =balanced_accuracy_score(y_test, y_pred)

    # Confusion Matrix
    cM = confusion_matrix(y_test, y_pred)

    #Output confusion matrix and accuracy reading
    if to_print:
        print('Overall accuracy score: {}'.format(round(oA, 2)))
        print('Mean per class accuracy: {}'.format(round(mcA, 2)))

        # Plot confusion matrix with seaborn
        plt.figure(figsize=(5,5))
        plt.grid(False)
        heatmap(cM, annot = True, cmap="Greens")
        plt.xlabel('Predicted label')
        plt.ylabel('Real label')
        plt.title('Confusion Matrix (Test Size {}%)'.format(test_size * 100))
        plt.show()

    return oA

#Establishing a learning curve over test/train sizes
def learningCurve(data, param):
    train_sizes = []
    accuracies = []

    #Running for all test/train splits
    for i in range(1, 100):
        test_size = i / 100
        score = supportVector(data, param, test_size)

        train_sizes.append(100 - (test_size * 100))
        accuracies.append(score)

    return train_sizes, accuracies

#Main Loop
def main(data, param_check):
    data = addLabels(data)
    constant = []
    #Preselected empirical best hyperparameters for this dataset
    preselected_param = [{
        'kernel' : 'rbf', 
        'C' : 10,
        'gamma' : 'scale',
        'degree' : 2
        }]
    
    #The parameter to check and display in graph
    check = 'kernel'
    
    #Possibility to check, keep constant and optimize parameters
    if param_check == True:
        #Full parameter dictionary
        param_dict = {
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
        'C' : [0.1, 1, 10, 100],
        'gamma' : ['scale', 'auto'],
        'degree' : [2, 3, 4, 5, 7, 9]
        }

        #Providing a constant value will override the full param_dict
        constant = [
            #('kernel', 'rbf'),
            #('C', 1),
            #('gamma', 'scale'),
            #('degree', 2)
            ]

        #Loop to override      
        for i in constant:
            param_dict[i[0]] = [i[1]]
        labels = param_dict[check]

        #Finding the best params to use for the different values to check
        params = findParameters(param_dict, check)
        print(params)

    #Select the preselected params by setting param_check to False 
    elif param_check == False:
        params = preselected_param
        labels = [preselected_param[0]['kernel']]
    
    #Catching param_check != Boolean value
    else:
        print('Please provide a boolean value for param_check')

    #For the provided param sets run the learning curve
    i = 0
    for param in params:
        train_sizes, accuracies = learningCurve(data, param)
        plt.plot(train_sizes, accuracies, label=check + "= {}".format(labels[i]))
        i += 1
    
    #Visualize the params to check
    plt.xlabel('Training size (%)')
    plt.ylabel('Overall Accuracy (%)')
    plt.title('Learning Curve')
    plt.figtext(0.15, 0.05, 'Constant:' + str(constant))
    plt.grid()
    plt.legend()
    plt.show()

    #Running at 30% Test size for confusion matrix and accuracy reading
    #Be warned this will always run for the preselected params
    supportVector(data, preselected_param[0], test_size = 0.3, to_print=True)

if __name__ == "__main__":
    np.random.seed(2)

    data = pd.read_csv('./code/csv_data.csv')
    param_check = False
    main(data, param_check)