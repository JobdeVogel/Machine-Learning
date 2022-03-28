import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from seaborn import heatmap

#Add labels to dataset
def addLabels(data):
    targets = np.zeros(len(data), dtype=np.int16)

    for i in range(len(data)):
        targets[i] = int(i // 100)
    
    data['labels'] = targets

    return data

#Find the best hyperparameters using GridSearchCV
def findParameters(data, param_dict, check, test_size):
    results = []
    x_train, x_test, y_train, y_test = train_test_split(data.drop('labels', axis=1), data['labels'], test_size=test_size)
    iterate = param_dict[check]

    #Saving only the found parameters to use in a seperately built model
    for i in iterate:
        param_dict[check] = [i]
        results.append(GridSearchCV(SVC(), param_dict, refit=True, verbose=1).fit(x_train, y_train).best_params_)

    return results

#Main function for svm classification
def supportVector(data, param, test_size, to_print = False):
    x_train, x_test, y_train, y_test = train_test_split(data.drop('labels', axis=1), data['labels'], test_size=test_size)
    clf = SVC(kernel='sigmoid',C=100,gamma='auto', degree=2)
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

def visualizeRBF2D(data, labels, features):
    param = {'kernel' : 'rbf', 'C' : 10, 'gamma' : 'scale'}
    data = data.loc[data['labels'].isin(labels)]
    x = data.iloc[:, [0,1]].to_numpy()
    y = data['labels'].to_numpy()
    
    clf = SVC(**param)
    clf.fit(x, y)
    ndict = {0: 'building', 1: 'car', 2: 'fence', 3: 'pole', 4: 'tree'}
    cdict = {0:'green', 1:'yellow', 2:'red', 3:'blue', 4:'orange'}
    for g in np.unique(y):
        ix = np.where(y == g)
        x0 = x[ix]
        c = y[ix]
        plt.scatter(x0[:, 0], x0[:, 1], c=cdict[g], s=30, label=ndict[g], cmap=plt.cm.Paired)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    yy_mesh, xx_mesh = np.meshgrid(yy, xx)
    xy = np.vstack([xx_mesh.ravel(), yy_mesh.ravel()]).T
    z = clf.decision_function(xy).reshape(xx_mesh.shape)
    
    ax.contour(xx_mesh, yy_mesh, z, colors='k', levels=[-1, 0, 1], alpha = 0.5, linestyles=['--','-','--'])
    ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
    ax.legend()
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.show()

def visualizeLinear3D(data, labels, features):
    param = {'kernel' : 'linear', 'C' : 100}
    data = data.loc[data['labels'].isin(labels)]
    x = data.iloc[:, [0,1,2]].to_numpy()
    y = data['labels'].to_numpy()
    ndict = {0: 'building', 1: 'car', 2: 'fence', 3: 'pole', 4: 'tree'}
    cdict = {0:'green', 1:'yellow', 2:'red', 3:'blue', 4:'orange'}


    clf = SVC(**param)
    clf.fit(x, y)

    z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x-clf.coef_[0][1]*y) / clf.coef_[0][2]

    tmp = np.linspace(0,1,5)
    x_mesh, y_mesh = np.meshgrid(tmp, tmp)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_mesh, y_mesh, z(x_mesh,y_mesh))
    ax.plot3D(x[y==0,0], x[y==0,1], x[y==0,2], 'ob', label=ndict[y[0]])
    ax.plot3D(x[y==1,0], x[y==1,1], x[y==1,2], 'sr', label=ndict[y[-1]])
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    ax.legend()
    plt.show()

#Main Loop
def main(data, param_check, features):
    data = addLabels(data)
    constant = []
    #Preselected empirical best hyperparameters for this dataset
    preselected_test_size = 0.3
    preselected_param = [{
        'kernel' : 'sigmoid', 
        'C' : 100,
        'gamma' : 'auto',
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
        'degree' : [2, 3, 4, 5, 7]
        }

        #Providing a constant value will override the full param_dict
        constant = [
            #('kernel', 'poly'),
            #('C', 1),
            #('gamma', 'scale'),
            #('degree', 2)
            ]

        #Loop to override      
        for i in constant:
            param_dict[i[0]] = [i[1]]
        labels = param_dict[check]

        #Finding the best params to use for the different values to check
        params = findParameters(data, param_dict, check, preselected_test_size)
        print(params)

    #Select the preselected params by setting param_check to False 
    elif param_check == False:
        params = preselected_param
        labels = [preselected_param[0][check]]
    
    #Catching param_check != Boolean value
    else:
        print('Please provide a boolean value for param_check')

    #For the provided param sets run the learning curve
    i = 0
    names = []
    areas = []
    for param in params:
        train_sizes, accuracies = learningCurve(data, param)
        area = np.trapz(accuracies)
        areas.append(area)
        names.append(param[check])     
        plt.plot(train_sizes, accuracies, label=check + "= {}".format(labels[i]))
        i += 1
    if param_check == True:
        areas, names = zip(*sorted(zip(areas, names)))   
        print('Best ' + check + ': ' + str(names[-1]) + ' at accuracy: ' + str(round(areas[-1], 4)) + '% (Based on area under graph)')
    
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
    supportVector(data, preselected_param[0], test_size = preselected_test_size, to_print=True)
    
    #Visualizing RBF kernel in 2D and Linear kernel in 3D. Select two labels to visualize.
    #Please Note first two-three FEATURES are selected for 2D/3D respectively
    # 0: building;
    # 1: car;
    # 2: fence;
    # 3: pole;
    # 4: tree.
    labels = (0, 1)
    visualizeRBF2D(data, labels, features)
    visualizeLinear3D(data, labels, features)


if __name__ == "__main__":
    np.random.seed(2)

    FEATURES = ['planarity', 'z_height', 'average_width']
    data = pd.read_csv('./code/csv_data.csv')
    data = data[FEATURES]
    param_check = False
    main(data, param_check)