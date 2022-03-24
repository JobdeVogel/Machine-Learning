import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add the labels to the proximity matrix dataframe
def add_labels(df):
    targets = np.zeros(len(df), dtype=np.int16)

    for i in range(len(df)):
        targets[i] = int(i // 100)

    df['labels'] = targets

    return df

#Main Support vector prediction, returning the accuracy score
def support_vector(data, test_size, grid, random_state):
    x_train, x_test, y_train, y_test = splitData(data, test_size, random_state)
    y_pred = grid.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    return score

#Using a learning curve to find the optimal kernel to use for the supplied dataset
def learning_curve(data, kernel, cycles, random_state):
    train_sizes = []
    accuracies = []
    
    #Finding the best parameters for the current kernel
    grid = findCV(data, kernel, random_state)
    print(grid.best_estimator_)

    #Limiting the split analyzed to 0-80% greatly reduces noise
    for i in range(20, 100):
        test_size = []
        scores = []
        
        #Running the full script for cycles to normalize random fluctuations
        for j in range(0, cycles):
            test_size = i / 100
            score = support_vector(data, test_size, grid, random_state)
            scores.append(score)
        
        #Averaging the scores    
        score = sum(scores) / len(scores)

        #Add the results to a list
        train_sizes.append(100 - (test_size * 100))
        accuracies.append(score * 100)

    return train_sizes, accuracies


#Splitting the data using train_test_split 
def splitData(data, test_size, random_state):
    x = data.drop('labels', axis=1)
    y = data['labels']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test

#Using GridsearchCV the parameters are optimized using a standard 50,50 
def findCV(data, kernel, random_state):
    #Using a 50/50 split on the data to find the best parameters instead of running it for every split, massively reducing computation 
    x_train, x_test, y_train, y_test = splitData(data, 0.5, random_state)

    #Specify parameters to check for C, Gamma and degree optimization
    #Add a degree check only to polynomial
    if kernel == 'poly':
        input_params = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'degree': [2, 3, 4, 5, 7, 9], 'kernel': [kernel]}
    else:
        input_params = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001], 'kernel': [kernel]}
    
    #Return the grid for prediction
    grid = GridSearchCV(SVC(random_state=random_state), input_params, refit=True, verbose=1)
    grid.fit(x_train, y_train)
    return grid

#Check for most and least accurate data labeling
def CheckLabelAcc(data):
    data = add_labels(data)
    kernel = 'rbf'
    test_size = 0.5
    random_state = None
    names = ['building', 'car', 'fence', 'pole', 'tree']
    # 000 - 099: building;
    # 100 - 199: car;
    # 200 - 299: fence;
    # 300 - 399: pole;
    # 400 - 499: tree.
    tick_labels = []
    scores = []
    freq = []
    width = 0.1
    x = 0
    for i in range(0,5):
        print (i)
        nameA = names[i]
        for j in range(0, 5):
            if j > i:
                x += 0.2
                nameB = names[j]
                select_data = data.loc[data['labels'].isin([i, j])]
                grid = findCV(select_data, kernel, random_state)
                score = support_vector(select_data, test_size, grid, random_state)
                print (score)
                tick_labels.append(nameA + '/' + nameB)
                scores.append(score)
                freq.append(x)

    fig, ax = plt.subplots()
    rects = ax.bar(freq, scores, width, color='#fcba03')
    ax.set_ylim(0,1.1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Specific Relation Accuracy')
    ax.set_xticks(np.add(freq,width/16)) # set the position of the x ticks
    ax.set_xticklabels(tick_labels)

    i = 0
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                scores[i],
                ha='center', va='bottom')
        i+=1

    plt.show()

#Main callable script loop
def main(data):
    data = add_labels(data)

    #List of Kernels to try
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    #Amount of cycles to average to reduce random fluctuations
    cycles = 10

    #Possibility to specify a seed to eliminate random chance (None or any int)
    #It is advised to set cycles to 1 when specifying a random_state
    random_state = None

    #Run the script
    for kernel in kernels:
        train_sizes, accuracies = learning_curve(data, kernel, cycles, random_state)  
        
        #Add the result lists to a plot
        plt.plot(train_sizes, accuracies, label="kernel = {}".format(kernel))
        print('')

    # Plot the result
    plt.xlabel('Training size (%)')
    plt.ylabel('Overall Accuracy (%)')
    plt.title('Learning Curve')
    plt.grid()
    plt.legend()
    plt.show()
    




data = pd.read_csv('./code/csv_data.csv')
#result = main(data)
CheckLabelAcc(data)
