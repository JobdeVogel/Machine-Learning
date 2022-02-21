import matplotlib.pyplot as plt

""" THIS FILE CONTAINS GENERAL FUNCTIONS """

def loading(file, amount_of_files):
    percentage = round((file / amount_of_files) * 100, 2)
    print('Preprocessing ' + str(percentage) + '% completed', end="\r")

"""
Green: Houses
Yellow: Cars
Red: Fences
Blue: Traffic Lights
Orange: Trees
"""

def color_plt(dataframe, *features):
    colors = ['green', 'yellow', 'red', 'blue', 'orange']
    color_selection = []

    for i in range(5):
        color = colors[i]
        for i in range(100):
            color_selection.append(color)

    if len(features) == 2:
        dataframe.plot(kind='scatter', x=features[0], y=features[1],color=color_selection)
    elif len(features) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])

        ax.scatter(dataframe[features[0]], dataframe[features[1]], dataframe[features[2]], color=color_selection)
    
    plt.show()