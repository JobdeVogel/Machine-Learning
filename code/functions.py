"""
FUNCTIONS

DEVELOPED BY:
Job de Vogel
"""

import matplotlib.pyplot as plt

""" THIS FILE CONTAINS GENERAL FUNCTIONS """

def loading(text, step, count):
    percentage = round((step / (count - 1)) * 100, 2)

    if percentage < 100:
        print(text + ' ' + str(percentage) + '% completed', end="\r")
    else:
        print(text + ' ' + str(percentage) + '% completed')
    return

def color_plt(dataframe, color_selection, *features):
    if len(features) == 2:
        dataframe.plot(kind='scatter', x=features[0], y=features[1],color=color_selection)
    elif len(features) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])

        for feature in features:
            if feature not in dataframe.columns:
                print('{} is not available in csv_data, please preprocess manually'.format(feature))
                print('Currently available in csv_data: {}'.format(dataframe.columns))
                return

        ax.scatter(dataframe[features[0]], dataframe[features[1]], dataframe[features[2]], color=color_selection)
    
    plt.show()