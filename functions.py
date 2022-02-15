import matplotlib.pyplot as plt

""" THIS FILE CONTAINS GENERAL FUNCTIONS """

def loading(file, amount_of_files):
    percentage = round((file / amount_of_files) * 100, 2)
    print('Preprocessing ' + str(percentage) + '% completed', end="\r")

def color_plt(dataframe):
    colors = ['green', 'yellow', 'red', 'blue', 'orange']
    color_selection = []

    for i in range(5):
        color = colors[i]
        for i in range(100):
            color_selection.append(color)

    dataframe.plot(kind='scatter', x='z_heights',y='convex_hull_areas',color=color_selection)
    plt.show()