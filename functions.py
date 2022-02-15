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

def color_plt3d(dataframe):
    colors = ['green', 'yellow', 'red', 'blue', 'orange']
    color_selection = []

    for i in range(5):
        color = colors[i]
        for i in range(100):
            color_selection.append(color)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('z_height')
    ax.set_ylabel('convex_hull_area')
    ax.set_zlabel('bounding_box_volume')

    ax.scatter(dataframe['z_height'], dataframe['convex_hull_area'], dataframe['bounding_box_volume'], color=color_selection)
    plt.show()