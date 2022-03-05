"""
SHAPE RATIO POINT CLOUD

DEVELOPED BY:
Job de Vogel
"""

"""
FIND THE RATIO BETWEEN MAX X_RANGE AND Y_RANGE
UNDER CERTAIN HEIGHT

THIS SHOULD EXCLUDE TRAFFIC LIGHTS FROM CARS FOR EXAMPLE
"""

def shapeRatio(dataframe, splitheight):
    max_height = dataframe['z'].min(0) + splitheight
    data = dataframe[dataframe['z'] <= max_height]

    x_ratio = data['x'].max(0) - data['x'].min(0)
    y_ratio = data['y'].max(0) - data['y'].min(0)

    return y_ratio / x_ratio