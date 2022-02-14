import numpy as np
import pandas as pd

"""
This is a very time consuming calculation. Therefore the pointcloud
is reduced by a specified factor. The precision of the final result
wil be based on the height of this reduction factor.
"""

def getConvexHullArea(points, prec_factor):
    # Reduce amount of points by precision factor
    seed = np.random.randint(0, len(points.values), int(prec_factor * len(points.values)))
    points = points.values[seed]

    #We need at least 3 points
    n = len(points)
    if n < 3:
        return

    def getOrientation(p, i, q):
        #We define orientation as: 0=collinear, 1=clockwise, 2=counterclockwise
        val = (i[1] - p[1]) * (q[0] - i[0]) - (i[0] - p[0]) * (q[1] - i[1])

        if val == 0:
            return 0
        elif val > 0:
            return 1
        else:
            return 2

    #First we find the left most point, defined as p
    min_x_index = points.min(0)

    p = 0
    for i in range(0, len(points)):
        if points[i][0] < points[p][0]:
            p = i
        elif points[i][0] == points[p][0]:
             if points[i][1] > points[p][1]:
                p = i

    #Now we will fill a list with all hull outer point coordinates
    hull_x = []
    hull_y = []
    p_start = p
    q = 0
    while(True):
        #Save X and Y of hull coordinates
        hull_x.append(points[p][0])
        hull_y.append(points[p][1])

        #find next list index, wrap at end
        q = (p + 1) % n

        #Find the most counterclockwise result of all other points
        for i in range(n):
            if (getOrientation(points[p], points[i], points[q]) == 2):
                q = i

        #Save this result as the new p
        p = q

        #When a full circle is completed: stop
        if (p == p_start):
            break

    #Return the area using the shoelace formula
    return 0.5*np.abs(np.dot(hull_x, np.roll(hull_y, 1))-np.dot(hull_y, np.roll(hull_x, 1)))