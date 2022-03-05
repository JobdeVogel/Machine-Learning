"""
Z-HEIGHT POINT CLOUD

DEVELOPED BY:
Job de Vogel
"""

import numpy as np
import pandas as pd

def z_height(dataframe):
    data = dataframe.values

    max = np.max(data[:, 2])
    min = np.min(data[:, 2])

    return max -  min