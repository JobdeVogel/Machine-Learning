import numpy as np
import pandas as pd
import os

def z_height(dataframe):
    data = dataframe.values

    max = np.max(data[:, 2])
    min = np.min(data[:, 2])

    return max -  min