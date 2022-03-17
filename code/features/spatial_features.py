"""
THIS CODE USES THE COVARIANCE MATRIX AND EIGENVALUES
TO CALCULATE SPATIAL FEATURES:
- Linearity
- Planarity
- Sphericity
- Omnivariance
- Anisotropy
- Eigentropy
- Sum of eigenvalues
"""

import numpy as np
import pandas as pd
import os

filename = os.path.join('./code/data', '007.xyz')
data = pd.read_table(filename, skiprows=0, delim_whitespace=True, names=['x', 'y', 'z'])