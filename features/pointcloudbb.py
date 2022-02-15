# from sklearn.decomposition import PCA
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

np.dot()


data = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6], [1.1, 0.9]])

data_df = pd.DataFrame(data, columns=['x', 'y'])

def eigen_vectors(data):
    pca = PCA()
    pca.fit(data)

    return pca.components_


x, y = eigen_vectors(data)

"""
origin = np.array([[1.5, 1.5],[1.5, 1.5]]) # origin point
data_df.plot(kind='scatter', x='x',y='y',color='red')
plt.quiver(*origin, V[:,0], V[:,1], color=['r','b'], scale=5)

#dot product!!

plt.show()
"""