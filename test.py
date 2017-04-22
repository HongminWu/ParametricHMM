import numpy as np
n_components = 12
self_trans = 0.5
transMatrix = np.identity(n_components) * self_trans
transMatrix[- 1, - 1] = 1
for i in np.arange(n_components - 1):
    for j in np.arange(n_components - i - 2):
        transMatrix[i, i + j + 1] = 1. / (n_components - 1. - i) - j / (
        (n_components - 1. - i) * (n_components - 2. - i))
transMatrix[n_components - 2, -1] = self_trans

print transMatrix.sum(axis= 1)