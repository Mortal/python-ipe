import numpy as np


data = np.load('counts.npz')['counts']
print(data.nonzero()[0][-1])

ind = np.argsort(data)
print(ind[-10:])
print(data[ind[-10:]])
