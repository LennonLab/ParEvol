import parevol_tools as pt
import numpy as np

c = np.array([[1,2],[3,0],[0,2]])

c_new = pt.get_random_matrix(c)

print(c)
print(np.mean(c, axis=0))



print(c - np.mean(c, axis=0))
