import numpy as np

rho = 0.3
A = np.array([[1,0,0,rho,rho], [0,1,rho,0,rho], [0,rho,1,rho,0], [rho,0,rho,1,0], [rho,rho,0,0,1]])

print(A)

print(np.all(np.linalg.eigvals(A) > 0))
