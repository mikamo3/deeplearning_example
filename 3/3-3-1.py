import numpy as np
A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))
print(np.shape(A))

A = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]])
print(A)
print(np.ndim(A))
print(np.shape(A))

A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([7, 8])
print(np.shape(A))
print(np.shape(B))
print(np.dot(A, B))
