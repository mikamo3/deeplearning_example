import numpy as np


def sigmoid(x):
    return 1 / (1+np.exp(-x))


X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1)+B1


Z1 = sigmoid(A1)
print(Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(A1, W2)+B2
Z2 = sigmoid(A2)

print(Z1.shape)
print(W2.shape)
print(B2.shape)
print(Z2)


def identiry_function(x):
    return x


W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.3])
A3 = np.dot(A2, W3)+B3
Y = identiry_function(A3)

print(Y)
