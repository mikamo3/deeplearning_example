import numpy as np


def softmax(x):
    exp_a = np.exp(x)
    sum_exp_a = np.sum(exp_a)
    print(sum_exp_a)
    y = exp_a/sum_exp_a
    print(y)
    return y


a = np.array([0.3, 2.9, 4.0])
softmax(a)
