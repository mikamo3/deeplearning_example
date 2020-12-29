import numpy as np


def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x-c)
    sum_exp_a = np.sum(exp_a)
    print(sum_exp_a)
    y = exp_a/sum_exp_a
    print(y)
    return y


a = np.array([0.3, 2.9, 4.0])
hoge = softmax(a)

print(np.sum(hoge))

a = np.array([1000, 2000, 3000])
softmax(a)
