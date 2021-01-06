from deeplfromscratch.dataset.mnist import load_mnist
from deeplfromscratch.common.util import shuffle_dataset
import numpy as np
(x_train, t_train), (x_test, t_test) = load_mnist()
x_train, t_train = shuffle_dataset(x_train, t_train)

validation_rate = 0.20
validation_num = int(x_train.shape[0]*validation_rate)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]

weight_decay = 10**np.random.uniform(-8, -4)
lr = 10**np.random.uniform(-6, -2)
