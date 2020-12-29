from PIL import Image
import numpy as np
from deeplfromscratch.dataset.mnist import load_mnist
import sys
import os
sys.path.append(os.pardir)

(x_train, t_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(y_test.shape)


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.save("h.png")


img = x_train[0]
label = t_train[0]
print(label)
print(img.shape)
img = img.reshape(28, 28)
print(img.shape)
img_show(img)
