import matplotlib.pyplot as plt
import numpy as np
a = [1, 3, 4, 12, 15]
print(len(a))
print(a[0:2])
print(a[1:])
print(a[:-1])

me = {"hoge": 1234}
print(me)

for i in [1, 2, 4]:
    print(i)


class Hoge:
    def __init__(self, a):
        self.a = a
        print(a)

    def hoge(self):
        print(self.a)


b = Hoge(1234)
b.hoge()


x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x)
print(type(x))
print(x+y)
print(x-y)
print(x+10)
x2 = np.array([[1, 2], [3, 4]])
y2 = np.array([[5, 6], [7, 8]])
print(x2+y2)
print(x2*y2)

# 1.5.5

A = np.array([[1, 2], [3, 4]])
B = np.array([[10, 20]])
print(A*B)

print(A.flatten())

# 1.6
x = np.arange(0, 6, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.savefig("hoge.png")

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, label="sin")
plt.plot(x, y2, label="cos", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
plt.legend()
plt.savefig("hoge2.png")
