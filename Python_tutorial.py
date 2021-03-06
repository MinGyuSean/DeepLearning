'''
Python Numpy
'''

'''
# Loops // enumerate : 순서와 리스트 내의 값을 전달

animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))

# Loops // List comprehension

nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)

Listcom1 = [x**2 for x in nums]
print(Listcom1)

Listcom2 = [x**2 for x in nums if x % 2 == 0]
print(Listcom2)

# Loops, Dictionary // List comprehension

nums = [0, 1, 2, 3, 4]
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))

even_num_to_square = {x: x**2 for x in nums if x % 2 == 0}
print(even_num_to_square)


# Loops, Set // List comprehension

animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))

from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)


# class

class Greeter(object):

    def __init__(self, name):
        self.name = name
    
    def greet(self, loud=False):
        if loud:
            print('Hello, %s!' %self.name.upper())
        else:
            print('Hello, %s' %self.name)

g = Greeter('Fred')
g.greet()
g.greet(loud='True')


# Numpy

import numpy as np

a = np.zeros((2,2))
print(a)
b = np.ones((2,2))
print(b)
c = np.full((2,2), 1)
print(c)
d = np.eye(2)     
print(d) 
e = np.random.random((2,2))
print(e)

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:2, 1:3]
row_r1 = a[1, :]
row_r2 = a[1:2, :]
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]

print(a)
print(b)
print(a[0,1])
print(b[0,0])
print(row_r1, row_r1.shape) # Rank1 view
print(row_r2, row_r2.shape) # Rank2 view
print(col_r1, col_r1.shape)
print(col_r2, col_r2.shape)

import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

print(a[[0, 1, 2], [0, 1, 0]])
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))

a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
b = np.array([0, 2, 0, 1])

print(a)
print(b)

a[np.arange(4), b] += 10
print(a)

import numpy as np

# x = np.array([[1,2],[3,4]], dtype=np.float64)
# y = np.array([[5,6],[7,8]], dtype=np.float64)

# print(x)
# print(y)

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])
v = np.array([9,10])
w = np.array([11,12])

# print(v.dot(w))
# print(np.dot(v, w))
# print(x.dot(v))
# print(np.dot(x, v))
# print(x.dot(y))
# print(np.dot(x, y))

print(np.sum(x))
print(np.sum(x, axis=0))
print(np.sum(x, axis=1))

import numpy as np

# x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
# v = np.array([0])
# y = np.empty_like(x)
# vv = np.tile(v, (5, 5))

# for i in range(4):
#     y[i, :] = x[i, :] + v

# print(x)
# print(y)
# print(vv)

v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])

print(np.reshape(v,(3,1))*w)
'''

import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

I = img.imread('C:/Temp/Test.jpg')
I_tinted = I * [1, 0.95, 0.9] #this leaves the red channel unchanged, and multiplies the green and blue channels by 0.95 and 0.9

plt.subplot(1, 2, 1)
plt.imshow(I)


'''
import numpy as np
from scipy.spatial.distance import pdist, squareform

x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)
d = squareform(pdist(x, 'euclidean'))
print(d)


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

plt.plot(x,y)
plt.show()
'''