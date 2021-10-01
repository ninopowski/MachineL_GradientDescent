import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from sympy import symbols, diff
from math import log



# Data visualisation with 3D charts



# Minimize
# f(x, y) = 1/(3**(-x**2-y**2)+1)


def f(x, y):
    return 1/(3**(-x**2-y**2)+1)

def fpx(x, y):
    return 2*3**(-x**2 - y**2)*x*log(3)/(3**(-x**2 - y**2) + 1)**2

def fpy(x, y):
    return 2*3**(-x**2 - y**2)*y*log(3)/(3**(-x**2 - y**2) + 1)**2

# Make data
x = np.linspace(start=-2, stop=2, num=200)
y = np.linspace(start=-2, stop=2, num=200)

x, y = np.meshgrid(x, y)
# Generating the 3D plot
fig = plt.figure(figsize=[16, 12])
ax = fig.gca(projection='3d')

ax.set_xlabel("X", fontsize=20)
ax.set_ylabel("Y", fontsize=20)
ax.set_zlabel("f(x, y) - cost", fontsize=20)
ax.plot_surface(x, y, f(x, y), cmap=cm.coolwarm, alpha=0.6)
#plt.show()


# Partial derivatives and symbolic computation
a, b = symbols("x, y")
print(f'Our cost function f(x, y) is: {f(a, b)}')
print(f'Partial derivative wrt x is: {diff(f(a, b), a)}')
print(f'Partial derivative wrt y is: {diff(f(a, b), b)}')
print(f'Value of (f(x, y) at x=1.8 and y=1.0 is: {f(a, b).evalf(subs={a: 1.8, b: 1.0})}')

print(f'Value of the slope wrt x at x=1.8 and y=1.0 is: {diff(f(a, b), a).evalf(subs={a: 1.8, b: 1.0})}')


# Batch gradient descent with sympy
# setup
learning_rate = 0.1
max_iter = 500
params = np.array([1.8, 1.0])  # initial guess

for n in range(max_iter):
    gradient_x = fpx(params[0], params[1])
    gradient_y = fpy(params[0], params[1])
    gradients = np.array([gradient_x, gradient_y])
    params = params - learning_rate * gradients

# Results
print(f'Values in gradient array {gradients}')
print(f'Minimum occurs at x value of: {params[0]}')
print(f'Minimum occurs at y value of: {params[1]}')
print(f'The cost is {f(params[0], params[1])}')


