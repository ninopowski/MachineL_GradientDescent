import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from sympy import symbols, diff



# Data visualisation with 3D charts



# Minimize
# f(x, y) = 1/(3**(-x**2-y**2)+1)


def f(x, y):
    return 1/(3**(-x**2-y**2)+1)

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
print(f'Value of (f(x, y) at x=1.8 and y=1.0 is: {f(a, b).evalf(subs={a: 1.8, b: 1.0})}')

print(f'Value of the slope wrt x at x=1.8 and y=1.0 is: {diff(f(a, b), a).evalf(subs={a: 1.8, b: 1.0})}')





