
# Imports and packages
import matplotlib.pyplot as plt
import numpy as np


# Example 1 - A simple cost function
# f(x) = x**2 + x + 1

def f(x):
    return x**2 + x + 1


# Make data
x_1 = np.linspace(start=-3, stop=3, num=100)


# Slope and derivatives

def df(x):
    return 2*x + 1


# Plot function and derivative side-by-side

# Chart 1, cost function
plt.subplot(2, 1, 1)

plt.title('Cost function')
plt.plot(x_1, f(x_1))
plt.xlabel('x')
plt.ylabel('f(x)')

# Chart 2, derivative
plt.subplot(2, 1, 2)

plt.title('Slope of the cost function')
plt.plot(x_1, df(x_1))
plt.xlabel('x')
plt.ylabel('df(x)')
plt.grid()

plt.show()

