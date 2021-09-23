
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


# Gradient descent

new_x = 3
previous_x = 0
step_multiplier = 0.1
precision = 0.0001

x_list = [new_x]
slope_list = [df(new_x)]


for n in range(500):
    previous_x = new_x
    gradient = df(previous_x)
    new_x = previous_x - step_multiplier * gradient

    x_list.append(new_x)
    slope_list.append(df(new_x))

    if abs(new_x - previous_x) < precision:
        print(f'Loop run {n} times.')
        break


print(f'The local minimum is at: {new_x}')
print(f'Slope df(x) at this value is: {df(new_x)}')
print(f'The cost value at this point is: {f(new_x)}')

# Superimpose the gradient descent calc
plt.subplot(1, 3, 1)

plt.show()