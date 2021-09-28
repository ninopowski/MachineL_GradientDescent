
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

# Plot function and derivative side-by-side

# Chart 1, cost function
# plt.subplot(2, 1, 1)
#
# plt.title('Cost function')
# plt.plot(x_1, f(x_1))
# plt.xlabel('x')
# plt.ylabel('f(x)')
#
# x_values = np.array(x_list)
# plt.scatter(x_list, f(x_values), color="red")
#
# # Chart 2, derivative
# plt.subplot(2, 1, 2)
#
# plt.title('Slope of the cost function')
# plt.plot(x_1, df(x_1))
# plt.xlabel('x')
# plt.ylabel('df(x)')
# plt.grid()
# plt.scatter(x_list, df(x_values), color="red")

#plt.show()



# Example 2 - Multiple minima vs initial guess, and advanced functions

# g(x) = x**4 - 4x**2 + 5

# Make some data
x_2 = np.linspace(start=-2, stop=2, num=100)


def g(x):
    return x**4 - 4*(x**2) + 5


def dg(x):
    return 4*(x**3) - 8*x



def gradient_descent(derivative_function, initial_guess, learning_rate=0.02, precision=0.001):
    new = initial_guess
    list_x = [new]
    list_slope = [derivative_function(new)]

    for n in range(30):
        previous = new
        gradient = derivative_function(previous)

        new = previous - gradient * learning_rate

        list_x.append(new)
        list_slope.append(derivative_function(new))

        if abs(new - previous) < precision:
            print(f'Loop run {n} times')
            break

    return new, list_x, list_slope


local_min, x_list, deriv_list = gradient_descent(dg, -0.2)
print(f'local min {local_min}')
print(f'Number of steps {len(x_list)}')



plt.subplot(1, 2, 1)
plt.plot(x_2, g(x_2))
plt.title('Example 2')
plt.xlabel('x_2')
plt.ylabel('g(x)')
plt.scatter(x_list, g(np.array(x_list)))

plt.subplot(1, 2, 2)
plt.plot(x_2, dg(x_2))
plt.xlabel('x_2')
plt.ylabel('dg(x)')
plt.grid()
plt.scatter(x_list, dg(np.array(x_list)))

plt.show()
