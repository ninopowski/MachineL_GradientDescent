import matplotlib.pyplot as plt
import numpy as np


def g(x):
    return x**4 - 4*(x**2) + 5


def dg(x):
    return 4*(x**3) - 8*x


def gradient_descent(derivative_function, initial_guess, learning_rate=0.02, precision=0.001, max_iter=300):
    new = initial_guess
    list_x = [new]
    list_slope = [derivative_function(new)]

    for n in range(max_iter):
        previous = new
        gradient = derivative_function(previous)

        new = previous - gradient * learning_rate

        list_x.append(new)
        list_slope.append(derivative_function(new))

        if abs(new - previous) < precision:
            print(f'Loop run {n} times')
            break

    return new, list_x, list_slope

n = 100

low_gamma = gradient_descent(derivative_function=dg,
                             initial_guess=3,
                             learning_rate=0.0005,
                             precision=0.0001,
                             max_iter=n)

mid_gamma = gradient_descent(derivative_function=dg,
                             initial_guess=3,
                             learning_rate=0.001,
                             precision=0.0001,
                             max_iter=n)

high_gamma = gradient_descent(derivative_function=dg,
                             initial_guess=3,
                             learning_rate=0.002,
                             precision=0.0001,
                             max_iter=n)

extreme_gamma = gradient_descent(derivative_function=dg,
                             initial_guess=1.9,
                             learning_rate=0.25,
                             precision=0.0001,
                             max_iter=n)


# x axis data
x_values = list(range(0, n+1))

# y axis data
low_values = np.array(low_gamma[1])
mid_values = np.array(mid_gamma[1])
high_values = np.array(high_gamma[1])
extreme_values = np.array(extreme_gamma[1])



# Plotting the three cases

plt.title('Effect of the learning rate')
plt.xlabel('number of iterations')
plt.ylabel('cost')

# Ploting the low learning rate
plt.plot(x_values, g(low_values), color='lightgreen')
plt.scatter(x_values, g(low_values), s=20)

# Ploting the mid learning rate
plt.plot(x_values, g(mid_values), color='lightblue')
plt.scatter(x_values, g(mid_values), s=20)

# Plotting the high learning rate
plt.plot(x_values, g(high_values), color='yellow')
plt.scatter(x_values, g(high_values), s=20)

# Plotting the extreme learning rate
plt.plot(x_values, g(extreme_values), color='red')
plt.scatter(x_values, g(extreme_values), s=20)

plt.show()