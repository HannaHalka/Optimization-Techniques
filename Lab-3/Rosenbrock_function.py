import numpy as np
import matplotlib.pyplot as plt


def calculate_objective(vector_x):
    x_val, y_val = vector_x[0], vector_x[1]
    rosenbrock_function = 10 * (y_val - x_val**2)**2 + (1 - x_val)**2
    return rosenbrock_function


def compute_gradient(vector_x):
    x_val, y_val = vector_x[0], vector_x[1]
    df_dx = -40 * x_val * (y_val - x_val**2) - 2 * (1 - x_val)
    df_dy = 20 * (y_val - x_val**2)
    return np.array([df_dx, df_dy])


def newton():
    pass


tol = 1e-6
max_iter = 10000

x_00 = np.array([2, 4])
x_01 = np.array([-2, 10])

plt.plot(...)
plt.title("Stochastic Gradient Descent")
plt.xlabel("Iteration")
plt.ylabel("Objective values")
plt.grid(True)
plt.show()
