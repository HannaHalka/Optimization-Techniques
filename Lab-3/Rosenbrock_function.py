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


def compute_hessian(vector_x):
    x_val, y_val = vector_x[0], vector_x[1]
    h11, h12 = 120 * x_val**2 - 40 * y_val + 2, -40 * x_val
    h21 = h12
    h22 = 20
    return np.array([[h11, h12], [h21, h22]])


def newton(vector_x, max_iter=10000, tol=1e-6):
    x = np.array(vector_x, dtype=float)
    f_history = [calculate_objective(x)]

    for i in range(max_iter):
        grad = compute_gradient(x)
        hess = compute_hessian(x)

        d = np.linalg.solve(hess, -grad)
        x = x + d

        f_history.append(calculate_objective(x))

        if abs(f_history[-1] - f_history[-2]) <= tol:
            break

    return x, f_history, i+1


x1, x2 = np.array([2, 4]), np.array([-2, 10])

x1_final, f_hist1, iter1 = newton(x1)
x2_final, f_hist2, iter2 = newton(x2)

fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=100)

ax[0].plot(f_hist1, color="black")
ax[0].set_title("Newton's Method (start: [2, 4])")
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Objective value")
ax[0].grid(True)

ax[1].plot(f_hist2, color="black")
ax[1].set_title("Newton's Method (start: [-2, 10])")
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Objective value")
ax[1].grid(True)

plt.show()

print(f"Min x* = {x1_final}, "
      f"start poit was = {x1}, "
      f"number of iterations = {iter1}.")

print(f"Min x* = {x2_final}, "
      f"start poit was = {x2}, "
      f"number of iterations = {iter2}.")
