from cProfile import label

import numpy as np
import time
import matplotlib.pyplot as plt


def calculate_objective(A, b, x):
    m = A.shape[0]
    residual = A.dot(x) - b
    return 0.5/m * np.dot(residual, residual)


def compute_gradient(A, b, x):
    m = A.shape[0]
    return (1/m) * A.T.dot(A.dot(x) - b)


def gradient_descent(A, b, x0, step, max_iter=50):
    x = x0
    objective_history = []
    for i in range(max_iter):
        current_obj = calculate_objective(A, b, x)
        objective_history.append(current_obj)

        grad = compute_gradient(A, b, x)
        x = x - step * grad
    return x, objective_history


m, n = 4, 5
matrix_A = np.random.rand(m, n)
b = np.random.rand(m)
x0 = np.zeros(n)

m = matrix_A.shape[0]
spectral_norm_A = np.linalg.norm(matrix_A, 2)
L1 = (spectral_norm_A ** 2) / m

AtA = matrix_A.T @ matrix_A
Atb = matrix_A.T @ b
norm_AtA = np.linalg.norm(AtA, 2)
norm_Atb = np.linalg.norm(Atb, 2)

L2 = (1 / m) * (norm_AtA * 20 + norm_Atb) # ||x|| < 20

step_size = [0.1, 1/L1, 1/L2]

result = {}
for step in step_size:
    start_time = time.time()
    x_min, obj_values = gradient_descent(matrix_A, b, x0, step)
    end_time = time.time() - start_time

    result[step] = (obj_values, end_time)
    print(f"Step size = {step:.3g},"
          f"final objective = {obj_values[-1]:.3g},"
          f"runtime (50 iters) = {end_time:.3f} s")

plt.figure(figsize=(8, 6))
for step_size, (obj_values, et) in result.items():
    plt.plot(obj_values, label=f"gamma={step_size:.3g}, time={et:.3f}s")

plt.title("Least Squares Estimation")
plt.xlabel("Intervals")
plt.ylabel("Objective values")
plt.legend()
plt.grid(True)
plt.show()

