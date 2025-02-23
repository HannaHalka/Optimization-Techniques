import numpy as np
import time
import matplotlib.pyplot as plt

def calculate_objective(X):
    x_val, y_val = X[0], X[1]
    rosenbrock_function = 100 * (y_val - x_val**2)**2 + (1 - x_val)**2
    return rosenbrock_function


def compute_gradient(X):
    x_val, y_val = X[0], X[1]
    df_dx = -400 * x_val * (y_val - x_val**2) - 2 * (1 - x_val)
    df_dy = 200 * (y_val - x_val**2)
    return np.array([df_dx, df_dy])


def gradient_descent(X_0, step, max_iter=10000, tol=0.0001):
    x_current = np.array(X_0, dtype=float)
    x_history = []
    f_history = []

    start_time = time.time()

    for i in range(max_iter):
        grad = compute_gradient(x_current)

        if np.linalg.norm(grad) <= tol:
            break

        x_current = x_current - step * grad
        x_history.append(x_current)
        f_history.append(calculate_objective(x_current))

    end_time = time.time() - start_time

    return x_history, f_history, i+1, end_time


if __name__ == "__main__":
    X_0 = np.array([-2, 2])
    # step_size = [0.1, 0.01, 0.001] to big step size
    step_size = [0.001]

    plt.figure(figsize=(8, 6))

    result = []
    for step in step_size:
        x_hist, f_hist, num_iter, t_spent = gradient_descent(X_0, step)
        result.append((step, num_iter, t_spent, x_hist[-1], f_hist[-1]))

        plt.plot(range(num_iter), f_hist, label=f"alpha={step}")

    plt.title("Gradient Descent on Rosenbrock function")
    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Results for different step size: ")
    print("---------------------------------")
    for (step, num_iter, t_spent, x_hist, f_hist) in result:
        print(f"Step size: {step}")
        print(f"Number of iterations: {num_iter}")
        print(f"Time spent: {t_spent}")
        print(f"Final point: {x_hist}")
        print(f"Final objective value: {f_hist}")
        print("---------------------------------")
