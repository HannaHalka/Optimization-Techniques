import numpy as np
import time
import matplotlib.pyplot as plt


def calculate_objective(X, function):
    if function == "f1":
        return 0.5 * (X - np.log(1 + X)) ** 2
    elif function == "f2":
        return 0.5 * (X - np.log(2 + X)) ** 2
    else:
        raise ValueError("Unknown function type. Accepts only 'f1' and 'f2'! ")


def compute_gradient(X, function):
    if function == "f1":
        return (X - np.log(1 + X)) * (1 - 1 / (1 + X))
    elif function == "f2":
        return (X - np.log(2 + X)) * (1 - 1 / (2 + X))
    else:
        raise ValueError("Unknown function type. Accepts only 'f1' and 'f2'! ")

def gradient_descent(X0, function, step=0.01, iters=100):
    x = X0
    history = []

    start_time = time.time()
    for i in range(iters):
        obj_value = calculate_objective(x, function)
        history.append((i, x, obj_value))

        grad = compute_gradient(x, function)
        x = x - step * grad

    stop_time = time.time() - start_time
    return x, history, stop_time


if __name__ == "__main__":
    x_vals = np.linspace(0, 2, 200)
    y_identity = x_vals
    y_f1 = np.log(1 + x_vals)  # f1(x) = ln(1 + x)
    y_f2 = np.log(2 + x_vals)  # f2(x) = ln(2 + x)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_identity, label="y = x")
    plt.plot(x_vals, y_f1, label="y = ln(1 + x)")
    plt.plot(x_vals, y_f2, label="y = ln(2 + x)")

    plt.xlim([0,2])
    plt.ylim([0,2])
    plt.grid(True)
    plt.legend()

    plt.title("Fixed Point Problems")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

x0_f1 = 1
step_f1 = 0.2

x_start_f1, hist_f1, time_f1 = gradient_descent(x0_f1, function="f1", step=step_f1)

print("=== Gradient Descent for g1(x) = 1/2 (x - ln(1 + x))^2 ===")
print(f"Initial gess: {x0_f1}, Step size: {step_f1}")
print(f"final solution: x* = {x_start_f1:.4g}")
print(f"Final g1(x*): {calculate_objective(x_start_f1, 'f1'):.4f}")
print(f"Total time: {time_f1:.4f}")

print(f"\nSelected iteration snapshots for g1(x):")
for (k, xx, val) in hist_f1:
    if k % 10 == 0:
        print(f"Iterations: {k:4d}, x = {xx:.4f}, g1(x) = {val:.4e}")


x0_f2 = 2
step_f2 = 0.1

x_start_f2, hist_f2, time_f2 = gradient_descent(x0_f2, function="f2", step=step_f2)

print("\n=== Gradient Descent for g2(x) = 1/2 (x - ln(2 + x))^2 ===")
print(f"Initial gess: {x0_f2}, Step size: {step_f2}")
print(f"final solution: x* = {x_start_f2:.4g}")
print(f"Final g1(x*): {calculate_objective(x_start_f2, 'f2'):.4f}")
print(f"Total time: {time_f2:.4f}")

print(f"\nSelected iteration snapshots for g2(x):")
for (k, xx, val) in hist_f2:
    if k % 10 == 0:
        print(f"Iterations: {k:4d}, x = {xx:.4f}, g1(x) = {val:.4e}")


it_f1 = [h[0] for h in hist_f1]
g1_vals = [h[2] for h in hist_f1]
it_f2 = [h[0] for h in hist_f2]
g2_vals = [h[2] for h in hist_f2]

if __name__ == "__main__":
    plt.figure(figsize=(7,5))
    plt.plot(it_f1, g1_vals, label="g1(x) = 1/2 (x - ln(1+x))^2")
    plt.plot(it_f2, g2_vals, label="g2(x) = 1/2 (x - ln(2+x))^2")
    plt.xlabel("Iteration")
    plt.ylabel("Objective value")
    plt.legend()
    plt.grid(True)
    plt.title("Convergence of g1(x) and g2(x)")
    plt.show()
