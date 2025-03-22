import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def calculate_objective(X, y, w):
    n = X.shape[0]
    e = y - X.dot(w)
    return 0.5/n * np.linalg.norm(e) ** 2


def compute_gradient(X, y, w):
    m = X.shape[0]
    i = random.randrange(m)
    h1 = (w[0] + w[1] * X[i, 1] - y[i])
    h2 = X[i, 1] * h1 # (y[i] - w[0] - w[1] * X[i, 1])
    return np.array([h1, h2],  dtype=np.float64)


def gradient_descent(X, y, w, step, tol=1e-1, max_iter=1000):
    objective_history = []
    for iter in range(max_iter):
        grad = compute_gradient(X, y, w)
#        if iter % 10000 == 0:
#            print(f"Iteration {iter}: Gradient norm = {np.linalg.norm(grad):.6f}, w = {w[0]} {w[1]}")
        if np.linalg.norm(grad) <= tol:
            break

        w = w - step * grad
        objective_history.append(calculate_objective(X, y, w))

    return w, objective_history, iter + 1


df = pd.read_csv("weight-height.csv")

df = df[['Height', 'Weight']]

X = df[['Height']].values
y = df[['Weight']].values

matrix_X = np.hstack((np.ones((X.shape[0], 1)), X))

m = X.shape[0]
spectral_norm = np.linalg.norm(matrix_X, 2)

L1 = (spectral_norm ** 2) / m
gamma = 1 / L1

w = np.array([[-346.44723667, 7.65286475]]).T

w_min, hist, i = gradient_descent(matrix_X, y, w, step=gamma, tol=1e-2, max_iter=1000000)
print(w_min)
print()
print('Step was: ', gamma)
print('Number of Iterations: ', i)

plt.plot(hist)
plt.title("Stochastic Gradient Descent")
plt.xlabel("Iteration")
plt.ylabel("Objective values")
plt.grid(True)
plt.show()
