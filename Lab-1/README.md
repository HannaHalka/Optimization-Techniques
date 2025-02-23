# Rosenbrock function

$100 \times  (y - x^2)^2 + (1 - x)^2$

We don't find the solution with step_size = 0,1 and step_size = 0,01.

![img.png](img.png)

Results for different step size: 

$--------------------$

Step size: 0.001

Number of iterations: 10000

Time spent: 0.3144083023071289

Final point: $[0.99117313 0.98238866]$

Final objective value: 7.80397926096646e-05

$--------------------$

# Least Squares Estimation
Calculates the least-squares objective:
$f(x) = \frac{1}{2m} \times ||A x - b||^2$, where m = number of rows in matrix_A.

Computes the gradient of the least-squares objective:
$grad f(x) = \frac{1}{m} \times A^T (A x - b)$, m the same.


Our matrix A looks like:

$\begin{vmatrix}
0.822&0.034&0.784&0.331&0.435\\
0.707&0.576&0.227&0.022&0.788\\
0.997&0.902&0.929&0.493&0.566\\
0.095&0.819&0.804&0.653&0.516\\
\end{vmatrix} $

And our vector b looks like:

$\begin{vmatrix}
0.694 \\
0.701 \\
0.906 \\
0.805 \\
\end{vmatrix} $

## Results

![img_2.png](img_2.png)


Step size = 0.1,final objective = 0.00301,runtime (50 iters) = 0.002 s

Step size = 0.558,final objective = 0.00108,runtime (50 iters) = 0.001 s

Step size = 0.0271,final objective = 0.00587,runtime (50 iters) = 0.001 s