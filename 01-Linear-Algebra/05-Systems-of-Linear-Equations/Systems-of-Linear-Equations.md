# System of Equations: Mathematics for Machine Learning and Reinforcement Learning

**Topics:** Linear Systems, Substitution, Elimination, Matrix Representation, Gaussian Elimination, Matrix Rank, Least Squares, and Numerical Solutions

**Level:** Beginner to Intermediate  
**Prerequisites:** Basic Algebra, Vectors, and Matrix Operations  
**Programming Language:** Python  
**Library:** NumPy

---

## 1. Learning Objectives

By the end of this lesson, you will be able to:

1. Understand what a system of equations is.
2. Distinguish between linear and nonlinear systems.
3. Solve systems of equations using substitution and elimination.
4. Represent linear systems using matrices and vectors.
5. Understand Gaussian elimination and elementary row operations.
6. Identify systems with unique solutions, infinitely many solutions, or no solution.
7. Understand how matrix rank determines the existence and uniqueness of solutions.
8. Solve linear systems using Python and NumPy.
9. Understand least-squares estimation and its connection to linear regression.
10. Explain how systems of equations appear in reinforcement learning, including Bellman equations and policy evaluation.
11. Understand applications of linear systems in intelligent scheduling and multi-agent reinforcement learning.

---

## 2. What Is a System of Equations?

A system of equations is a collection of equations that must be satisfied simultaneously.

Consider the following system:

```text
x + y = 5
x - y = 1
```

We need to find values of `x` and `y` that satisfy both equations.

The solution is:

```text
x = 3
y = 2
```

Verification:

First equation:

```text
x + y = 3 + 2 = 5
```

Second equation:

```text
x - y = 3 - 2 = 1
```

Both equations are satisfied. Therefore, the solution is:

```text
(x, y) = (3, 2)
```

A solution to a system of equations is a set of variable values that makes every equation in the system true.

### Why Are Systems of Equations Important?

Systems of equations are useful whenever several unknown quantities are related by multiple constraints.

Examples include:

- Estimating the parameters of a mathematical model.
- Calculating the coefficients of a regression model.
- Solving electrical circuit equations.
- Calculating equilibrium points.
- Evaluating policies in finite-state reinforcement learning.
- Modeling resource allocation in cloud and fog computing.
- Estimating value functions in certain reinforcement learning algorithms.

---

## 3. Linear and Nonlinear Systems

### 3.1. Linear Systems

A linear equation with `n` variables has the general form:

```text
a1*x1 + a2*x2 + ... + an*xn = b
```

Here:

- `x1, x2, ..., xn` are the unknown variables.
- `a1, a2, ..., an` are known coefficients.
- `b` is a known constant.

In a linear equation, the variables appear only to the first power and are not multiplied by one another.

For example:

```text
2*x + y = 7
x + 3*y = 11
```

This is a linear system.

In two dimensions, each linear equation represents a straight line. If the system has a unique solution, the two lines intersect at exactly one point.

### 3.2. Nonlinear Systems

Consider:

```text
x^2 + y = 5
x + y = 3
```

The first equation contains `x^2`, so the system is nonlinear.

Other examples of nonlinear terms include:

```text
x*y
sin(x)
exp(x)
x^3
```

Nonlinear systems may have no solution, one solution, several solutions, or infinitely many solutions.

Their solution methods differ from those used for ordinary linear systems.

**Important:** This lesson focuses primarily on linear systems because they are fundamental to linear algebra, machine learning, and many reinforcement learning methods.

---

## 4. Solving Linear Systems by Substitution

The substitution method involves solving one equation for one variable and substituting the result into another equation.

Consider:

```text
x + y = 7
x - y = 1
```

### Step 1: Solve the First Equation for x

Starting with:

```text
x + y = 7
```

Subtract `y` from both sides:

```text
x = 7 - y
```

### Step 2: Substitute into the Second Equation

The second equation is:

```text
x - y = 1
```

Substitute `x = 7 - y`:

```text
7 - y - y = 1
```

Simplify:

```text
7 - 2*y = 1
-2*y = -6
y = 3
```

### Step 3: Calculate x

Using:

```text
x = 7 - y
```

we obtain:

```text
x = 7 - 3
x = 4
```

The final solution is:

```text
x = 4
y = 3
```

### When Is Substitution Useful?

Substitution is especially useful for small systems in which one equation can easily be rearranged to isolate a variable.

For large systems, matrix-based methods are generally more practical.

---

## 5. Solving Linear Systems by Elimination

The elimination method combines equations to eliminate one or more variables.

Consider:

```text
2*x + y = 8
x - y = 1
```

### Step 1: Add the Equations

Add the left-hand sides and right-hand sides:

```text
(2*x + y) + (x - y) = 8 + 1
```

The `y` terms cancel:

```text
3*x = 9
x = 3
```

### Step 2: Calculate y

Substitute `x = 3` into the second equation:

```text
x - y = 1
3 - y = 1
-y = -2
y = 2
```

The final solution is:

```text
x = 3
y = 2
```

### Why Does Elimination Work?

Adding or subtracting equations can produce another equation that is satisfied by the same solution, provided the operations are performed correctly.

The goal is to eliminate variables until the remaining equations become easy to solve.

This idea generalizes to Gaussian elimination, a fundamental technique in computational linear algebra.

---

## 6. Representing a System of Equations Using Matrices

Matrix notation provides a compact representation of a system of linear equations.

Consider:

```text
2*x + y = 5
x + 3*y = 7
```

The system can be written as:

```text
A*x = b
```

Here, `A` is the coefficient matrix, `x` is the unknown vector, and `b` is the right-hand-side vector.

The coefficient matrix is:

```text
A = [ [2, 1],
      [1, 3] ]
```

The unknown vector is:

```text
x = [x, y]
```

The right-hand-side vector is:

```text
b = [5, 7]
```

The matrix equation is equivalent to the original system:

```text
2*x + 1*y = 5
1*x + 3*y = 7
```

### 6.1. General Matrix Form

A system of `m` linear equations in `n` unknowns can be represented as:

```text
A*x = b
```

The dimensions are:

```text
A: m rows and n columns
x: n elements
b: m elements
```

In mathematical notation, these dimensions are often written as:

```text
A: m x n
x: n x 1
b: m x 1
```

The matrix `A` contains the coefficients of the unknown variables. The vector `x` contains the unknowns, and `b` contains the constants on the right-hand side.

This representation is the starting point for many numerical methods used in machine learning and scientific computing.

---

## 7. Solving a System Using the Inverse Matrix

Consider the matrix equation:

```text
A*x = b
```

If `A` is a square, invertible matrix, we can multiply both sides by its inverse, `A_inverse`:

```text
A_inverse * A * x = A_inverse * b
```

Because:

```text
A_inverse * A = I
```

where `I` is the identity matrix, we obtain:

```text
I*x = A_inverse * b
```

Therefore:

```text
x = A_inverse * b
```

This formula is mathematically useful, but explicitly calculating the inverse is not normally the best way to solve a system numerically.

### 7.1. Worked Example

Consider:

```text
2*x + y = 5
x + 3*y = 7
```

The coefficient matrix is:

```text
A = [ [2, 1],
      [1, 3] ]
```

The right-hand-side vector is:

```text
b = [5, 7]
```

For a 2-by-2 matrix:

```text
A = [ [a, b],
      [c, d] ]
```

the determinant is:

```text
det(A) = a*d - b*c
```

For our matrix:

```text
det(A) = 2*3 - 1*1
det(A) = 5
```

Because the determinant is nonzero, the matrix is invertible.

The inverse is:

```text
A_inverse = (1/5) * [ [ 3, -1],
                       [-1,  2] ]
```

Now multiply the inverse by the right-hand-side vector:

```text
x = A_inverse * b
```

The first component is:

```text
x = (3*5 - 1*7)/5
x = 8/5
x = 1.6
```

The second component is:

```text
y = (-1*5 + 2*7)/5
y = 9/5
y = 1.8
```

The solution is:

```text
x = 1.6
y = 1.8
```

### 7.2. An Important Numerical Computing Rule

Although the inverse formula is useful for understanding the mathematics, use a numerical solver in practical Python applications.

```python
x = np.linalg.solve(A, b)
```

For a suitable square, nonsingular matrix, this is generally more efficient and numerically preferable to explicitly computing the inverse.

---

## 8. Gaussian Elimination

Gaussian elimination is a systematic method for solving linear systems by transforming their equations into an easier form.

It uses elementary row operations.

The three elementary row operations are:

1. Swap two rows.
2. Multiply a row by a nonzero number.
3. Add a multiple of one row to another row.

These operations preserve the solution set of the system.

### 8.1. The Augmented Matrix

Consider:

```text
2*x + y = 5
x + 3*y = 7
```

The augmented matrix contains the coefficients and the right-hand-side constants:

```text
[ 2   1 | 5 ]
[ 1   3 | 7 ]
```

The vertical line separates the coefficient matrix from the constants.

First, swap the two rows:

```text
[ 1   3 | 7 ]
[ 2   1 | 5 ]
```

Now subtract twice the first row from the second row:

```text
New row 2 = Old row 2 - 2 * Row 1
```

The resulting augmented matrix is:

```text
[ 1   3 |  7 ]
[ 0  -5 | -9 ]
```

The second row represents:

```text
-5*y = -9
```

Therefore:

```text
y = 9/5
y = 1.8
```

Substitute into the first equation:

```text
x + 3*y = 7
x + 3*1.8 = 7
x = 1.6
```

The final solution is:

```text
x = 1.6
y = 1.8
```

### 8.2. Row-Echelon Form

Gaussian elimination transforms a matrix into row-echelon form.

In this form:

- Nonzero rows appear above zero rows.
- Each leading entry appears to the right of the leading entry in the row above.
- Entries below each leading entry are zero.

Once the system reaches this form, back-substitution can be used to calculate the unknowns.

Gaussian elimination is widely used in numerical linear algebra and is closely related to the methods used by many linear-system solvers.

---

## 9. Three Possible Outcomes of a Linear System

A system of linear equations can have:

1. Exactly one solution.
2. Infinitely many solutions.
3. No solution.

Understanding these possibilities is important because not every mathematical model has a unique solution.

### 9.1. Case 1: A Unique Solution

Consider:

```text
x + y = 5
x - y = 1
```

Add the equations:

```text
2*x = 6
x = 3
```

Substitute into the first equation:

```text
3 + y = 5
y = 2
```

The unique solution is:

```text
x = 3
y = 2
```

Geometrically, the two lines intersect at one point.

For a square coefficient matrix `A`, a unique solution exists for every right-hand-side vector `b` if and only if `A` is invertible.

For an `n`-by-`n` matrix, this is equivalent to:

```text
rank(A) = n
```

It is also equivalent to:

```text
det(A) != 0
```

### 9.2. Case 2: Infinitely Many Solutions

Consider:

```text
x + y = 4
2*x + 2*y = 8
```

The second equation is twice the first equation.

Both equations describe the same line. Therefore, every pair satisfying `x + y = 4` is a solution.

Examples include:

```text
(x, y) = (0, 4)
(x, y) = (1, 3)
(x, y) = (2, 2)
```

There are infinitely many solutions.

The coefficient matrix is:

```text
A = [ [1, 1],
      [2, 2] ]
```

Its rank is:

```text
rank(A) = 1
```

The augmented matrix also has rank 1. Since the number of unknowns is 2 and the system is consistent, the system has infinitely many solutions.

### 9.3. Case 3: No Solution

Consider:

```text
x + y = 4
x + y = 7
```

The same expression cannot equal both 4 and 7.

Therefore, the system has no solution.

The augmented matrix is:

```text
[ 1   1 | 4 ]
[ 1   1 | 7 ]
```

Subtract the first row from the second row:

```text
[ 1   1 | 4 ]
[ 0   0 | 3 ]
```

The second row represents:

```text
0*x + 0*y = 3
```

or simply:

```text
0 = 3
```

This is a contradiction, so the system has no solution.

Geometrically, the two equations represent parallel, distinct lines.

---

## 10. Matrix Rank and the Existence of Solutions

Matrix rank provides a systematic way to determine whether a linear system is consistent and whether its solution is unique.

Consider:

```text
A*x = b
```

The augmented matrix is formed by appending `b` as an additional column:

```text
[A | b]
```

Let:

```text
rA = rank(A)
rAug = rank([A | b])
n = number of unknowns
```

The system has at least one solution if and only if:

```text
rA = rAug
```

This is the consistency condition.

If the system is consistent:

- If `rA = n`, the solution is unique.
- If `rA < n`, infinitely many solutions exist.

If the ranks differ, the system has no solution.

### Summary Table

| Condition | Result |
|---|---|
| rank(A) != rank([A \| b]) | No solution |
| rank(A) = rank([A \| b]) = n | Unique solution |
| rank(A) = rank([A \| b]) < n | Infinitely many solutions |

This framework applies to both square and rectangular systems.

**Key insight:** Matrix rank describes the number of linearly independent rows or columns. It helps reveal whether the equations provide enough independent information to determine the unknown variables.

---

## 11. Overdetermined and Underdetermined Systems

Not every system has the same number of equations and unknowns.

### 11.1. Overdetermined Systems

An overdetermined system has more equations than unknowns.

For example:

```text
x + y = 2
2*x - y = 1
x + 2*y = 5
```

There are three equations and two unknowns.

The equations may be inconsistent, meaning no values satisfy all of them exactly.

In data analysis, overdetermined systems often arise when many observations are used to estimate a smaller number of parameters.

When an exact solution does not exist, we can seek a least-squares solution that minimizes:

```text
sum of squared errors = sum((A*x - b)^2)
```

In vector notation, the objective is:

```text
minimize ||A*x - b||_2^2
```

Here, `||...||_2` represents the Euclidean norm.

Least squares is fundamental to linear regression.

### 11.2. Underdetermined Systems

An underdetermined system has fewer equations than unknowns.

For example:

```text
x + y + z = 6
```

There is one equation and three unknowns.

Examples of solutions include:

```text
(x, y, z) = (6, 0, 0)
(x, y, z) = (0, 6, 0)
(x, y, z) = (1, 2, 3)
```

Infinitely many solutions exist.

Additional constraints or assumptions may be required to choose one solution.

In machine learning, regularization, prior knowledge, and structural assumptions can help select useful solutions from a large set of possibilities.

---

## 12. Least-Squares Solutions and Machine Learning

Suppose we have a dataset containing `m` observations and `n` features.

Let:

```text
X = feature matrix with shape (m, n)
y = target vector with length m
w = model parameter vector with length n
```

We want to find model parameters such that:

```text
X*w approximately equals y
```

An exact solution may not exist because real-world data can contain noise or because the system has more equations than unknowns.

We can minimize the squared prediction error:

```text
minimize ||X*w - y||_2^2
```

The residual vector is:

```text
e = X*w - y
```

The residual measures the difference between the model's predictions and the target values.

### 12.1. Normal Equations

For an unconstrained least-squares problem, the normal equations are:

```text
X_transpose * X * w_hat = X_transpose * y
```

Here, `X_transpose` is the transpose of `X`, and `w_hat` denotes an estimated parameter vector.

If `X` has full column rank, then `X_transpose * X` is invertible. The solution can be written as:

```text
w_hat = inverse(X_transpose * X) * X_transpose * y
```

This is the familiar closed-form expression for ordinary least-squares regression under the full-column-rank condition.

However, explicitly forming the inverse is generally not the recommended numerical implementation. QR decomposition, singular value decomposition (SVD), and least-squares solvers are often preferable.

If the features are linearly dependent, the coefficients may not be uniquely determined. A least-squares solution still exists, but the pseudoinverse or regularization may be needed to select a particular solution.

### 12.2. Connection to Linear Regression

In linear regression, the model predicts a target using a weighted combination of features.

For example:

```text
prediction = w1*feature1 + w2*feature2 + ... + wn*featuren
```

The goal is to find weights that minimize prediction error across the dataset.

Thus, least-squares estimation converts a data-fitting problem into a mathematical optimization problem closely related to linear systems.

---

## 13. Python Implementation with NumPy

NumPy provides efficient tools for solving linear systems and performing matrix operations.

Install NumPy if necessary:

```bash
pip install numpy
```

### 13.1. Solving a Square System

Consider:

```text
2*x + y = 5
x + 3*y = 7
```

The coefficient matrix and right-hand-side vector are:

```text
A = [ [2, 1],
      [1, 3] ]

b = [5, 7]
```

Python implementation:

```python
import numpy as np

A = np.array([
    [2, 1],
    [1, 3]
], dtype=float)

b = np.array([5, 7], dtype=float)

x = np.linalg.solve(A, b)

print("Solution:")
print(x)

print("\nVerification:")
print(A @ x)

print("\nRight-hand side:")
print(b)
```

Expected output:

```text
Solution:
[1.6 1.8]

Verification:
[5. 7.]

Right-hand side:
[5. 7.]
```

The solution is:

```text
x = 1.6
y = 1.8
```

The expression `A @ x` performs matrix-vector multiplication.

The verification checks whether:

```text
A @ x approximately equals b
```

Small differences can occur because computers represent many decimal values approximately.

### 13.2. Checking the Residual

The residual measures the difference between the calculated left-hand side and the target vector.

```python
import numpy as np

A = np.array([
    [2, 1],
    [1, 3]
], dtype=float)

b = np.array([5, 7], dtype=float)

x = np.linalg.solve(A, b)

residual = A @ x - b

print("Solution:", x)
print("Residual:", residual)
print("Residual norm:", np.linalg.norm(residual))
```

The residual should be close to zero for a correctly solved system.

The residual norm is calculated as:

```text
||A @ x - b||_2
```

A small residual indicates that the computed solution approximately satisfies the equations.

However, a small residual does not always guarantee a small error in the unknown variables when the coefficient matrix is poorly conditioned.

### 13.3. Detecting a Singular Matrix

Consider:

```text
A = [ [1, 2],
      [2, 4] ]
```

The determinant is zero, so the matrix is singular.

```python
import numpy as np

A = np.array([
    [1, 2],
    [2, 4]
], dtype=float)

b = np.array([3, 6], dtype=float)

print("Rank:", np.linalg.matrix_rank(A))
print("Determinant:", np.linalg.det(A))

try:
    solution = np.linalg.solve(A, b)
    print("Solution:", solution)
except np.linalg.LinAlgError:
    print("The matrix is singular; a unique solution cannot be obtained.")
```

The system is:

```text
x + 2*y = 3
2*x + 4*y = 6
```

Both equations describe the same constraint. Therefore, infinitely many solutions exist.

A singular matrix does not always mean that the system has no solution. It means that the system does not have a unique solution when it is consistent.

**Note:** `np.linalg.solve()` is intended for square, nonsingular systems. It is not a general method for identifying every possible solution of a singular system.

### 13.4. Solving an Overdetermined System

Consider the coefficient matrix:

```text
A = [ [1, 1],
      [1, 2],
      [1, 3] ]
```

and the target vector:

```text
b = [2, 3, 5]
```

There are three equations and two unknowns.

We can calculate a least-squares solution:

```python
import numpy as np

A = np.array([
    [1, 1],
    [1, 2],
    [1, 3]
], dtype=float)

b = np.array([2, 3, 5], dtype=float)

solution, residuals, rank, singular_values = np.linalg.lstsq(
    A, b, rcond=None
)

print("Least-squares solution:")
print(solution)

print("\nResiduals:")
print(residuals)

print("\nRank:")
print(rank)

print("\nPredictions:")
print(A @ solution)
```

The least-squares solution is approximately:

```text
[0.33333333, 1.5]
```

The predictions are approximately:

```text
[1.83333333, 3.33333333, 4.83333333]
```

They do not exactly match every target value.

Instead, the solution minimizes the total squared error.

This is the basic principle behind ordinary least-squares regression.

---

## 14. Systems of Equations in Reinforcement Learning

Systems of equations have an important role in the mathematical analysis of finite Markov reward processes and Markov decision processes.

### 14.1. The Bellman Equation

Consider a fixed policy, denoted by `pi`, in a finite-state Markov reward process.

The state-value function is the expected discounted return when starting from state `s` and following policy `pi`:

```text
V_pi(s) = expected discounted return starting from state s
```

More explicitly:

```text
V_pi(s) = E_pi[
    R_1 + gamma*R_2 + gamma^2*R_3 + ...
    | S_0 = s
]
```

Here:

- `V_pi(s)` is the value of state `s` under policy `pi`.
- `R_t` is the reward at time step `t`.
- `gamma` is the discount factor.
- `E_pi` denotes expectation when following policy `pi`.

For a finite-state problem, we can collect all state values into a vector:

```text
v_pi = [V_pi(s1), V_pi(s2), ..., V_pi(sn)]
```

Let:

- `P_pi` be the state transition matrix under policy `pi`.
- `r_pi` be the expected immediate reward vector.
- `gamma` be the discount factor.

The Bellman expectation equation is:

```text
v_pi = r_pi + gamma * P_pi * v_pi
```

Rearranging:

```text
v_pi - gamma * P_pi * v_pi = r_pi
```

Factoring out the value vector:

```text
(I - gamma * P_pi) * v_pi = r_pi
```

This is a system of linear equations.

For a finite-state Markov reward process with a valid stochastic transition matrix and `0 <= gamma < 1`, the matrix `I - gamma * P_pi` is invertible.

Therefore, the value vector has a unique solution:

```text
v_pi = inverse(I - gamma * P_pi) * r_pi
```

This formula demonstrates the connection between linear algebra and policy evaluation.

In practical numerical computing, the linear system is normally solved directly rather than by explicitly calculating the inverse.

### 14.2. Worked Reinforcement Learning Example

Consider a Markov reward process with two states:

```text
S = {s1, s2}
```

The transition matrix is:

```text
P = [ [0.5, 0.5],
      [0.0, 1.0] ]
```

The reward vector is:

```text
r = [1, 2]
```

The discount factor is:

```text
gamma = 0.5
```

The Bellman equation is:

```text
v = r + 0.5 * P * v
```

Let:

```text
v = [V1, V2]
```

where `V1` is the value of state `s1` and `V2` is the value of state `s2`.

For state `s1`:

```text
V1 = 1 + 0.5*(0.5*V1 + 0.5*V2)
```

Simplify:

```text
V1 = 1 + 0.25*V1 + 0.25*V2
```

Rearrange:

```text
0.75*V1 - 0.25*V2 = 1
```

For state `s2`:

```text
V2 = 2 + 0.5*V2
```

Therefore:

```text
0.5*V2 = 2
V2 = 4
```

Substitute `V2 = 4` into the first equation:

```text
0.75*V1 - 0.25*4 = 1
0.75*V1 - 1 = 1
0.75*V1 = 2
V1 = 8/3
V1 approximately equals 2.6667
```

The final value vector is:

```text
v = [2.6667, 4.0]
```

This example demonstrates how evaluating a fixed policy can be reduced to solving a system of linear equations.

### 14.3. Python Implementation for Policy Evaluation

```python
import numpy as np

P = np.array([
    [0.5, 0.5],
    [0.0, 1.0]
], dtype=float)

r = np.array([1.0, 2.0], dtype=float)

gamma = 0.5

I = np.eye(P.shape[0])

A = I - gamma * P

v = np.linalg.solve(A, r)

print("State-value vector:")
print(v)

print("\nBellman verification:")
print(r + gamma * P @ v)
```

Expected output:

```text
State-value vector:
[2.66666667 4.        ]

Bellman verification:
[2.66666667 4.        ]
```

The code solves the Bellman expectation equation for a small finite-state Markov reward process.

**Important distinction:** This direct method evaluates a fixed policy. It does not, by itself, discover an optimal policy. Policy improvement or other decision-making procedures are needed to search for better policies.

---

## 15. Connection to Multi-Agent Reinforcement Learning and Fog Scheduling

Consider a reinforcement learning scheduler for a fog computing environment.

The scheduler observes several fog nodes and assigns tasks while considering competing objectives such as:

- Execution time.
- Energy consumption.
- Network delay.
- Resource utilization.
- Task completion rate.

Systems of equations and matrix operations can support feature processing, parameter estimation, and value-function approximation in such a system.

### 15.1. Representing the State

Suppose there are three fog nodes and three features:

1. CPU utilization.
2. Memory utilization.
3. Network delay.

A simplified feature matrix might be:

```text
X = [ [0.70, 0.60, 0.20],
      [0.40, 0.80, 0.50],
      [0.90, 0.30, 0.10] ]
```

Each row represents a fog node, and each column represents a feature.

This representation allows the scheduler to process information about multiple nodes using matrix operations.

### 15.2. Linear Scoring Model

Suppose a learned model assigns the following weights:

```text
w = [0.5, 0.3, -0.2]
```

A simple score vector can be calculated as:

```text
s = X * w
```

In NumPy, the operation is:

```python
import numpy as np

X = np.array([
    [0.70, 0.60, 0.20],
    [0.40, 0.80, 0.50],
    [0.90, 0.30, 0.10]
], dtype=float)

w = np.array([0.5, 0.3, -0.2], dtype=float)

scores = X @ w

print("Node scores:")
print(scores)
```

The resulting score for each node is a weighted combination of its features.

These scores could represent a learned value approximation or another scoring function, depending on the model design.

However, the scores do not automatically constitute an optimal scheduling policy. Their meaning depends on how the model was trained and how the action-selection procedure uses them.

### 15.3. Linear Systems in Model Estimation

Some reinforcement learning algorithms use linear systems to estimate parameters or value functions.

For example, a linear value function may be represented as:

```text
V_hat(s) = phi(s) dot w
```

Here:

- `V_hat(s)` is the estimated value of state `s`.
- `phi(s)` is a feature vector describing the state.
- `w` is a vector of learnable parameters.
- `dot` denotes the dot product.

Methods based on least squares can estimate `w` by solving an associated linear system.

If the features are redundant or insufficiently informative, the system may be singular or poorly conditioned.

This is where matrix rank and numerical stability become important.

### 15.4. Research Insight

In multi-agent reinforcement learning, equations may involve multiple agents, state-action features, shared resources, and coupled value estimates.

However, solving a linear system does not by itself resolve competing objectives.

A multi-objective scheduler must also define how to handle trade-offs, constraints, and the evaluation of different policies.

The mathematical role of the linear system is to support a particular estimation or solution step within the larger learning and decision-making process.

---

## 16. Common Mistakes

### Mistake 1: Assuming Every System Has a Unique Solution

A system may have no solution or infinitely many solutions.

**Correction:** Check consistency and matrix rank before assuming uniqueness.

### Mistake 2: Confusing a Singular Matrix with an Inconsistent System

A singular coefficient matrix does not necessarily mean that the system has no solution.

A consistent singular system has infinitely many solutions. An inconsistent system has no solution.

**Correction:** Compare the ranks of the coefficient matrix and augmented matrix.

### Mistake 3: Explicitly Inverting a Matrix Unnecessarily

Explicitly calculating a matrix inverse is often less suitable than solving the system directly.

**Correction:** For a square, nonsingular system, prefer:

```python
x = np.linalg.solve(A, b)
```

### Mistake 4: Assuming a Small Residual Guarantees an Accurate Solution

A poorly conditioned matrix can produce a solution with a small residual but a substantial error in the unknown variables.

**Correction:** Consider the condition number when numerical stability matters.

### Mistake 5: Assuming Least Squares Always Gives Unique Coefficients

A least-squares solution exists for a finite-dimensional problem, but it may not be unique if the design matrix is rank-deficient.

**Correction:** Consider the matrix rank, the Moore-Penrose pseudoinverse, or regularization.

### Mistake 6: Ignoring the Meaning of the Variables

A mathematically valid solution may not be meaningful for the application.

For example, a resource allocation model may require nonnegative allocations, capacity constraints, or integer task assignments.

**Correction:** Include the application's constraints in the mathematical model.

---

## 17. Exercises

Try to solve these exercises before reading the answers.

### Exercise 1: Substitution

Solve the following system and find `x` and `y`:

```text
x + y = 9
x - y = 3
```

### Exercise 2: Elimination

Solve:

```text
2*x + 3*y = 12
x - y = 1
```

### Exercise 3: Matrix Representation

Represent the following system in the form `A*x = b`.

```text
3*x + 2*y = 8
x + 4*y = 6
```

Identify the coefficient matrix `A`, the unknown vector `x`, and the right-hand-side vector `b`.

### Exercise 4: Unique Solution

Determine whether the following system has a unique solution. Use the determinant of the coefficient matrix to justify your answer.

```text
x + 2*y = 5
3*x + y = 7
```

### Exercise 5: Infinitely Many Solutions

Determine the number of solutions and explain why.

```text
x + 2*y = 6
2*x + 4*y = 12
```

### Exercise 6: No Solution

Determine whether the following system is consistent. Explain your reasoning.

```text
x + 2*y = 3
2*x + 4*y = 10
```

### Exercise 7: Python

Use NumPy to solve the following system and verify your result by calculating `A @ x`.

```text
2*x + y = 8
x + 3*y = 9
```

### Exercise 8: Least Squares

Suppose the coefficient matrix and target vector are:

```text
A = [ [1, 1],
      [1, 2],
      [1, 3] ]

b = [2, 3, 5]
```

Use `np.linalg.lstsq()` to find the least-squares solution.

Why might the result fail to satisfy all three equations exactly?

### Exercise 9: Reinforcement Learning

A fixed-policy Markov reward process has:

```text
P = [ [0.5, 0.5],
      [0.0, 1.0] ]

r = [1, 2]

gamma = 0.5
```

Write the Bellman equation in matrix form and calculate the value vector.

### Exercise 10: Research Thinking

Suppose two features in a fog scheduling model contain exactly the same information.

Explain how this redundancy might affect:

1. The rank of the feature matrix.
2. The uniqueness of a least-squares parameter estimate.
3. The stability of numerical calculations.
4. The design of a learning-based scheduler.

---

## 18. Exercise Answers

### Answer 1: Substitution or Elimination

Given:

```text
x + y = 9
x - y = 3
```

Add the equations:

```text
2*x = 12
x = 6
```

Substitute into the first equation:

```text
6 + y = 9
y = 3
```

Answer:

```text
x = 6
y = 3
```

### Answer 2: Elimination

Given:

```text
2*x + 3*y = 12
x - y = 1
```

From the second equation:

```text
x = y + 1
```

Substitute into the first equation:

```text
2*(y + 1) + 3*y = 12
2*y + 2 + 3*y = 12
5*y = 10
y = 2
```

Therefore:

```text
x = 2 + 1
x = 3
```

Answer:

```text
x = 3
y = 2
```

### Answer 3: Matrix Representation

Given:

```text
3*x + 2*y = 8
x + 4*y = 6
```

The coefficient matrix is:

```text
A = [ [3, 2],
      [1, 4] ]
```

The unknown vector is:

```text
x = [x, y]
```

The right-hand-side vector is:

```text
b = [8, 6]
```

Therefore:

```text
A*x = b
```

### Answer 4: Unique Solution

The coefficient matrix is:

```text
A = [ [1, 2],
      [3, 1] ]
```

Its determinant is:

```text
det(A) = 1*1 - 2*3
det(A) = -5
```

Because:

```text
det(A) != 0
```

the matrix is invertible, so the system has a unique solution.

Solving the system gives:

```text
x = 9/5 = 1.8
y = 8/5 = 1.6
```

### Answer 5: Infinitely Many Solutions

Given:

```text
x + 2*y = 6
2*x + 4*y = 12
```

The second equation is twice the first equation.

Both equations describe the same line. Therefore, infinitely many solutions exist.

### Answer 6: No Solution

Given:

```text
x + 2*y = 3
2*x + 4*y = 10
```

Multiplying the first equation by 2 gives:

```text
2*x + 4*y = 6
```

However, the second equation requires:

```text
2*x + 4*y = 10
```

These conditions contradict each other.

Therefore, the system has no solution.

### Answer 7: Python

Use the following code:

```python
import numpy as np

A = np.array([
    [2, 1],
    [1, 3]
], dtype=float)

b = np.array([8, 9], dtype=float)

solution = np.linalg.solve(A, b)

print("Solution:", solution)
print("Verification:", A @ solution)
```

The system is:

```text
2*x + y = 8
x + 3*y = 9
```

From the first equation:

```text
y = 8 - 2*x
```

Substitute into the second equation:

```text
x + 3*(8 - 2*x) = 9
x + 24 - 6*x = 9
-5*x = -15
x = 3
```

Then:

```text
y = 8 - 2*3
y = 2
```

Answer:

```text
x = 3
y = 2
```

Verification:

```text
A @ solution = [8, 9]
```

### Answer 8: Least Squares

Use:

```python
import numpy as np

A = np.array([
    [1, 1],
    [1, 2],
    [1, 3]
], dtype=float)

b = np.array([2, 3, 5], dtype=float)

solution, residuals, rank, singular_values = np.linalg.lstsq(
    A, b, rcond=None
)

print("Solution:", solution)
print("Predictions:", A @ solution)
print("Residuals:", residuals)
```

The least-squares solution is approximately:

```text
[0.33333333, 1.5]
```

The system has three equations and two unknowns. The three target values do not lie exactly on the same line represented by the model.

Consequently, no parameter vector satisfies all three equations simultaneously.

Least squares finds the parameters that minimize the sum of squared residuals.

### Answer 9: Reinforcement Learning

The Bellman equation is:

```text
v = r + gamma * P * v
```

Rearranging:

```text
(I - gamma * P) * v = r
```

For the given transition matrix and discount factor:

```text
I - gamma * P = [ [0.75, -0.25],
                  [0.00,  0.50] ]
```

The system is:

```text
0.75*V1 - 0.25*V2 = 1
0.50*V2 = 2
```

Solving gives:

```text
V2 = 4
V1 = 8/3 approximately 2.6667
```

The value vector is:

```text
v = [2.6667, 4.0]
```

### Answer 10: Research Thinking

If two features are identical, one column of the feature matrix duplicates another.

This can reduce the matrix rank because the columns are linearly dependent.

When the design matrix lacks full column rank, the least-squares coefficients may not be uniquely determined.

Redundant features can also create numerical difficulties, particularly when columns are nearly dependent rather than exactly identical.

Possible responses include:

- Removing redundant features.
- Applying dimensionality reduction.
- Using regularization.
- Selecting more informative features.
- Checking the rank and condition number of the feature matrix.

The appropriate solution depends on the scheduling model, the meaning of the features, and the learning objective.

---

## 19. Key Takeaways

- A system of equations requires all equations to be satisfied simultaneously.
- Linear systems can be solved using substitution, elimination, Gaussian elimination, or numerical solvers.
- The matrix equation `A*x = b` is fundamental to linear algebra.
- A system can have a unique solution, infinitely many solutions, or no solution.
- Matrix rank helps determine consistency and uniqueness when considered alongside the augmented matrix.
- Overdetermined systems often lead to least-squares estimation.
- Linear regression can be formulated as a least-squares problem.
- Bellman equations for finite-state policy evaluation can be represented as linear systems.
- In practical numerical computing, use suitable solvers instead of explicitly computing matrix inverses.
- Matrix operations support parameter estimation and value-function approximation in reinforcement learning.
- In fog scheduling and multi-agent reinforcement learning, linear algebra is one part of a broader solution involving objectives, constraints, and decision-making rules.

---

## 20. Further Learning

After completing this lesson, study the following topics:

1. Eigenvalues and Eigenvectors.
2. Singular Value Decomposition (SVD).
3. Linear Independence and Basis Vectors.
4. Least-Squares Estimation.
5. Gradient Descent.
6. Probability and Expected Value.
7. Markov Chains and Markov Decision Processes.
8. Bellman Equations and Dynamic Programming.
9. Linear Value-Function Approximation.
10. Multi-Agent Reinforcement Learning.

These topics build on systems of equations and provide a stronger mathematical foundation for advanced machine learning and reinforcement learning.

---

## About This Lesson

This lesson is part of the **Mathematics for Machine Learning and Reinforcement Learning** series.

The goal is to connect mathematical concepts with Python implementations and research applications in machine learning, reinforcement learning, and intelligent systems.

**Suggested next lesson:** Orthogonality and Projection.
