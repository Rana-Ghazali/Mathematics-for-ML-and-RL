# Orthogonality and Projection

## Mathematics for Machine Learning and Reinforcement Learning

**Topics:** Orthogonal Vectors, Orthogonal Sets, Orthogonal Projection, Projection Matrices, Least-Squares Approximation, and Residuals

**Level:** Beginner to Intermediate

**Prerequisites:** Vectors, Dot Product, Matrix Operations, Systems of Equations

**Programming Language:** Python

**Library:** NumPy

---

## 1. Learning Objectives

By the end of this lesson, you will be able to:

1. Define orthogonal vectors using the dot product.
2. Determine whether two vectors are orthogonal.
3. Understand orthogonal and orthonormal sets of vectors.
4. Calculate the projection of one vector onto another.
5. Interpret projection geometrically.
6. Construct a projection matrix.
7. Explain the relationship between projection and least-squares estimation.
8. Understand how orthogonality appears in linear regression and machine learning.
9. Connect projections to feature representation and dimensionality reduction.
10. Implement orthogonality and projection operations using Python and NumPy.
11. Explain selected connections to reinforcement learning.

---

## 2. Introduction: Why Orthogonality and Projection Matter

Vectors represent numerical quantities, observations, features, and model parameters.

However, understanding individual vectors is not always enough. We also need to understand the relationships between vectors.

Two particularly important concepts are:

- **Orthogonality:** A relationship between vectors that generalizes the idea of perpendicular directions.
- **Projection:** A way to extract the component of one vector that lies along another vector or within a subspace.

These concepts are important in many areas of mathematics and machine learning.

For example:

- In linear regression, predictions can be interpreted as projections onto a model's column space.
- In least-squares estimation, the residual is orthogonal to the model's column space at a least-squares solution.
- In dimensionality reduction, projections map data into lower-dimensional representations.
- In signal processing, orthogonal components help separate signals.
- In reinforcement learning, projections appear in linear function approximation and some value-estimation methods.
- In optimization, projection operations can enforce constraints on parameters.

The central idea is simple:

**Orthogonality describes perpendicular directions, while projection extracts the component of a vector along a chosen direction or subspace.**

---

## 3. Orthogonality

### 3.1. Definition

Two vectors are orthogonal if their dot product equals zero.

For two vectors, u and v, in an n-dimensional Euclidean space, the orthogonality condition is:

```text
u · v = 0
```

Equivalently, using matrix notation:

```text
u^T v = 0
```

Here, `u^T` means the transpose of vector `u`.

For nonzero vectors in ordinary Euclidean space, orthogonality means that the vectors meet at a right angle.

### Example 1: Orthogonal Vectors in Two Dimensions

Consider:

```text
u = [1, 0]

v = [0, 1]
```

Their dot product is:

```text
u · v = (1)(0) + (0)(1)
      = 0 + 0
      = 0
```

Therefore:

```text
u · v = 0
```

The vectors are orthogonal. Geometrically, they point along the horizontal and vertical axes.

### 3.2. Orthogonality in Three Dimensions

Consider:

```text
u = [1,  2, 3]

v = [2, -1, 0]
```

Calculate the dot product:

```text
u · v = (1)(2) + (2)(-1) + (3)(0)
      = 2 - 2 + 0
      = 0
```

Therefore, the vectors are orthogonal.

This definition works in any finite-dimensional Euclidean space, not just in two or three dimensions.

---

## 4. Orthogonality and the Angle Between Vectors

The dot product is related to the angle between two nonzero vectors:

```text
u · v = ||u|| * ||v|| * cos(theta)
```

where:

- `||u||` is the Euclidean norm of vector `u`.
- `||v||` is the Euclidean norm of vector `v`.
- `theta` is the angle between the vectors.

If the vectors are nonzero and orthogonal:

```text
u · v = 0
```

Therefore:

```text
cos(theta) = 0
```

Thus:

```text
theta = 90 degrees
```

Equivalently, in radians:

```text
theta = pi / 2
```

**Important note:** The angle formula requires both vectors to be nonzero. The zero vector has a dot product of zero with every vector, but its direction and angle are undefined. Therefore, it is algebraically orthogonal to every vector, although it does not have a geometric direction.

---

## 5. Orthogonal, Parallel, and Neither

Consider two nonzero vectors, `u` and `v`.

Their dot product provides information about their relative directions.

### Case 1: Orthogonal vectors

If:

```text
u · v = 0
```

the vectors are orthogonal.

### Case 2: An acute angle

If the angle between the vectors is less than 90 degrees:

```text
u · v > 0
```

The dot product is positive.

This does not necessarily mean that the vectors are parallel.

### Case 3: An obtuse angle

If the angle between the vectors is greater than 90 degrees:

```text
u · v < 0
```

The dot product is negative.

### Example

Let:

```text
u = [1, 2]

v1 = [ 2, -1]

v2 = [1, 1]

v3 = [-1, -2]
```

For `v1`:

```text
u · v1 = (1)(2) + (2)(-1)
       = 2 - 2
       = 0
```

Therefore, `u` and `v1` are orthogonal.

For `v2`:

```text
u · v2 = (1)(1) + (2)(1)
       = 3
```

The dot product is positive, so the angle is acute.

For `v3`:

```text
u · v3 = (1)(-1) + (2)(-2)
       = -1 - 4
       = -5
```

The dot product is negative, so the angle is obtuse.

---

## 6. Orthogonal Sets of Vectors

A set of vectors is orthogonal if every pair of distinct vectors in the set is orthogonal.

Consider:

```text
v1 = [1, 0, 0]

v2 = [0, 1, 0]

v3 = [0, 0, 1]
```

Their pairwise dot products are:

```text
v1 · v2 = 0

v1 · v3 = 0

v2 · v3 = 0
```

These vectors form an orthogonal set. They are the standard basis vectors of three-dimensional Euclidean space.

### 6.1. Orthonormal Vectors

A set of vectors is orthonormal if:

1. Every pair of distinct vectors is orthogonal.
2. Every vector has unit length.

In plain notation, the conditions are:

```text
vi · vj = 1    if i = j

vi · vj = 0    if i != j
```

Here, `i != j` means that the indices are different.

Equivalently:

```text
||vi|| = 1
```

for every vector in the set, and distinct vectors have a dot product of zero.

The standard basis vectors above are orthonormal because each has Euclidean norm 1.

### Why Are Orthonormal Vectors Useful?

Orthonormal vectors simplify calculations.

If `q1, q2, ..., qk` form an orthonormal basis for a subspace, the projection of a vector `u` onto that subspace is the sum of its projections onto the individual basis vectors:

```text
projection(u) =
    (u · q1)q1 + (u · q2)q2 + ... + (u · qk)qk
```

This property is useful in numerical linear algebra, signal processing, and dimensionality reduction.

---

## 7. Projection onto a Vector

### 7.1. Intuition

Suppose a vector `v` points in a particular direction.

We want to determine how much of another vector `u` lies along the direction of `v`.

The projection of `u` onto `v` gives the vector component of `u` along that direction.

We can write this as:

```text
proj_v(u)
```

The projection is parallel to `v`. The difference between the original vector and its projection is perpendicular to `v`.

### 7.2. Projection Formula

For a nonzero vector `v`, the projection of `u` onto `v` is:

```text
proj_v(u) = ((u · v) / (v · v)) * v
```

Because:

```text
v · v = ||v||^2
```

we can also write:

```text
proj_v(u) = ((u · v) / ||v||^2) * v
```

The calculation has three steps:

1. Calculate the dot product `u · v`.
2. Divide by the squared norm of `v`.
3. Multiply the resulting scalar by vector `v`.

The output is a vector in the direction of `v`.

---

## 8. Worked Example: Projecting One Vector onto Another

Let:

```text
u = [3, 4]

v = [1, 0]
```

We want to calculate `proj_v(u)`.

### Step 1: Calculate the dot product

```text
u · v = (3)(1) + (4)(0)
      = 3
```

### Step 2: Calculate the squared norm of v

```text
v · v = 1^2 + 0^2
      = 1
```

### Step 3: Apply the projection formula

```text
proj_v(u) = (3 / 1) * [1, 0]
          = [3, 0]
```

Therefore:

```text
Projection = [3, 0]
```

The projection extracts the horizontal component of the original vector.

### Step 4: Calculate the residual

The residual is the original vector minus its projection:

```text
r = u - proj_v(u)
```

Therefore:

```text
r = [3, 4] - [3, 0]
  = [0, 4]
```

Check orthogonality:

```text
r · v = (0)(1) + (4)(0)
      = 0
```

The residual is orthogonal to the projection direction.

---

## 9. Another Worked Example

Let:

```text
u = [3, 4]

v = [2, 1]
```

Calculate `proj_v(u)`.

### Step 1: Dot product

```text
u · v = (3)(2) + (4)(1)
      = 10
```

### Step 2: Squared norm

```text
v · v = 2^2 + 1^2
      = 5
```

### Step 3: Projection

```text
proj_v(u) = (10 / 5) * [2, 1]
          = 2 * [2, 1]
          = [4, 2]
```

Therefore:

```text
Projection = [4, 2]
```

### Step 4: Residual

```text
r = u - proj_v(u)
  = [3, 4] - [4, 2]
  = [-1, 2]
```

Check orthogonality:

```text
r · v = (-1)(2) + (2)(1)
      = -2 + 2
      = 0
```

The residual is orthogonal to the projection direction.

---

## 10. Scalar Projection versus Vector Projection

There are two related concepts.

### 10.1. Scalar Projection

The scalar projection, also called the signed component of `u` along `v`, is:

```text
scalar_proj_v(u) = (u · v) / ||v||
```

It gives the signed length of the component of `u` along the direction of `v`.

The scalar projection can be positive, negative, or zero.

### 10.2. Vector Projection

The vector projection is:

```text
vector_proj_v(u) = ((u · v) / ||v||^2) * v
```

It provides both magnitude and direction.

| Concept | Output | Meaning |
|---|---|---|
| Scalar projection | A number | Signed component length |
| Vector projection | A vector | Component in the direction of `v` |

Do not confuse these two quantities.

---

## 11. Projection onto a Subspace

Projecting onto one vector is a special case of projecting onto a subspace.

A subspace is a set of vectors closed under vector addition and scalar multiplication.

Suppose a subspace `S` is spanned by vectors:

```text
v1, v2, ..., vk
```

We want to find the vector in `S` that is closest to a given vector `u`.

The orthogonal projection of `u` onto `S` is the vector `p` in `S` that minimizes:

```text
||u - p||^2
```

The difference:

```text
r = u - p
```

is called the residual.

For an orthogonal projection, the residual is orthogonal to every vector in the subspace:

```text
r is orthogonal to S
```

Equivalently, for every vector `v` in `S`:

```text
v · (u - p) = 0
```

This is the geometric foundation of least-squares estimation.

---

## 12. Projection onto a Subspace Using a Matrix

Suppose the columns of matrix `A` span a subspace:

```text
S = Col(A)
```

Here, `Col(A)` denotes the column space of `A`.

Let:

```text
A has dimensions m x n

b has m components
```

We want to project vector `b` onto the column space of `A`.

We seek a vector:

```text
p = A*x_hat
```

that is closest to `b`.

The least-squares problem is:

```text
Minimize ||A*x - b||^2
```

At the least-squares solution, the residual is:

```text
r = b - A*x_hat
```

The residual is orthogonal to the columns of `A`. Therefore:

```text
A^T * (b - A*x_hat) = 0
```

Expanding this expression gives:

```text
A^T*b - A^T*A*x_hat = 0
```

Rearranging:

```text
A^T*A*x_hat = A^T*b
```

These equations are called the **normal equations**.

If the columns of `A` are linearly independent, then `A^T*A` is invertible, and:

```text
x_hat = inverse(A^T*A) * A^T * b
```

The projected vector is:

```text
p = A*x_hat
```

Therefore:

```text
p = A * inverse(A^T*A) * A^T * b
```

This leads to the projection matrix.

**Numerical note:** The normal-equation formula is useful for mathematical understanding. For numerical computations, QR decomposition or SVD-based least-squares solvers are often preferable to explicitly forming the inverse of `A^T*A`.

---

## 13. The Projection Matrix

If the columns of `A` are linearly independent, the orthogonal projection matrix onto the column space of `A` is:

```text
P = A * inverse(A^T*A) * A^T
```

The projection of vector `b` is:

```text
p = P*b
```

The projection matrix transforms any vector into its orthogonal projection onto the column space of `A`.

### 13.1. Important Properties of an Orthogonal Projection Matrix

An orthogonal projection matrix satisfies several important properties.

**Property 1: Symmetry**

```text
P^T = P
```

**Property 2: Idempotence**

```text
P^2 = P
```

Idempotence means that projecting an already projected vector does not change it.

If:

```text
p = P*b
```

then:

```text
P*p = P*(P*b)
    = P^2*b
    = P*b
    = p
```

**Property 3: Projection onto the Column Space**

For every vector `b`, the vector `P*b` belongs to the column space of `A`.

**Property 4: Orthogonal Residual**

The residual is:

```text
r = b - P*b
```

It is orthogonal to the column space of `A`. Therefore:

```text
A^T*r = 0
```

These properties are useful for understanding least squares and orthogonal decomposition.

If the columns of `A` are not linearly independent, the formula involving the inverse of `A^T*A` is not available. The orthogonal projection onto the column space still exists and can be computed using an orthonormal basis or the Moore-Penrose pseudoinverse.

---

## 14. Worked Example: Projection onto a Subspace

Consider the matrix:

```text
A = [[1],
     [1]]
```

Its column space is the line:

```text
S = span([1, 1])
```

Let:

```text
b = [3, 1]
```

We want to project `b` onto `S`.

### Step 1: Calculate A^T*A

```text
A^T = [[1, 1]]
```

Therefore:

```text
A^T*A = [[1, 1]] * [[1],
                    [1]]

       = [[2]]
```

Thus:

```text
inverse(A^T*A) = [[0.5]]
```

### Step 2: Calculate the projection matrix

```text
P = A * inverse(A^T*A) * A^T
```

Substituting the values:

```text
P = [[1],
     [1]] * [[0.5]] * [[1, 1]]
```

Therefore:

```text
P = [[0.5, 0.5],
     [0.5, 0.5]]
```

### Step 3: Project the vector

```text
p = P*b
```

Therefore:

```text
p = [[0.5, 0.5],     [[3],
     [0.5, 0.5]]  *   [1]]

  = [2, 2]
```

The projection lies on the line `y = x`.

### Step 4: Calculate the residual

```text
r = b - p
  = [3, 1] - [2, 2]
  = [1, -1]
```

Check orthogonality:

```text
[1, 1] · [1, -1]
= (1)(1) + (1)(-1)
= 0
```

The residual is orthogonal to the subspace.

---

## 15. Orthogonality and Linear Regression

Orthogonality is central to the geometry of ordinary least-squares regression.

Suppose:

```text
X has dimensions m x n

y has m components
```

Here, `X` is the design matrix and `y` is the target vector.

Linear regression seeks parameters `w_hat` that minimize:

```text
||X*w - y||^2
```

The prediction vector is:

```text
y_hat = X*w_hat
```

The residual vector is:

```text
e = y - y_hat
```

At a least-squares minimizer:

```text
X^T*e = 0
```

This means that the residual is orthogonal to every column of the design matrix.

The prediction vector is the orthogonal projection of `y` onto the column space of `X`.

This geometric interpretation explains the normal equations:

```text
X^T*X*w_hat = X^T*y
```

When `X` has full column rank, the least-squares coefficient solution is unique.

When `X` is rank-deficient, the least-squares prediction vector remains unique, but the coefficient vector may not be unique.

This distinction is important when interpreting regression models with redundant features.

---

## 16. Orthogonality in Machine Learning

### 16.1. Feature Representation

A data point can be represented as a vector:

```text
x = [x1, x2, ..., xn]
```

Different feature directions may represent different aspects of the data.

Orthogonal feature vectors are independent as directions in Euclidean space. However, orthogonal feature vectors do not automatically imply that the corresponding random variables are statistically independent.

Statistical independence is a stronger concept.

### 16.2. Dimensionality Reduction

Projection can map a high-dimensional data point into a lower-dimensional representation.

Suppose the columns of matrix `Q` form an orthonormal basis for a subspace.

The coordinates of vector `x` in that basis are:

```text
z = Q^T*x
```

The reconstruction from those coordinates is:

```text
x_hat = Q*z
```

Therefore:

```text
x_hat = Q*Q^T*x
```

If `Q` has fewer columns than rows, this representation can reduce dimensionality.

When `Q` contains selected principal components from PCA, projection onto those components produces a lower-dimensional representation of the data.

### 16.3. PCA and Orthogonality

Principal Component Analysis (PCA) identifies orthogonal directions that capture successive amounts of variance in a dataset.

The principal component directions are eigenvectors of the covariance matrix. Because a real covariance matrix is symmetric, its eigenvectors can be chosen to be orthonormal.

Projecting observations onto selected principal components reduces the number of dimensions while retaining as much variance as possible for the chosen number of components.

Orthogonality is important because it gives the component directions a clear geometric interpretation.

However, PCA is not the only dimensionality-reduction technique, and not every low-dimensional representation is based on orthogonal projection.

### 16.4. Orthogonality and Numerical Linear Algebra

QR decomposition represents a matrix as:

```text
A = Q*R
```

where:

- The columns of `Q` are orthonormal.
- `R` is an upper-triangular matrix.

QR decomposition is useful for solving least-squares problems and is often numerically preferable to directly solving the normal equations.

Singular Value Decomposition (SVD) also provides orthonormal directions and is widely used for dimensionality reduction and numerical analysis.

---

## 17. Orthogonality and Reinforcement Learning

Orthogonality and projection also appear in reinforcement learning, especially in function approximation and policy evaluation.

### 17.1. Linear Value-Function Approximation

Suppose the value function is approximated using features:

```text
V_hat(s) = phi(s)^T*w
```

where:

- `phi(s)` is a feature vector describing state `s`.
- `w` is a parameter vector.

For a collection of states, the approximated values can be written as:

```text
v_hat = Phi*w
```

where `Phi` is the feature matrix.

The vector `v_hat` belongs to the column space of `Phi`.

Therefore, linear value-function approximation restricts the value estimates to a particular subspace.

This is a geometric interpretation of linear function approximation.

### 17.2. Least-Squares Value Estimation

Suppose we want to approximate a target vector `y` using feature matrix `Phi`.

A least-squares estimate solves:

```text
Minimize ||Phi*w - y||^2
```

The estimated value vector is:

```text
v_hat = Phi*w_hat
```

The residual is:

```text
e = y - v_hat
```

At the least-squares solution, the residual is orthogonal to the column space of `Phi`:

```text
Phi^T*e = 0
```

This explains the role of orthogonality in linear least-squares estimation.

### 17.3. Projection in Temporal-Difference Learning

Some theoretical formulations of temporal-difference learning can be expressed using projected Bellman equations.

A projected Bellman equation may be written conceptually as:

```text
v_hat = Pi*T_pi(v_hat)
```

where:

- `T_pi` is the Bellman operator for a fixed policy.
- `Pi` is a projection operator onto a chosen approximation space.
- `v_hat` is an approximate value vector.

The projection keeps the value estimate within the chosen function-approximation space.

The exact definition of the projection depends on the feature representation and the weighting or norm used in the formulation.

This distinction matters: projections in reinforcement learning are not always ordinary Euclidean projections, and projected Bellman equations are not identical to simply solving an unprojected Bellman equation.

### 17.4. Why This Matters for Research

In a reinforcement learning scheduler for fog computing, the state representation might contain:

- CPU utilization.
- Memory utilization.
- Network delay.
- Queue length.
- Energy consumption.

A value function may use these features to estimate the long-term utility of a state or action.

Projection and least-squares methods can help explain how an approximate model represents information within a limited feature space.

Orthogonality can also be useful in constructing numerically stable bases and analyzing estimation errors.

However, orthogonal features are not automatically optimal for RL. Their usefulness depends on the environment, objective, learning algorithm, and feature representation.

---

## 18. Python Implementation Using NumPy

We will implement:

1. Testing orthogonality.
2. Vector projection.
3. Residual verification.
4. Projection onto a subspace.
5. Projection matrix properties.

### 18.1. Import NumPy

```python
import numpy as np
```

### 18.2. Check Whether Two Vectors Are Orthogonal

```python
import numpy as np

u = np.array([1.0, 2.0, 3.0])
v = np.array([2.0, -1.0, 0.0])

dot_product = np.dot(u, v)

print("Dot product:", dot_product)

if np.isclose(dot_product, 0.0):
    print("The vectors are orthogonal.")
else:
    print("The vectors are not orthogonal.")
```

Expected output:

```text
Dot product: 0.0
The vectors are orthogonal.
```

For floating-point vectors, use a numerical tolerance rather than testing exact equality with zero.

### 18.3. Calculate Vector Projection

```python
import numpy as np


def project_vector(u, v):
    """
    Return the vector projection of u onto v.

    Both inputs must be one-dimensional vectors of equal length.
    The projection direction v must be nonzero.
    """

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if u.ndim != 1 or v.ndim != 1:
        raise ValueError("Both inputs must be one-dimensional vectors.")

    if u.shape != v.shape:
        raise ValueError("The vectors must have the same dimensions.")

    denominator = np.dot(v, v)

    if np.isclose(denominator, 0.0):
        raise ValueError("Cannot project onto the zero vector.")

    return (np.dot(u, v) / denominator) * v


u = np.array([3.0, 4.0])
v = np.array([2.0, 1.0])

projection = project_vector(u, v)
residual = u - projection

print("Original vector:", u)
print("Projection:", projection)
print("Residual:", residual)
print("Residual dot projection direction:", np.dot(residual, v))
```

Expected output:

```text
Original vector: [3. 4.]
Projection: [4. 2.]
Residual: [-1.  2.]
Residual dot projection direction: 0.0
```

This confirms that the residual is orthogonal to the projection direction.

### 18.4. Projection onto a Subspace

Suppose the columns of `A` span the subspace onto which we want to project vector `b`.

A numerically preferable approach is to calculate an orthonormal basis using QR decomposition.

This implementation assumes that `A` has linearly independent columns.

```python
import numpy as np


def project_onto_column_space(A, b):
    """
    Project b onto the column space of A using QR decomposition.

    Assumes that A has linearly independent columns.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix.")

    if b.ndim != 1:
        raise ValueError("b must be a one-dimensional vector.")

    if A.shape[0] != b.shape[0]:
        raise ValueError(
            "The number of rows in A must equal the length of b."
        )

    if A.shape[1] == 0:
        return np.zeros_like(b)

    if np.linalg.matrix_rank(A) < A.shape[1]:
        raise ValueError(
            "A must have linearly independent columns."
        )

    Q, R = np.linalg.qr(A, mode="reduced")

    projection = Q @ (Q.T @ b)

    return projection


A = np.array([
    [1.0],
    [1.0]
])

b = np.array([3.0, 1.0])

p = project_onto_column_space(A, b)
residual = b - p

print("Projection:", p)
print("Residual:", residual)
print("Orthogonality check:", A.T @ residual)
```

Expected output:

```text
Projection: [2. 2.]
Residual: [ 1. -1.]
Orthogonality check: [[0.]]
```

The implementation uses an orthonormal basis rather than explicitly forming the inverse-based projection matrix.

For rank-deficient matrices, an SVD-based method can be used to identify the effective column space.

### 18.5. Verify Projection Matrix Properties

```python
import numpy as np

A = np.array([
    [1.0],
    [1.0]
])

Q, R = np.linalg.qr(A, mode="reduced")

P = Q @ Q.T

print("Projection matrix:")
print(P)

print("\nIs the matrix symmetric?")
print(np.allclose(P.T, P))

print("\nIs the matrix idempotent?")
print(np.allclose(P @ P, P))
```

Expected output:

```text
Projection matrix:
[[0.5 0.5]
 [0.5 0.5]]

Is the matrix symmetric?
True

Is the matrix idempotent?
True
```

The results confirm two important properties:

```text
P^T = P

P^2 = P
```

---

## 19. Common Mistakes

### Mistake 1: Confusing Orthogonality with Independence

Orthogonal nonzero vectors are linearly independent.

However, linearly independent vectors are not necessarily orthogonal.

### Mistake 2: Forgetting the Squared Norm in Vector Projection

The correct formula is:

```text
proj_v(u) = ((u · v) / (v · v)) * v
```

The denominator is the squared norm of the projection direction.

### Mistake 3: Confusing Scalar and Vector Projections

The scalar projection is a number.

The vector projection is a vector.

They are related but are not the same quantity.

### Mistake 4: Projecting onto the Zero Vector

The projection formula requires a nonzero projection direction. The denominator would be zero if the direction vector were the zero vector.

### Mistake 5: Assuming the Residual Is Always Zero

Projection generally produces an approximation.

The residual is zero only if the original vector already belongs to the subspace.

Otherwise, the residual measures the component that cannot be represented within the chosen subspace.

### Mistake 6: Assuming Every Projection Uses the Euclidean Inner Product

The formulas in this lesson use the standard Euclidean inner product.

Some applications use weighted inner products or other geometries, leading to different projection formulas.

This distinction is important in advanced numerical methods and some reinforcement learning formulations.

---

## 20. Exercises

Try to solve these exercises before checking the answers.

### Exercise 1: Orthogonality

Determine whether the following vectors are orthogonal. Show your calculations.

```text
u = [2, 3]

v = [3, -2]
```

### Exercise 2: Orthonormal Vectors

Determine whether the following vectors form an orthonormal set. Explain why.

```text
v1 = [1, 0]

v2 = [0, 1]
```

### Exercise 3: Vector Projection

Calculate the projection of:

```text
u = [4, 3]
```

onto:

```text
v = [1, 2]
```

Then calculate the residual and verify orthogonality.

### Exercise 4: Projection onto a Subspace

Let:

```text
A = [[1],
     [1]]

b = [5, 1]
```

Calculate the projection of `b` onto the column space of `A`.

### Exercise 5: Projection Matrix

For:

```text
A = [[1],
     [1]]
```

Calculate:

```text
P = A * inverse(A^T*A) * A^T
```

Verify that:

```text
P^T = P

P^2 = P
```

### Exercise 6: Linear Regression

Explain why the residual vector in ordinary least-squares regression is orthogonal to the columns of the design matrix.

Write the corresponding matrix equation.

### Exercise 7: Python Implementation

Use NumPy to:

1. Define two vectors.
2. Calculate their dot product.
3. Determine whether they are orthogonal.
4. Calculate the projection of one vector onto the other.
5. Verify that the residual is orthogonal to the projection direction.

### Exercise 8: Reinforcement Learning

Suppose a linear value function is:

```text
V_hat(s) = phi(s)^T * w
```

Explain:

1. What the feature vector represents.
2. Why the predicted value vector belongs to the column space of the feature matrix.
3. How projection can be used to approximate a target value vector.
4. Why the least-squares residual is orthogonal to the feature space.

### Exercise 9: Research Thinking

Consider a fog computing scheduler that uses CPU utilization, memory utilization, and network delay as features.

Discuss how projection could be used to:

1. Represent observations in a lower-dimensional space.
2. Approximate a target vector of observed or estimated values.
3. Analyze the residual between actual and predicted values.
4. Identify limitations of linear projection when the relationship between features and scheduling outcomes is nonlinear.

---

## 21. Exercise Answers

### Answer 1: Orthogonality

Calculate the dot product:

```text
u · v = (2)(3) + (3)(-2)
      = 6 - 6
      = 0
```

Therefore, `u` and `v` are orthogonal.

### Answer 2: Orthonormal Vectors

First, calculate the dot product:

```text
v1 · v2 = 0
```

Thus, the vectors are orthogonal.

Their norms are:

```text
||v1|| = 1

||v2|| = 1
```

Both vectors have unit length. Therefore, they form an orthonormal set.

### Answer 3: Vector Projection

Given:

```text
u = [4, 3]

v = [1, 2]
```

Calculate the dot product:

```text
u · v = (4)(1) + (3)(2)
      = 10
```

The squared norm of `v` is:

```text
v · v = 1^2 + 2^2
      = 5
```

Therefore:

```text
proj_v(u) = (10 / 5) * [1, 2]
          = [2, 4]
```

The residual is:

```text
r = u - proj_v(u)
  = [4, 3] - [2, 4]
  = [2, -1]
```

Verify:

```text
r · v = (2)(1) + (-1)(2)
      = 0
```

The residual is orthogonal to the projection direction.

### Answer 4: Projection onto a Subspace

Given:

```text
A = [[1],
     [1]]

b = [5, 1]
```

The projection matrix is:

```text
P = [[0.5, 0.5],
     [0.5, 0.5]]
```

Therefore:

```text
p = P*b
  = [3, 3]
```

The projection is:

```text
p = [3, 3]
```

### Answer 5: Projection Matrix

First:

```text
A^T*A = [[2]]
```

Therefore:

```text
inverse(A^T*A) = [[0.5]]
```

The projection matrix is:

```text
P = [[0.5, 0.5],
     [0.5, 0.5]]
```

It is symmetric:

```text
P^T = P
```

Multiplying `P` by itself gives:

```text
P^2 = [[0.5, 0.5],
       [0.5, 0.5]]
     = P
```

Thus:

```text
P^T = P

P^2 = P
```

### Answer 6: Linear Regression

The least-squares solution minimizes:

```text
||X*w - y||^2
```

At a least-squares minimizer:

```text
X^T*X*w_hat = X^T*y
```

The residual is:

```text
e = y - X*w_hat
```

Therefore:

```text
X^T*e = X^T*y - X^T*X*w_hat
      = 0
```

Hence:

```text
X^T*e = 0
```

The residual is orthogonal to the column space of `X`.

### Answer 7: Python Implementation

One possible implementation is:

```python
import numpy as np

u = np.array([4.0, 3.0])
v = np.array([1.0, 2.0])

dot_product = np.dot(u, v)

print("Dot product:", dot_product)
print("Orthogonal:", np.isclose(dot_product, 0.0))

projection = (dot_product / np.dot(v, v)) * v

residual = u - projection

print("Projection:", projection)
print("Residual:", residual)
print("Residual dot v:", np.dot(residual, v))
```

The output should confirm that the residual is orthogonal to the projection direction.

### Answer 8: Reinforcement Learning

1. The feature vector describes the numerical characteristics of a state.
2. For multiple states, predicted values have the form `v_hat = Phi*w`, so they belong to the column space of `Phi`.
3. Projection finds the vector in the feature space that is closest to the target vector under the chosen norm.
4. At the least-squares solution, the residual is orthogonal to the feature space, expressed as `Phi^T*e = 0`.

### Answer 9: Research Thinking

Possible answers include:

1. Projection onto a lower-dimensional subspace can reduce the number of features used by the scheduler.
2. A linear model can approximate target values using the feature matrix and learned parameters.
3. Residual analysis can help quantify approximation error and reveal patterns that the model fails to capture.
4. If the relationship between resource features and scheduling outcomes is nonlinear, a simple linear projection may not preserve the information needed for accurate predictions. Nonlinear feature transformations or more expressive models may be necessary.

---

## 22. Key Takeaways

- Two vectors are orthogonal when their dot product is zero.
- Orthonormal vectors are orthogonal vectors with unit length.
- The projection of a vector onto a nonzero vector extracts its component in that direction.
- The residual after orthogonal projection is perpendicular to the projection subspace.
- Projection onto a subspace can be expressed using a projection matrix.
- Orthogonal projection matrices are symmetric and idempotent.
- Ordinary least-squares predictions are projections onto the column space of the design matrix.
- The least-squares residual is orthogonal to that column space.
- Projection and orthogonality are important in PCA, QR decomposition, and linear function approximation.
- In reinforcement learning, projections help explain how approximate value functions represent target values within a restricted feature space.
- Numerical implementations should use stable methods such as QR decomposition or SVD when appropriate.

---

## 23. Further Learning

After completing this lesson, study:

1. Eigenvalues and Eigenvectors.
2. Singular Value Decomposition (SVD).
3. QR Decomposition.
4. Least-Squares Estimation.
5. Principal Component Analysis (PCA).
6. Derivatives and Gradients.
7. Gradient Descent.
8. Probability and Expected Value.
9. Markov Decision Processes.
10. Linear Function Approximation in Reinforcement Learning.

These topics extend the ideas of orthogonality and projection into numerical optimization, dimensionality reduction, and reinforcement learning.

---

## About This Lesson

This lesson is part of the **Mathematics for Machine Learning and Reinforcement Learning** series.

Its purpose is to connect mathematical foundations with practical Python implementations and research applications in machine learning, reinforcement learning, and intelligent systems.

**Suggested next lesson:** Eigenvalues, Eigenvectors.
