# Matrix Operations for Machine Learning and Reinforcement Learning

## Matrix Multiplication, Determinants, Inverse Matrices, and Matrix Rank

**Level:** Beginner to Intermediate  
**Prerequisites:** Vectors, Dot Product, Basic Algebra, Python  
**Programming Language:** Python  
**Library:** NumPy  
**Applications:** Machine Learning, Deep Learning, Reinforcement Learning, Optimization, and Intelligent Systems

---

## 1. Learning Objectives

By the end of this lesson, you will be able to:

1. Understand the mathematical structure of matrices.
2. Multiply matrices and determine when multiplication is valid.
3. Explain the relationship between matrix multiplication and linear transformations.
4. Calculate the determinant of a square matrix.
5. Determine whether a matrix is invertible.
6. Calculate the inverse of a matrix when it exists.
7. Understand the concept of matrix rank.
8. Implement matrix operations using Python and NumPy.
9. Explain how matrices are used in Machine Learning and Reinforcement Learning.
10. Understand why solving linear systems is generally preferable to explicitly calculating matrix inverses in numerical applications.

---

## 2. Introduction: What Is a Matrix?

A matrix is a rectangular arrangement of numbers organized into rows and columns.

For example, consider matrix A:

```text
A = [ 1  2 ]
    [ 3  4 ]
```

This matrix has 2 rows and 2 columns.

Its dimensions are:

```text
2 × 2
```

Another example:

```text
B = [ 1  2  3 ]
    [ 4  5  6 ]
```

Matrix B has 2 rows and 3 columns. Its dimensions are:

```text
2 × 3
```

In general, a matrix with m rows and n columns has dimensions:

```text
m × n
```

Matrices are fundamental to Machine Learning because they provide an efficient way to represent and process numerical data.

For example, a dataset containing five observations and three features can be represented as:

```text
X = [ x11  x12  x13 ]
    [ x21  x22  x23 ]
    [ x31  x32  x33 ]
    [ x41  x42  x43 ]
    [ x51  x52  x53 ]
```

Here:

- Each row represents one observation.
- Each column represents one feature.
- xij represents the value in row i and column j.

In Deep Learning, matrices represent datasets, model parameters, intermediate activations, and transformations between layers.

In Reinforcement Learning, matrices can represent transition probabilities, state-feature transformations, and other components of mathematical models.

This lesson focuses on four important concepts:

1. Matrix multiplication
2. Determinants
3. Inverse matrices
4. Matrix rank

---

## 3. Matrix Multiplication

### 3.1. Definition

Matrix multiplication combines two matrices to produce another matrix.

Suppose:

```text
A has dimensions m × n
B has dimensions n × p
```

Then the product AB is defined and has dimensions:

```text
(m × n) × (n × p) = (m × p)
```

The number of columns in A must equal the number of rows in B.

**Example: Valid multiplication**

```text
A: 2 × 3
B: 3 × 2

AB: 2 × 2
```

The multiplication is valid because the inner dimensions are both 3.

**Example: Invalid multiplication**

```text
A: 2 × 3
B: 2 × 2
```

AB is undefined because A has 3 columns, while B has 2 rows.

Remember:

```text
(m × n) × (n × p) = (m × p)
```

### 3.2. The Mathematical Formula

Suppose C = AB.

Each element of C is calculated by multiplying the corresponding entries of a row of A and a column of B, then adding the products.

Using index notation:

```text
c[i,j] = sum of a[i,k] × b[k,j]
          for k = 1, 2, ..., n
```

In other words, each element of the result is the dot product of one row of A and one column of B.

### 3.3. Worked Example: Multiplying Two 2 × 2 Matrices

Consider:

```text
A = [ 1  2 ]
    [ 3  4 ]

B = [ 5  6 ]
    [ 7  8 ]
```

We want to calculate:

```text
C = AB
```

The result is:

```text
C = [ c11  c12 ]
    [ c21  c22 ]
```

**Step 1: Calculate c11**

Multiply the first row of A by the first column of B:

```text
c11 = 1 × 5 + 2 × 7
    = 5 + 14
    = 19
```

**Step 2: Calculate c12**

```text
c12 = 1 × 6 + 2 × 8
    = 6 + 16
    = 22
```

**Step 3: Calculate c21**

```text
c21 = 3 × 5 + 4 × 7
    = 15 + 28
    = 43
```

**Step 4: Calculate c22**

```text
c22 = 3 × 6 + 4 × 8
    = 18 + 32
    = 50
```

Therefore:

```text
AB = [ 19  22 ]
     [ 43  50 ]
```

Each output element is a dot product between one row and one column.

---

## 4. Matrix Multiplication vs. Element-Wise Multiplication

Matrix multiplication is different from element-wise multiplication.

Consider:

```text
A = [ 1  2 ]
    [ 3  4 ]

B = [ 5  6 ]
    [ 7  8 ]
```

The matrix product is:

```text
AB = [ 19  22 ]
     [ 43  50 ]
```

Element-wise multiplication multiplies corresponding entries:

```text
A * B = [ 1 × 5  2 × 6 ]
        [ 3 × 7  4 × 8 ]
```

Therefore:

```text
A * B = [  5  12 ]
        [ 21  32 ]
```

These operations are different.

In NumPy:

- `A @ B` performs matrix multiplication.
- `A * B` performs element-wise multiplication for compatible array shapes.

Understanding this distinction is essential when implementing Machine Learning algorithms.

---

## 5. Important Properties of Matrix Multiplication

### 5.1. Matrix Multiplication Is Not Generally Commutative

For ordinary numbers:

```text
2 × 3 = 3 × 2
```

However, matrix multiplication generally satisfies:

```text
AB ≠ BA
```

Using the previous matrices:

```text
AB = [ 19  22 ]
     [ 43  50 ]
```

Reversing the order gives:

```text
BA = [ 23  34 ]
     [ 31  46 ]
```

Therefore:

```text
AB ≠ BA
```

Sometimes BA is not even defined because the dimensions are incompatible.

### 5.2. Associativity

Matrix multiplication is associative when the dimensions are compatible:

```text
(AB)C = A(BC)
```

This property allows matrix products to be regrouped without changing the mathematical result.

However, the computational cost can depend on the order of multiplication.

### 5.3. Distributivity

Matrix multiplication distributes over matrix addition:

```text
A(B + C) = AB + AC
```

Similarly:

```text
(A + B)C = AC + BC
```

These identities hold when the dimensions are compatible.

### 5.4. Identity Matrix

The identity matrix contains ones on its main diagonal and zeros elsewhere.

For example:

```text
I = [ 1  0 ]
    [ 0  1 ]
```

For a compatible matrix A:

```text
AI = IA = A
```

The identity matrix plays a role similar to the number 1 in ordinary multiplication.

---

## 6. Matrix Multiplication in Machine Learning

### 6.1. Linear Models

A linear model can be written as:

```text
y_hat = Xw + b
```

where:

- X is the feature matrix.
- w is the vector of model weights.
- b is the bias term or bias vector.
- y_hat is the vector of predictions.

Suppose:

```text
X = [ 1  2 ]
    [ 3  4 ]
    [ 5  6 ]

w = [  2 ]
    [ -1 ]
```

Calculate Xw:

```text
Xw = [ 1 × 2 + 2 × (-1) ]
     [ 3 × 2 + 4 × (-1) ]
     [ 5 × 2 + 6 × (-1) ]
```

Therefore:

```text
Xw = [ 0 ]
     [ 2 ]
     [ 4 ]
```

Matrix multiplication calculates predictions for multiple observations simultaneously.

For a general linear model, the bias term must have a shape compatible with the prediction vector.

### 6.2. Neural Networks

A fully connected neural network layer can be expressed as:

```text
H = f(XW + b)
```

where:

- X is the input matrix.
- W is the weight matrix.
- b is the bias vector.
- f is an activation function.
- H is the output activation matrix.

Matrix multiplication allows neural networks to process many observations and neurons efficiently.

Modern Deep Learning frameworks use optimized matrix operations on CPUs and GPUs.

### 6.3. Transformer Attention

Transformer architectures use matrix multiplication to calculate attention scores.

The scaled dot-product attention operation is:

```text
Attention(Q, K, V) =
    softmax((Q × transpose(K)) / sqrt(d_k)) × V
```

where:

- Q represents query vectors.
- K represents key vectors.
- V represents value vectors.
- d_k is the dimension of each key vector.

The product:

```text
Q × transpose(K)
```

calculates query-key compatibility scores.

The softmax function converts these scores into attention weights. The final multiplication combines the value vectors using those weights.

This is one reason matrix multiplication is central to modern language models.

---

## 7. Matrix Multiplication in Reinforcement Learning

### 7.1. Transition Probability Matrix

Consider an environment with three states:

```text
S = {s1, s2, s3}
```

Suppose the transition matrix under a fixed policy is:

```text
P = [ 0.5  0.5  0.0 ]
    [ 0.0  0.5  0.5 ]
    [ 0.2  0.0  0.8 ]
```

Each row represents the current state, and each column represents the next state.

For example:

```text
P[1,2] = 0.5
```

means that the probability of transitioning from state s1 to state s2 is 0.5 under the specified policy.

Every row sums to 1.

Suppose the current state distribution is:

```text
mu = [ 1  0  0 ]
```

The next-state distribution is:

```text
mu_next = mu × P
```

Therefore:

```text
mu_next = [ 1  0  0 ] × P

mu_next = [ 0.5  0.5  0.0 ]
```

After one transition, the agent has a 0.5 probability of being in s1, a 0.5 probability of being in s2, and zero probability of being in s3.

### 7.2. Bellman Equation

For a fixed policy in a discounted Markov reward process, the value function satisfies:

```text
v = r + gamma × P × v
```

where:

- v is the state-value vector.
- r is the expected immediate reward vector.
- P is the transition probability matrix.
- gamma is the discount factor, with 0 ≤ gamma < 1.

Rearranging:

```text
v - gamma × P × v = r
```

Therefore:

```text
(I - gamma × P) × v = r
```

This is a system of linear equations.

For a finite-state process with a valid stochastic transition matrix and 0 ≤ gamma < 1, the matrix `I - gamma × P` is invertible.

Thus, the value function can be expressed mathematically as:

```text
v = inverse(I - gamma × P) × r
```

This formula connects matrix operations with exact policy evaluation.

**Practical note:** Numerical implementations generally solve the linear system directly rather than explicitly computing the inverse.

---

## 8. Determinants

### 8.1. Definition

The determinant is a scalar associated with a square matrix.

It provides information about invertibility and the scaling of areas or volumes under the associated linear transformation.

The determinant of matrix A is written as:

```text
det(A)
```

A determinant is defined only for square matrices.

### 8.2. Determinant of a 2 × 2 Matrix

Consider:

```text
A = [ a  b ]
    [ c  d ]
```

The determinant is:

```text
det(A) = a × d - b × c
```

**Worked example**

```text
A = [ 3  2 ]
    [ 1  4 ]
```

Calculate:

```text
det(A) = 3 × 4 - 2 × 1
       = 12 - 2
       = 10
```

Therefore:

```text
det(A) = 10
```

Since the determinant is nonzero, the matrix is invertible.

### 8.3. Determinant of a 3 × 3 Matrix

Consider:

```text
A = [ a  b  c ]
    [ d  e  f ]
    [ g  h  i ]
```

Expanding along the first row:

```text
det(A) = a × (e × i - f × h)
       - b × (d × i - f × g)
       + c × (d × h - e × g)
```

**Worked example**

```text
A = [ 1  2  3 ]
    [ 0  4  5 ]
    [ 1  0  6 ]
```

Expanding along the first row:

```text
det(A) = 1 × (4 × 6 - 5 × 0)
       - 2 × (0 × 6 - 5 × 1)
       + 3 × (0 × 0 - 4 × 1)
```

Simplifying:

```text
det(A) = 24 - 2 × (-5) + 3 × (-4)
       = 24 + 10 - 12
       = 22
```

Therefore:

```text
det(A) = 22
```

Because the determinant is nonzero, the matrix is invertible.

---

## 9. Geometric Interpretation of the Determinant

A determinant describes how a square matrix transforms area or volume.

In two dimensions, the absolute value of the determinant gives the area-scaling factor.

For example:

```text
abs(det(A)) = 3
```

means that areas are multiplied by a factor of 3 under the transformation represented by A.

If:

```text
det(A) < 0
```

the transformation reverses orientation.

In three dimensions, the absolute value of the determinant describes the volume-scaling factor.

If:

```text
det(A) = 0
```

the transformation collapses at least one dimension. The image of the transformation has lower dimension than the original space.

This explains why a zero determinant is associated with a singular matrix.

---

## 10. Determinants and Matrix Invertibility

A square matrix A is invertible if there exists a matrix called its inverse, written as `inverse(A)`, such that:

```text
A × inverse(A) = inverse(A) × A = I
```

A fundamental result in linear algebra is:

```text
A is invertible if and only if det(A) ≠ 0
```

Therefore:

- If det(A) is nonzero, A is invertible.
- If det(A) is zero, A is singular and has no ordinary inverse.

### Example 1: Invertible Matrix

```text
A = [ 2  1 ]
    [ 1  3 ]
```

Calculate:

```text
det(A) = 2 × 3 - 1 × 1
       = 6 - 1
       = 5
```

Since det(A) is nonzero, A is invertible.

### Example 2: Singular Matrix

```text
B = [ 1  2 ]
    [ 2  4 ]
```

Calculate:

```text
det(B) = 1 × 4 - 2 × 2
       = 4 - 4
       = 0
```

The second row is twice the first row.

The rows are linearly dependent, so B is singular and has no ordinary inverse.

---

## 11. Inverse Matrices

### 11.1. Definition

The inverse of a square matrix A is denoted by `inverse(A)`.

It satisfies:

```text
A × inverse(A) = inverse(A) × A = I
```

The inverse reverses the linear transformation represented by A, provided that the transformation is invertible.

A matrix has an inverse if and only if its determinant is nonzero.

### 11.2. Inverse of a 2 × 2 Matrix

For:

```text
A = [ a  b ]
    [ c  d ]
```

If:

```text
a × d - b × c ≠ 0
```

then:

```text
inverse(A) = 1 / (a × d - b × c)
             ×
             [  d  -b ]
             [ -c   a ]
```

The procedure is:

1. Calculate the determinant.
2. Swap the diagonal entries.
3. Change the signs of the off-diagonal entries.
4. Divide every entry by the determinant.

### 11.3. Worked Example

Consider:

```text
A = [ 2  1 ]
    [ 1  3 ]
```

**Step 1: Calculate the determinant**

```text
det(A) = 2 × 3 - 1 × 1
       = 5
```

**Step 2: Swap the diagonal entries**

```text
[ 3  1 ]
[ 1  2 ]
```

**Step 3: Change the signs of the off-diagonal entries**

```text
[  3  -1 ]
[ -1   2 ]
```

**Step 4: Divide by the determinant**

```text
inverse(A) = 1/5 × [  3  -1 ]
                     [ -1   2 ]
```

Therefore:

```text
inverse(A) = [  0.6  -0.2 ]
             [ -0.2   0.4 ]
```

### 11.4. Verification

Multiply A by its inverse:

```text
A × inverse(A)

= [ 2  1 ] × 1/5 × [  3  -1 ]
  [ 1  3 ]          [ -1   2 ]
```

The result is:

```text
A × inverse(A) = [ 1  0 ]
                 [ 0  1 ]
```

This is the identity matrix, so the inverse is correct.

---

## 12. Solving Linear Systems Using Matrices

Many mathematical problems can be represented as:

```text
A × x = b
```

where:

- A is the coefficient matrix.
- x is the unknown vector.
- b is the right-hand-side vector.

If A is invertible, the solution can be written as:

```text
x = inverse(A) × b
```

This is a useful theoretical result.

### Worked Example

Consider the equations:

```text
2x + y = 5
x + 3y = 7
```

In matrix form:

```text
A = [ 2  1 ]
    [ 1  3 ]

x = [ x ]
    [ y ]

b = [ 5 ]
    [ 7 ]
```

The system is:

```text
A × x = b
```

Using the inverse calculated earlier:

```text
x = 1/5 × [  3  -1 ] × [ 5 ]
           [ -1   2 ]   [ 7 ]
```

Calculate the first component:

```text
x = (3 × 5 - 1 × 7) / 5
  = 8/5
  = 1.6
```

Calculate the second component:

```text
y = (-1 × 5 + 2 × 7) / 5
  = 9/5
  = 1.8
```

Therefore:

```text
x = 1.6
y = 1.8
```

Substituting these values into the original equations confirms the solution.

**Numerical computing note:** For practical numerical problems, use a linear-system solver rather than explicitly calculating the inverse. This is generally more efficient and numerically reliable.

---

## 13. Matrix Rank

### 13.1. Definition

The rank of a matrix is the maximum number of linearly independent rows or columns.

The number of linearly independent rows always equals the number of linearly independent columns.

Rank tells us how much independent information a matrix contains.

### 13.2. Full-Rank Matrix

Consider:

```text
A = [ 1  0 ]
    [ 0  1 ]
```

Its two columns are linearly independent.

Therefore:

```text
rank(A) = 2
```

This matrix has full rank because its rank equals the smaller of its number of rows and columns.

### 13.3. Rank-Deficient Matrix

Consider:

```text
B = [ 1  2 ]
    [ 2  4 ]
```

The second row is twice the first row. The second column is also twice the first column.

Only one independent row or column exists.

Therefore:

```text
rank(B) = 1
```

The matrix is rank-deficient because its rank is less than 2.

### 13.4. Rectangular Matrix

Consider:

```text
C = [ 1  2  3 ]
    [ 2  4  6 ]
```

This is a 2 × 3 matrix.

Its second row is twice its first row.

Therefore:

```text
rank(C) = 1
```

A matrix does not need to be square to have a rank.

For any matrix with m rows and n columns:

```text
0 ≤ rank(A) ≤ min(m, n)
```

---

## 14. How Can We Determine Matrix Rank?

One way to determine rank is to transform a matrix into row-echelon form using elementary row operations.

The rank equals the number of pivots in row-echelon form.

Consider:

```text
A = [ 1  2 ]
    [ 2  4 ]
```

Subtract twice the first row from the second row:

```text
R2 = R2 - 2 × R1
```

The result is:

```text
A = [ 1  2 ]
    [ 0  0 ]
```

Only one nonzero row remains.

Therefore:

```text
rank(A) = 1
```

This method helps explain the concept of rank.

For larger matrices, numerical software commonly uses elimination methods or Singular Value Decomposition (SVD).

---

## 15. Connection Between Determinant, Inverse, and Rank

For a square n × n matrix A, the following statements are equivalent:

1. A is invertible.
2. det(A) is nonzero.
3. rank(A) = n.
4. The columns of A are linearly independent.
5. The equation A × x = 0 has only the zero solution.

These properties describe the same fundamental condition: the matrix represents an invertible linear transformation.

### Example

Consider:

```text
A = [ 1  2 ]
    [ 2  4 ]
```

We know:

```text
det(A) = 0
rank(A) = 1
```

Since the rank is less than 2, the matrix is not invertible.

This demonstrates the mathematical connection between determinant, inverse, and rank.

---

## 16. Applications of Matrix Operations in Machine Learning

### 16.1. Matrix Multiplication

Matrix multiplication is widely used in:

- Linear regression
- Logistic regression
- Neural networks
- Transformer attention
- Feature transformations
- Batch processing

It combines features with weights and efficiently computes outputs.

### 16.2. Determinants

Determinants appear in some Machine Learning methods, including:

- Multivariate probability distributions
- Gaussian models
- Covariance-matrix calculations
- Change-of-variables formulas
- Volume-related calculations

For example, the probability density of a nondegenerate multivariate Gaussian distribution contains a factor proportional to:

```text
1 / sqrt(det(Sigma))
```

Here, Sigma is the covariance matrix.

For a valid nondegenerate Gaussian distribution, Sigma is symmetric positive definite, so its determinant is positive.

Determinants are not required explicitly in every Machine Learning algorithm.

### 16.3. Inverse Matrices

Matrix inverses appear in:

- Theoretical derivations of linear regression
- Analytical solutions to some optimization problems
- Linear systems
- Gaussian probability calculations
- Certain estimation methods

For example, the ordinary least-squares solution is often written as:

```text
w_hat = inverse(transpose(X) × X) × transpose(X) × y
```

This formula applies when `transpose(X) × X` is invertible.

However, it is not generally the preferred numerical implementation for large or poorly conditioned datasets.

Practical alternatives include QR decomposition, SVD, and specialized least-squares solvers.

### 16.4. Matrix Rank

Matrix rank helps us understand:

- Linear dependence between features
- Redundant columns in a dataset
- Identifiability in linear models
- Dimensionality reduction
- Low-rank matrix approximations
- Parameter redundancy

If one feature is an exact linear combination of other features, the design matrix may have deficient column rank.

This can prevent unique identification of all regression coefficients using ordinary least squares.

---

## 17. Matrix Operations in Reinforcement Learning

### 17.1. Transition Matrices

In finite-state Markov Decision Processes, transition probabilities can be represented using matrices.

For a fixed policy, the transition matrix describes how the agent moves between states.

Matrix-vector multiplication can calculate the next-state distribution and expected next-state values.

### 17.2. Policy Evaluation

The Bellman equation for a fixed policy can be written as:

```text
v = r + gamma × P × v
```

Rearranging:

```text
(I - gamma × P) × v = r
```

Solving this system yields the state-value vector for the policy.

This is a direct connection between linear algebra and dynamic programming.

### 17.3. Determinants and Inverse Matrices in RL

In a small finite-state problem, the matrix equation can be studied using determinants and inverse matrices.

However, practical Reinforcement Learning implementations rarely need to calculate a determinant or explicitly invert the Bellman matrix.

Iterative policy evaluation, Temporal-Difference learning, and neural value-function approximation often avoid explicit matrix inversion.

These concepts remain useful for understanding the mathematical structure of RL.

### 17.4. Matrix Rank in RL

Matrix rank can be relevant when studying:

- Linear value-function approximation
- Feature redundancy
- Least-Squares Temporal-Difference methods
- Model identification
- Low-rank representations
- Multi-agent state and feature representations

Redundant state features can make a matrix used in a linear estimation method singular or poorly conditioned.

Understanding rank helps explain why some systems do not have a unique solution or may be numerically unstable.

---

## 18. Python Implementation Using NumPy

### 18.1. Install NumPy

If NumPy is not installed, run:

```bash
pip install numpy
```

### 18.2. Matrix Multiplication

```python
import numpy as np

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

C = A @ B

print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)

print("\nA multiplied by B:")
print(C)
```

Expected output:

```text
Matrix A:
[[1 2]
 [3 4]]

Matrix B:
[[5 6]
 [7 8]]

A multiplied by B:
[[19 22]
 [43 50]]
```

The `@` operator performs matrix multiplication.

### 18.3. Determinant

```python
import numpy as np

A = np.array([
    [3, 2],
    [1, 4]
], dtype=float)

det_A = np.linalg.det(A)

print("Determinant:", det_A)
```

Expected output, approximately:

```text
Determinant: 10.0
```

Small floating-point differences may occur.

### 18.4. Inverse Matrix

```python
import numpy as np

A = np.array([
    [2, 1],
    [1, 3]
], dtype=float)

A_inv = np.linalg.inv(A)

print("Inverse matrix:")
print(A_inv)

print("\nA multiplied by its inverse:")
print(A @ A_inv)
```

Expected output, approximately:

```text
Inverse matrix:
[[ 0.6 -0.2]
 [-0.2  0.4]]

A multiplied by its inverse:
[[1. 0.]
 [0. 1.]]
```

Floating-point results may contain small numerical errors.

### 18.5. Matrix Rank

```python
import numpy as np

A = np.array([
    [1, 2],
    [2, 4]
], dtype=float)

rank_A = np.linalg.matrix_rank(A)

print("Matrix rank:", rank_A)
```

Output:

```text
Matrix rank: 1
```

NumPy estimates matrix rank using Singular Value Decomposition and a numerical tolerance.

For floating-point matrices, numerical rank can depend on the chosen tolerance and data precision.

---

## 19. A Complete Python Example

The following program calculates matrix multiplication, determinant, inverse, and rank.

```python
import numpy as np


def analyze_matrix(A):
    """Display basic properties of a square matrix."""

    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    print("Matrix A:")
    print(A)

    print("\nDimensions:")
    print(A.shape)

    print("\nDeterminant:")
    print(np.linalg.det(A))

    rank_A = np.linalg.matrix_rank(A)

    print("\nRank:")
    print(rank_A)

    print("\nInverse:")

    if rank_A == A.shape[0]:
        A_inv = np.linalg.solve(
            A,
            np.eye(A.shape[0])
        )

        print(A_inv)

        print("\nA multiplied by its inverse:")
        print(A @ A_inv)

    else:
        print("The matrix is singular or numerically rank-deficient.")
        print("A stable ordinary inverse is not available.")


A = np.array([
    [2, 1],
    [1, 3]
])

B = np.array([
    [1, 0],
    [2, 1]
])

print("Matrix multiplication A @ B:")
print(A @ B)

print("\nMatrix analysis:")
analyze_matrix(A)
```

This program demonstrates all four concepts.

**Important numerical note:** A numerical rank test is not a universal proof of invertibility. For difficult numerical problems, matrix conditioning and solver tolerances must also be considered.

---

## 20. Common Mistakes

### Mistake 1: Ignoring Matrix Dimensions

Matrix multiplication requires the number of columns in the first matrix to equal the number of rows in the second matrix.

Always check dimensions before multiplying matrices.

### Mistake 2: Assuming Matrix Multiplication Is Commutative

In general:

```text
AB ≠ BA
```

The order of matrix multiplication matters.

### Mistake 3: Calculating an Inverse When It Does Not Exist

If:

```text
det(A) = 0
```

the square matrix has no ordinary inverse.

### Mistake 4: Treating a Nearly Singular Matrix as Numerically Stable

A matrix may be technically invertible but so ill-conditioned that numerical calculations become unreliable.

A nonzero determinant alone does not guarantee numerical stability.

### Mistake 5: Explicitly Inverting a Matrix to Solve a Linear System

Although:

```text
x = inverse(A) × b
```

is mathematically correct when A is invertible, a numerical solver is generally preferable.

In Python, use:

```python
x = np.linalg.solve(A, b)
```

when A is a suitable nonsingular square matrix.

### Mistake 6: Confusing Rank with Matrix Dimensions

A 3 × 4 matrix has three rows and four columns, but its rank may be 0, 1, 2, or 3.

Dimensions do not determine rank by themselves.

---

## 21. Exercises

Try to solve the following exercises before checking the answers.

### Exercise 1: Matrix Multiplication

Given:

```text
A = [ 1  2 ]
    [ 3  4 ]

B = [ 2  0 ]
    [ 1  3 ]
```

Calculate AB and BA.

Are the results equal?

### Exercise 2: Matrix Dimensions

Suppose:

```text
A: 3 × 2
B: 2 × 4
```

Determine:

1. Whether AB is defined.
2. The dimensions of AB.
3. Whether BA is defined.

### Exercise 3: Determinant

Calculate the determinant of:

```text
A = [ 4  2 ]
    [ 1  3 ]
```

Is the matrix invertible?

### Exercise 4: Inverse Matrix

Calculate the inverse of:

```text
A = [ 1  2 ]
    [ 3  7 ]
```

Verify that:

```text
A × inverse(A) = I
```

### Exercise 5: Matrix Rank

Determine the rank of:

```text
A = [ 1  2  3 ]
    [ 2  4  6 ]
    [ 3  6  9 ]
```

Explain your reasoning.

### Exercise 6: Machine Learning Application

Suppose:

```text
X = [ 1  2 ]
    [ 3  4 ]
    [ 5  6 ]

w = [  2 ]
    [ -1 ]
```

Calculate Xw.

Explain why matrix multiplication is useful for making predictions for multiple observations.

### Exercise 7: Reinforcement Learning Application

Suppose a fixed-policy Markov reward process has:

```text
P = [ 0.5  0.5 ]
    [ 0.0  1.0 ]

r = [ 1 ]
    [ 2 ]

gamma = 0.5
```

Write the Bellman equation:

```text
v = r + gamma × P × v
```

Rearrange it into a linear system, then solve for the value vector.

### Exercise 8: Research Application

Suppose you are designing a multi-objective Reinforcement Learning scheduler for fog computing.

Your state representation includes:

- CPU utilization
- Memory utilization
- Network delay

Discuss:

1. How a matrix can represent observations from multiple fog nodes.
2. How matrix multiplication can combine state features with learned weights.
3. Why matrix rank may be relevant when some features are redundant.
4. Why a singular matrix can prevent a unique solution to some linear estimation problems.
5. Why matrix multiplication alone does not solve a multi-objective scheduling problem.

---

## 22. Exercise Answers

### Answer 1: Matrix Multiplication

Given:

```text
A = [ 1  2 ]
    [ 3  4 ]

B = [ 2  0 ]
    [ 1  3 ]
```

First calculate AB:

```text
AB = [ 1×2 + 2×1    1×0 + 2×3 ]
     [ 3×2 + 4×1    3×0 + 4×3 ]
```

Therefore:

```text
AB = [  4   6 ]
     [ 10  12 ]
```

Now calculate BA:

```text
BA = [ 2×1 + 0×3    2×2 + 0×4 ]
     [ 1×1 + 3×3    1×2 + 3×4 ]
```

Thus:

```text
BA = [  2   4 ]
     [ 10  14 ]
```

Therefore:

```text
AB ≠ BA
```

### Answer 2: Matrix Dimensions

Given:

```text
A: 3 × 2
B: 2 × 4
```

The inner dimensions match for AB.

Therefore:

```text
AB: 3 × 4
```

For BA:

```text
(2 × 4) × (3 × 2)
```

The inner dimensions, 4 and 3, do not match.

Therefore, BA is undefined.

### Answer 3: Determinant

Given:

```text
A = [ 4  2 ]
    [ 1  3 ]
```

Calculate:

```text
det(A) = 4×3 - 2×1
       = 12 - 2
       = 10
```

Since the determinant is nonzero, the matrix is invertible.

### Answer 4: Inverse Matrix

Given:

```text
A = [ 1  2 ]
    [ 3  7 ]
```

Calculate the determinant:

```text
det(A) = 1×7 - 2×3
       = 7 - 6
       = 1
```

Therefore:

```text
inverse(A) = [  7  -2 ]
             [ -3   1 ]
```

Verification:

```text
A × inverse(A)

= [ 1  2 ] × [  7  -2 ]
  [ 3  7 ]   [ -3   1 ]
```

The result is:

```text
A × inverse(A) = [ 1  0 ]
                 [ 0  1 ]
```

This is the identity matrix.

### Answer 5: Matrix Rank

Given:

```text
A = [ 1  2  3 ]
    [ 2  4  6 ]
    [ 3  6  9 ]
```

The second row is twice the first row, and the third row is three times the first row.

Only one independent row exists.

Therefore:

```text
rank(A) = 1
```

### Answer 6: Machine Learning Application

Given:

```text
X = [ 1  2 ]
    [ 3  4 ]
    [ 5  6 ]

w = [  2 ]
    [ -1 ]
```

Calculate:

```text
Xw = [ 1×2 + 2×(-1) ]
     [ 3×2 + 4×(-1) ]
     [ 5×2 + 6×(-1) ]
```

Therefore:

```text
Xw = [ 0 ]
     [ 2 ]
     [ 4 ]
```

Matrix multiplication calculates predictions for all three observations in one operation.

### Answer 7: Reinforcement Learning Application

The Bellman equation is:

```text
v = r + gamma × P × v
```

Rearranging:

```text
(I - gamma × P) × v = r
```

Substitute the given values:

```text
I - gamma × P

= [ 1  0 ] - 0.5 × [ 0.5  0.5 ]
  [ 0  1 ]         [ 0.0  1.0 ]
```

Therefore:

```text
I - gamma × P = [ 0.75  -0.25 ]
                [ 0.00   0.50 ]
```

The linear system is:

```text
0.75v1 - 0.25v2 = 1
0.50v2 = 2
```

From the second equation:

```text
v2 = 4
```

Substitute into the first equation:

```text
0.75v1 - 0.25×4 = 1
0.75v1 - 1 = 1
0.75v1 = 2
v1 = 8/3
```

Therefore:

```text
v = [ 8/3 ]
    [ 4   ]
```

This example demonstrates how a Bellman equation can be converted into a system of linear equations.

### Answer 8: Research Application

Possible answers include:

1. Each row can represent a fog node or state observation, while columns represent CPU utilization, memory utilization, and network delay.
2. Matrix multiplication can combine features with learned parameters to estimate values or scores.
3. If one feature is an exact linear combination of other features, the feature matrix may have deficient rank.
4. A singular coefficient matrix can prevent a unique solution. Depending on the right-hand side, a linear system may have no solution or infinitely many solutions.
5. Multi-objective scheduling requires objectives, constraints, and a decision strategy for balancing competing goals. Matrix multiplication is only one mathematical operation that may be used in the model.

---

## 23. Key Takeaways

- Matrix multiplication combines rows and columns through dot products.
- The inner dimensions must match for matrix multiplication to be defined.
- Matrix multiplication is generally not commutative.
- The determinant is a scalar associated with a square matrix.
- A square matrix is invertible if and only if its determinant is nonzero.
- Matrix rank measures the number of linearly independent rows or columns.
- A square matrix is invertible if and only if it has full rank.
- Matrix multiplication is fundamental to neural networks and Transformer architectures.
- Matrix operations are used in transition models and Bellman equations in Reinforcement Learning.
- Solving linear systems is generally preferable to explicitly calculating matrix inverses in numerical applications.
- Determinants, inverse matrices, and rank provide useful theoretical insights, but not every ML or RL algorithm requires all of them.

---

## 24. Further Learning

After completing this lesson, study:

1. Matrix Transpose
2. Identity and Diagonal Matrices
3. Vector Norms and Matrix Norms
4. Linear Independence and Basis Vectors
5. Eigenvalues and Eigenvectors
6. Singular Value Decomposition (SVD)
7. Gradients and Jacobian Matrices
8. Linear Regression and Least-Squares Estimation
9. Bellman Equations and Dynamic Programming
10. Linear Function Approximation in Reinforcement Learning

These topics provide the mathematical foundations needed for deeper study of Machine Learning and Reinforcement Learning.

---

## About This Lesson

This lesson is part of the **Mathematics for Machine Learning and Reinforcement Learning** learning series.

The objective is to connect mathematical foundations with Python implementations and practical applications in Machine Learning, Reinforcement Learning, and intelligent systems.

**Suggested next lesson:** System of Linear Equations.
