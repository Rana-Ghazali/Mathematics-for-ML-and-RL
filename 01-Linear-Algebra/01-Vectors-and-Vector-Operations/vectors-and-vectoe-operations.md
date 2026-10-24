# Vectors and Vector Operations
## Mathematical Foundations for Machine Learning and Reinforcement Learning

## 1. Introduction

Vectors are fundamental mathematical objects in Linear Algebra, Machine Learning (ML), Deep Learning, and Reinforcement Learning (RL).

A vector is an ordered collection of numerical values. Vectors can represent data samples, features, model parameters, environmental states, and actions.

For example, a vector representing three numerical features can be written as:

**x = [2, 4, 6]**

The components of this vector are 2, 4, and 6.

Vectors allow machine learning algorithms to represent and process numerical information efficiently.

### Learning Objectives

By the end of this lesson, you will be able to:

- Understand vectors and their dimensions.
- Perform vector addition and subtraction.
- Apply scalar multiplication and linear combinations.
- Calculate vector magnitude and distance.
- Understand unit vectors, basis vectors, and span.
- Implement vector operations using Python and NumPy.
- Explain the applications of vectors in ML and RL.

---

## 2. Mathematical Foundations

### 2.1 Vector Representation

A vector containing n numerical components is called an n-dimensional vector.

A general vector can be represented as:

**x = [x₁, x₂, x₃, ..., xₙ]**

Here, xᵢ represents the i-th component of the vector.

Examples:

- One-dimensional vector: x = [5]
- Two-dimensional vector: x = [2, 3]
- Three-dimensional vector: x = [2, 4, 6]
- Five-dimensional vector: x = [1, 2, 3, 4, 5]

In Machine Learning, a sample may contain hundreds or thousands of features. Therefore, its feature vector may have a high dimension.

### 2.2 Vector Addition

Vector addition combines two vectors by adding their corresponding components.

Suppose:

a = [1, 2, 3]

b = [4, 5, 6]

Their sum is:

a + b = [1 + 4, 2 + 5, 3 + 6]

**Result: a + b = [5, 7, 9]**

Vectors must have the same dimension to be added directly.

**Python example:**

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)
```

Output:

```text
[5 7 9]
```

### 2.3 Vector Subtraction

Vector subtraction is performed by subtracting corresponding components.

Suppose:

a = [1, 2, 3]

b = [4, 5, 6]

Then:

a - b = [1 - 4, 2 - 5, 3 - 6]

**Result: a - b = [-3, -3, -3]**

Vector subtraction is useful for calculating differences between data points, model parameters, and numerical representations.

**Python example:**

```python
print(a - b)
```

Output:

```text
[-3 -3 -3]
```

### 2.4 Scalar Multiplication

A scalar is a single numerical value. Scalar multiplication multiplies every component of a vector by the scalar.

Suppose:

a = [1, 2, 3]

Multiplying the vector by 3 gives:

3a = [3 × 1, 3 × 2, 3 × 3]

**Result: 3a = [3, 6, 9]**

A negative scalar reverses the vector's direction, while the absolute value of the scalar determines how its magnitude changes.

**Python example:**

```python
a = np.array([1, 2, 3])

print(3 * a)
```

Output:

```text
[3 6 9]
```

### 2.5 Linear Combinations

A linear combination is created by multiplying vectors by scalar coefficients and adding the results.

For two vectors a and b:

**z = αa + βb**

Here, α and β are scalar coefficients.

Consider:

a = [1, 2]

b = [3, 4]

Let α = 2 and β = 3.

Then:

2a = [2, 4]

3b = [9, 12]

Therefore:

z = 2a + 3b

**Result: z = [11, 16]**

**Python implementation:**

```python
a = np.array([1, 2])
b = np.array([3, 4])

z = 2 * a + 3 * b

print(z)
```

Output:

```text
[11 16]
```

Linear combinations are important in linear regression, neural networks, and linear transformations.

### 2.6 Vector Magnitude and Norms

The magnitude of a vector measures its length.

The Euclidean norm, also called the L2 norm, is calculated by taking the square root of the sum of the squared components.

For a vector x:

**L2 norm = √(x₁² + x₂² + ... + xₙ²)**

Consider:

x = [3, 4]

The magnitude is:

L2 norm = √(3² + 4²)

L2 norm = √(9 + 16)

**Result: L2 norm = 5**

The L1 norm is the sum of the absolute values of the components.

**L1 norm = |x₁| + |x₂| + ... + |xₙ|**

For x = [3, 4]:

L1 norm = |3| + |4| = **7**

**Python implementation:**

```python
x = np.array([3, 4])

l2_norm = np.linalg.norm(x)
l1_norm = np.linalg.norm(x, ord=1)

print("L2 norm:", l2_norm)
print("L1 norm:", l1_norm)
```

Output:

```text
L2 norm: 5.0
L1 norm: 7.0
```

**Applications:** Norms are used in optimization, regularization, measuring model parameters, and evaluating prediction errors.

### 2.7 Unit Vectors

A unit vector is a vector whose magnitude equals 1.

To normalize a nonzero vector, divide each component by the vector's L2 norm.

**Unit vector = x / ||x||₂**

Here, ||x||₂ denotes the Euclidean norm of x.

Consider:

x = [3, 4]

Its magnitude is 5. Therefore:

Unit vector = [3/5, 4/5]

**Result: Unit vector = [0.6, 0.8]**

The magnitude of this normalized vector is 1.

**Python implementation:**

```python
x = np.array([3, 4])

unit_vector = x / np.linalg.norm(x)

print(unit_vector)
print(np.linalg.norm(unit_vector))
```

Output:

```text
[0.6 0.8]
1.0
```

Normalization is useful when direction matters more than magnitude, such as in cosine similarity and some machine learning representations.

A zero vector cannot be normalized using this formula because its magnitude is zero.

### 2.8 Distance Between Vectors

The Euclidean distance between two vectors is the magnitude of their difference.

**Distance(x, y) = ||x - y||₂**

Equivalently, the distance is the square root of the sum of squared differences between corresponding components.

Consider:

x = [1, 2]

y = [4, 6]

The component-wise differences are:

x - y = [-3, -4]

The Euclidean distance is:

Distance = √((-3)² + (-4)²)

Distance = √(9 + 16)

**Result: Distance = 5**

**Python implementation:**

```python
x = np.array([1, 2])
y = np.array([4, 6])

distance = np.linalg.norm(x - y)

print(distance)
```

Output:

```text
5.0
```

Distance measures are used in clustering, nearest-neighbor algorithms, and comparisons between state representations.

### 2.9 Basis Vectors and Span

A basis is a set of linearly independent vectors that can represent every vector in a given vector space through linear combinations.

The standard basis vectors in two-dimensional space are:

e₁ = [1, 0]

e₂ = [0, 1]

For example, the vector [3, 5] can be expressed as:

**[3, 5] = 3e₁ + 5e₂**

The span of a set of vectors is the collection of all possible linear combinations of those vectors.

Basis vectors and span are important in understanding vector spaces, linear transformations, dimensionality reduction, and feature representations.

---

## 3. Worked Examples

### Example 1: Vector Addition

Given:

a = [2, 3]

b = [4, 1]

Calculate a + b.

**Solution:**

a + b = [2 + 4, 3 + 1]

**Answer: [6, 4]**

### Example 2: Scalar Multiplication

Given:

x = [2, -3, 4]

Calculate -2x.

**Solution:**

-2x = [-2 × 2, -2 × (-3), -2 × 4]

**Answer: [-4, 6, -8]**

### Example 3: Vector Magnitude

Given:

x = [6, 8]

Calculate its L2 norm.

**Solution:**

L2 norm = √(6² + 8²)

L2 norm = √(36 + 64)

**Answer: 10**

### Example 4: Euclidean Distance

Given:

x = [2, 3]

y = [5, 7]

Calculate their Euclidean distance.

**Solution:**

Distance = √((5 - 2)² + (7 - 3)²)

Distance = √(9 + 16)

**Answer: 5**

### Example 5: Linear Combination

Given:

a = [1, 2]

b = [3, 4]

Calculate z = 2a + 3b.

**Solution:**

2a = [2, 4]

3b = [9, 12]

z = [2 + 9, 4 + 12]

**Answer: z = [11, 16]**

---

## 4. Python Implementation

NumPy is a widely used Python library for numerical computing. It provides efficient operations on vectors and multidimensional arrays.

### 4.1 Create Vectors

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Vector a:", a)
print("Vector b:", b)
```

### 4.2 Addition and Subtraction

```python
print("Addition:", a + b)
print("Subtraction:", a - b)
```

Output:

```text
Addition: [5 7 9]
Subtraction: [-3 -3 -3]
```

### 4.3 Scalar Multiplication

```python
print("Scalar multiplication:", 3 * a)
```

Output:

```text
Scalar multiplication: [3 6 9]
```

### 4.4 Calculate Norms

```python
x = np.array([3, 4])

print("L2 norm:", np.linalg.norm(x))
print("L1 norm:", np.linalg.norm(x, ord=1))
```

Output:

```text
L2 norm: 5.0
L1 norm: 7.0
```

### 4.5 Calculate Euclidean Distance

```python
x = np.array([1, 2])
y = np.array([4, 6])

distance = np.linalg.norm(x - y)

print("Euclidean distance:", distance)
```

Output:

```text
Euclidean distance: 5.0
```

### 4.6 Normalize a Vector

```python
x = np.array([3, 4])

unit_vector = x / np.linalg.norm(x)

print("Unit vector:", unit_vector)
```

Output:

```text
Unit vector: [0.6 0.8]
```

### 4.7 Linear Combination

```python
a = np.array([1, 2])
b = np.array([3, 4])

z = 2 * a + 3 * b

print("Linear combination:", z)
```

Output:

```text
Linear combination: [11 16]
```

---

## 5. Applications in Machine Learning

### 5.1 Feature Representation

A data sample can be represented as a feature vector.

For example, a house might be described by:

x = [120, 3, 10]

The values could represent:

- Floor area: 120 square meters.
- Number of bedrooms: 3.
- Building age: 10 years.

These features form a numerical representation that a machine learning algorithm can process.

In practice, feature scaling may be necessary because the features have different units and numerical ranges.

### 5.2 Linear Regression

Linear regression estimates a numerical output using a weighted combination of input features.

**Prediction = w · x + b**

Here:

- x is the input feature vector.
- w is the weight vector.
- w · x is the dot product of the vectors.
- b is the bias.
- Prediction is the estimated output.

For example:

x = [2, 3]

w = [4, 5]

b = 1

The prediction is:

Prediction = 4 × 2 + 5 × 3 + 1

**Answer: 24**

### 5.3 Nearest-Neighbor Algorithms

The k-Nearest Neighbors algorithm compares feature vectors to find similar samples.

Euclidean distance is one common measure of similarity-related distance. The appropriate distance metric depends on the data representation and application.

### 5.4 Clustering

Clustering algorithms group similar data points.

For example, the k-means algorithm calculates distances between data vectors and cluster centroids to assign samples to clusters.

### 5.5 Neural Networks

Neural networks use vectors to represent inputs, activations, gradients, and model parameters.

Vectorized calculations allow neural networks to process many numerical values efficiently.

---

## 6. Applications in Reinforcement Learning

### 6.1 State Representation

In Reinforcement Learning, a state can be represented by a vector of numerical features.

For example, a mobile robot's state might be:

s = [x, y, vx, vy]

Where:

- x and y represent the robot's position.
- vx represents horizontal velocity.
- vy represents vertical velocity.

An RL agent can use this state representation to select an action.

### 6.2 Action Representation

In continuous-control environments, an action may be represented as a vector.

For example:

a = [a₁, a₂]

The components could represent control signals for two robot motors.

The exact interpretation depends on the environment and the action space.

### 6.3 Value Function Approximation

A value function estimates the expected return associated with a state.

A simple linear value function is:

**V(s) = w · s**

Here:

- s is the state vector.
- w is a vector of learnable weights.
- V(s) is the estimated state value.

More advanced RL methods can use neural networks to approximate the value function.

### 6.4 Policy Representation

A policy maps a state to an action or a probability distribution over possible actions.

A simple linear transformation can be written as:

**z = Ws + b**

Here:

- s is the state vector.
- W is a weight matrix.
- b is a bias vector.
- z is the resulting vector.

A policy may transform z into action probabilities or continuous actions, depending on the algorithm.

### 6.5 Parameter Updates

Many RL algorithms use gradients to improve model parameters.

A simplified gradient-descent update is:

**New parameters = Current parameters - Learning rate × Gradient**

In symbols, using text notation:

theta_new = theta_old - alpha × gradient

Here, theta represents the model parameters and alpha is the learning rate.

The precise update rule depends on the algorithm, objective, and optimization method.

---

## 7. Exercises

Try to solve the following exercises before checking the answers.

### Exercise 1: Vector Addition

Given:

a = [2, 4]

b = [3, 1]

Calculate a + b.

### Exercise 2: Scalar Multiplication

Given:

x = [2, -1, 3]

Calculate 4x.

### Exercise 3: Vector Magnitude

Calculate the L2 norm of:

x = [5, 12]

### Exercise 4: Euclidean Distance

Calculate the Euclidean distance between:

x = [1, 1]

y = [4, 5]

### Exercise 5: Linear Combination

Given:

a = [1, 2]

b = [3, 4]

Calculate:

z = 2a + 3b

### Exercise 6: Python Practice

Write a Python program that:

1. Creates two NumPy vectors.
2. Calculates their sum and difference.
3. Calculates the magnitude of each vector.
4. Calculates the Euclidean distance between them.

### Exercise 7: Machine Learning

Explain how a feature vector represents a data sample in a classification problem.

### Exercise 8: Reinforcement Learning

Give an example of a state vector for a mobile robot and explain what each component represents.

---

## 8. Exercise Answers

### Answer 1

a + b = [2 + 3, 4 + 1]

**Answer: [5, 5]**

### Answer 2

4x = [4 × 2, 4 × (-1), 4 × 3]

**Answer: [8, -4, 12]**

### Answer 3

L2 norm = √(5² + 12²)

L2 norm = √(25 + 144)

**Answer: 13**

### Answer 4

Distance = √((4 - 1)² + (5 - 1)²)

Distance = √(9 + 16)

**Answer: 5**

### Answer 5

2a = [2, 4]

3b = [9, 12]

z = [2 + 9, 4 + 12]

**Answer: [11, 16]**

### Answer 6

One possible implementation is:

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Sum:", a + b)
print("Difference:", a - b)
print("Magnitude of a:", np.linalg.norm(a))
print("Magnitude of b:", np.linalg.norm(b))
print("Euclidean distance:", np.linalg.norm(a - b))
```

### Answer 7

A feature vector contains the numerical attributes describing one sample. A classification algorithm processes these features to predict the sample's class.

### Answer 8

A robot's state vector could contain its horizontal position, vertical position, and velocity. These values describe the robot's current situation and help an RL policy choose an action.

---

## 9. Summary

In this lesson, we studied vectors and their role in Machine Learning and Reinforcement Learning.

Key concepts include:

- Vector representation and dimension.
- Vector addition and subtraction.
- Scalar multiplication and linear combinations.
- Vector magnitude and norms.
- Unit vectors and normalization.
- Euclidean distance.
- Basis vectors and span.
- Vector operations using NumPy.
- Applications in feature representation, regression, clustering, neural networks, and RL.

Vectors provide the mathematical foundation for representing data, states, actions, and model parameters. Understanding vectors makes it easier to learn dot products, matrix operations, projections, and optimization methods.

## 10. References

1. NumPy Documentation: https://numpy.org/doc/stable/
2. GitHub Documentation — Writing mathematical expressions: https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions
3. Strang, G. *Introduction to Linear Algebra*. Wellesley-Cambridge Press.
