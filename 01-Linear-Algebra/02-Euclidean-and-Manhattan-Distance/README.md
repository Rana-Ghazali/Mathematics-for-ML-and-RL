# Euclidean and Manhattan Distance

### Mathematical Foundations for Machine Learning and Reinforcement Learning

Welcome to this lesson on **Euclidean and Manhattan Distance**, two fundamental distance metrics used in Machine Learning (ML), data analysis, clustering, and Reinforcement Learning (RL).

This lesson introduces distance measurement in multidimensional spaces, explains the mathematical foundations of Euclidean and Manhattan distance, and demonstrates their practical implementation using Python and NumPy.

## 📚 Lesson Content

The complete lesson covers the following topics:

### 1. Introduction
- What is distance in mathematics?
- Why distance metrics are important
- Geometric intuition behind distance measurement
- Learning objectives

### 2. Mathematical Foundations
- Points and vectors in multidimensional spaces
- Euclidean distance
- Manhattan distance
- Two-dimensional and multidimensional distance calculations
- Geometric interpretation of distance metrics
- Differences between Euclidean and Manhattan distance
- The effect of feature scaling on distance calculations

### 3. Worked Examples
- Calculating Euclidean distance between two points
- Calculating Manhattan distance between two points
- Comparing distance metrics in two dimensions
- Extending distance calculations to higher-dimensional spaces
- Interpreting distance values in practical problems

### 4. Python Implementation
- Creating numerical vectors with NumPy
- Implementing Euclidean distance
- Implementing Manhattan distance
- Using NumPy for distance calculations
- Comparing multiple distance metrics
- Building reusable distance utility functions

### 5. Applications in Machine Learning
- Distance-based classification
- K-Nearest Neighbors (KNN)
- Clustering and data grouping
- Similarity and nearest-neighbor search
- Feature scaling and normalization
- Distance-based anomaly detection

### 6. Applications in Reinforcement Learning
- Measuring distances between environment states
- Grid-world navigation
- Manhattan distance as a heuristic for pathfinding
- Distance-based reward design
- Potential-based reward shaping
- State-space representation and exploration

### 7. Exercises
- Basic Euclidean distance calculations
- Manhattan distance calculations
- Comparing distance metrics
- Python implementation exercises
- Machine Learning applications
- Reinforcement Learning applications
- Mini project: Distance Metrics Toolkit

## 🎯 Learning Objectives

After completing this lesson, you will be able to:

- Explain the mathematical meaning of distance metrics.
- Calculate Euclidean distance between two points.
- Calculate Manhattan distance in multidimensional spaces.
- Compare the geometric properties of different distance metrics.
- Implement distance calculations using Python and NumPy.
- Explain how feature scaling affects distance-based algorithms.
- Apply distance metrics to classification, clustering, and nearest-neighbor problems.
- Understand how distance measurements support state evaluation, navigation, and reward design in Reinforcement Learning.

## 🛠️ Requirements

To run the Python examples, you need:

- Python 3.9 or later
- NumPy
- Jupyter Notebook (optional)

Install NumPy using:

```bash
pip install numpy
```

## 💻 Quick Example

The following example demonstrates how to calculate Euclidean and Manhattan distances using NumPy.

```python
import numpy as np

# Define two points
a = np.array([1, 2])
b = np.array([4, 6])

# Euclidean distance
euclidean_distance = np.linalg.norm(a - b)

# Manhattan distance
manhattan_distance = np.sum(np.abs(a - b))

print("Euclidean distance:", euclidean_distance)
print("Manhattan distance:", manhattan_distance)
```

Expected output:

```text
Euclidean distance: 5.0
Manhattan distance: 7
```

The Euclidean distance measures the straight-line distance between the points, while the Manhattan distance measures the sum of the absolute coordinate differences.

## 🤖 Why Distance Metrics Matter in ML and RL

Distance metrics provide a mathematical foundation for measuring differences between data points, feature vectors, and environment states.

| Concept | Application |
|---|---|
| Euclidean distance | Measuring straight-line distance between feature vectors |
| Manhattan distance | Measuring coordinate-wise distance |
| KNN | Identifying nearby training examples |
| Clustering | Grouping data points according to distance |
| Feature scaling | Preventing features with large numerical ranges from dominating distances |
| State distance | Comparing states in an RL environment |
| Grid-world navigation | Estimating the number of horizontal and vertical moves |
| Reward shaping | Designing distance-informed learning signals under appropriate theoretical conditions |

Understanding distance metrics helps build a foundation for studying similarity measures, clustering algorithms, search methods, and advanced Machine Learning and Reinforcement Learning techniques.

## 📖 Complete Lesson

**[Read the complete lesson: Euclidean and Manhattan Distance](02-euclidean-and-manhattan-distance.md)**

The full lesson includes mathematical explanations, worked examples, Python implementations, ML/RL applications, exercises, and a mini project.

## 🧪 Practice and Mini Project

As a practical exercise, implement a reusable distance toolkit in Python that supports:

- Euclidean distance
- Manhattan distance
- Distance calculations for vectors of different dimensions
- Comparison of multiple distance metrics
- Input validation for incompatible vector dimensions
- Testing with two-dimensional and multidimensional data

Extend the toolkit by applying the distance functions to a small KNN classification task or a grid-world navigation problem.

## 🗺️ Learning Path

This lesson is part of a broader mathematical learning path for Machine Learning and Reinforcement Learning.

**Recommended sequence:**

1. Vectors and Vector Operations
2. Dot Product and Cross Product
3. Euclidean and Manhattan Distance
4. Matrix Operations
5. Systems of Linear Equations
6. Orthogonality and Projection
7. Eigenvalues, Eigenvectors, and SVD
8. Cosine Similarity and Jaccard Similarity
9. Principal Component Analysis (PCA)

## 📌 Key Takeaway

Euclidean and Manhattan distance are fundamental tools for quantifying differences between points and vectors. Understanding their mathematical properties, implementation, and limitations helps you select appropriate distance metrics for Machine Learning algorithms and Reinforcement Learning environments.

---

**Previous lesson:** Vectors and Vector Operations

**Next lesson:** Matrix Operations

**Course:** Mathematical Foundations for Machine Learning and Reinforcement Learning
