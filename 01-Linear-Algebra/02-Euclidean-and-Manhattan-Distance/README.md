# Euclidean and Manhattan Distance

## Mathematical Foundations for Machine Learning and Reinforcement Learning

This lesson explains two fundamental distance metrics used in machine learning, data analysis, and reinforcement learning: **Euclidean Distance** and **Manhattan Distance**.

You will learn their mathematical foundations, work through numerical examples, implement both metrics in Python, and explore their applications in K-Nearest Neighbors (KNN), clustering, grid-world navigation, and reinforcement learning.

## Learning Objectives

By completing this lesson, you will be able to:

- Understand the concept of distance between numerical vectors.
- Calculate Euclidean distance in two or more dimensions.
- Calculate Manhattan distance in two or more dimensions.
- Implement distance metrics using Python and NumPy.
- Compare the properties and applications of the two metrics.
- Understand why feature scaling matters in distance-based algorithms.
- Apply distance measures to grid-world navigation and RL state representations.
- Explain the difference between geometric distance and cumulative reward.

## Mathematical Foundations

### 1. Euclidean Distance

Euclidean distance measures the straight-line distance between two points.

For two vectors, A and B, with n dimensions:

```text
A = (a1, a2, ..., an)
B = (b1, b2, ..., bn)
```

The Euclidean distance is:

```text
d_E(A, B) = sqrt(sum from i = 1 to n of (ai - bi)^2)
```

**Example**

```text
A = (2, 3)
B = (5, 7)

d_E(A, B) = sqrt((5 - 2)^2 + (7 - 3)^2)
          = sqrt(9 + 16)
          = 5
```

### 2. Manhattan Distance

Manhattan distance measures the sum of the absolute differences between corresponding coordinates.

```text
d_M(A, B) = sum from i = 1 to n of |ai - bi|
```

**Example**

```text
A = (2, 3)
B = (5, 7)

d_M(A, B) = |5 - 2| + |7 - 3|
          = 3 + 4
          = 7
```

For these points, the Euclidean distance is 5, while the Manhattan distance is 7.

## Python Implementation

This lesson uses NumPy to calculate both distance measures.

```python
import numpy as np

A = np.array([2, 3])
B = np.array([5, 7])

euclidean = np.linalg.norm(A - B, ord=2)
manhattan = np.linalg.norm(A - B, ord=1)

print("Euclidean Distance:", euclidean)
print("Manhattan Distance:", manhattan)
```

**Expected output**

```text
Euclidean Distance: 5.0
Manhattan Distance: 7.0
```

## Applications in Machine Learning

Distance metrics are useful for comparing observations and measuring similarity.

This lesson covers:

- **K-Nearest Neighbors (KNN):** Identifying nearby training observations for classification and regression.
- **Clustering:** Comparing observations and understanding distance-based grouping.
- **Feature Scaling:** Understanding how different numerical scales influence distance calculations.
- **Similarity Analysis:** Selecting an appropriate metric for a particular dataset and task.

## Applications in Reinforcement Learning

Distance measures can support several reinforcement learning tasks:

- **Grid-World Navigation:** Estimating the minimum number of horizontal and vertical moves to a goal.
- **State-Space Representation:** Comparing states represented by numerical vectors.
- **Reward Shaping:** Exploring distance-based learning signals and potential-based reward shaping.
- **Intelligent Scheduling:** Investigating state similarities in resource-allocation and fog-computing environments.

An important distinction is that geometric distance does not necessarily represent expected cumulative reward. Obstacles, transition costs, and uncertainty can affect the best action or route.

## Euclidean vs. Manhattan Distance

| Property | Euclidean Distance | Manhattan Distance |
|---|---|---|
| Geometric meaning | Straight-line distance | Sum of coordinate-wise differences |
| Mathematical basis | Squared differences and square root | Absolute differences |
| Common use | KNN, K-means, geometric problems | Grid navigation and L1-based methods |
| Grid movement | Represents straight-line separation | Models horizontal and vertical movement |
| Higher dimensions | Supported | Supported |

The appropriate distance measure depends on the data representation, feature scales, algorithm, and application.

## Lesson Contents

The complete lesson includes:

1. Introduction to distance metrics
2. Euclidean distance and worked examples
3. Euclidean distance in higher dimensions
4. Manhattan distance and worked examples
5. Manhattan distance in higher dimensions
6. Comparison of Euclidean and Manhattan distance
7. Python implementation with NumPy
8. Applications in machine learning
9. Applications in reinforcement learning
10. Grid-world distance calculator
11. Practical guidance for choosing a metric
12. Common mistakes
13. Exercises and solutions
14. Key takeaways and further learning

## Exercises

The lesson includes exercises on:

- Calculating distances in two and three dimensions.
- Implementing distance functions in Python.
- Calculating the minimum number of moves in grid navigation.
- Selecting distance measures for RL state representations.
- Analyzing distance metrics in fog-computing task scheduling.

## Requirements

- Python 3.x
- NumPy

Install NumPy with:

```bash
pip install numpy
```

## Who Should Study This Lesson?

This lesson is designed for:

- Students learning the mathematical foundations of machine learning.
- Beginners developing Python skills for data science and AI.
- Researchers studying reinforcement learning and intelligent systems.
- Learners interested in similarity measures, clustering, and state-space representations.

Basic knowledge of Python and vectors is helpful but not required.

## Learning Path

This lesson is part of the **Mathematical Foundations for Machine Learning and Reinforcement Learning** learning series.

Recommended learning sequence:

1. Functions and Graphs
2. Vectors and Vector Operations
3. Euclidean and Manhattan Distance
4. Vector Norms and Similarity Measures
5. Dot Product and Cosine Similarity
6. Matrix Operations
7. Eigenvalues, Eigenvectors, and SVD
8. Principal Component Analysis (PCA)

## Next Lesson

**Vector Norms and Similarity Measures: L1, L2, and Cosine Similarity**

The next lesson explores vector norms and similarity measures that are widely used in machine learning and reinforcement learning.

---

**Author:** Rana Ghazali

**Topics:** Mathematics for Machine Learning, Reinforcement Learning, Python, NumPy, Distance Metrics, KNN, Clustering, Grid-World Navigation, Intelligent Systems

**Learning Goal:** Build a strong mathematical foundation for understanding and applying machine learning and reinforcement learning algorithms.
