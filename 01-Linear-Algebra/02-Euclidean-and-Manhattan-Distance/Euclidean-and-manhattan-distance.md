# Euclidean and Manhattan Distance

## Mathematical Foundations for Machine Learning and Reinforcement Learning

**Level:** Beginner to Intermediate  
**Topics:** Mathematics for Machine Learning, Distance Metrics, Python  
**Applications:** Machine Learning, Clustering, K-Nearest Neighbors, Grid-Based Navigation, Reinforcement Learning

---

## 1. Learning Objectives

By the end of this lesson, you will be able to:

1. Understand the mathematical concept of distance between two points.
2. Calculate Euclidean distance using its mathematical formula.
3. Calculate Manhattan distance using its mathematical formula.
4. Implement both distance measures in Python.
5. Understand the differences between Euclidean and Manhattan distance.
6. Explain how distance measures are used in machine learning algorithms.
7. Understand the role of distance in reinforcement learning and grid-based navigation.
8. Identify situations in which each distance measure is appropriate.

---

## 2. Introduction: What Is Distance?

Distance is a mathematical measure of how far apart two points are.

In machine learning, data points are often represented as vectors containing numerical features.

For example, a student might be represented by two features:

- Mathematics score
- Programming score

Suppose two students are represented by these points:

```text
A = (2, 3)
B = (5, 7)
```

We can calculate the distance between these points using different mathematical methods.

Two commonly used methods are:

- **Euclidean distance**
- **Manhattan distance**

Although both measure the separation between points, they calculate it differently.

Understanding these differences is important because the choice of distance measure can affect the behavior of machine learning algorithms.

---

## 3. Euclidean Distance

### 3.1. Definition

Euclidean distance is the ordinary straight-line distance between two points.

Imagine two locations on a map. If you could travel directly from one location to the other without following roads or avoiding obstacles, the length of that straight path would be the Euclidean distance.

It is based on the Pythagorean theorem.

### Mathematical Formula in Two Dimensions

Suppose we have two points:

```text
A = (x₁, y₁)
B = (x₂, y₂)
```

Their Euclidean distance is:

```text
d_E(A, B) = √((x₂ - x₁)² + (y₂ - y₁)²)
```

Here:

- `(x₁, y₁)` represents the coordinates of the first point.
- `(x₂, y₂)` represents the coordinates of the second point.
- `d_E(A, B)` represents the Euclidean distance between the points.
- `√x` means the square root of `x`.

The formula calculates the differences between corresponding coordinates, squares those differences, adds them together, and takes the square root.

### 3.2. Worked Example

Consider the points:

```text
A = (2, 3)
B = (5, 7)
```

**Step 1: Calculate the coordinate differences.**

```text
dx = 5 - 2 = 3
dy = 7 - 3 = 4
```

**Step 2: Square the differences.**

```text
dx² = 3² = 9
dy² = 4² = 16
```

**Step 3: Add the squared differences.**

```text
9 + 16 = 25
```

**Step 4: Calculate the square root.**

```text
d_E(A, B) = √25 = 5
```

Therefore:

```text
Euclidean distance = 5 units
```

### Geometric Interpretation

The coordinate differences form the two perpendicular sides of a right triangle.

Their lengths are 3 and 4 units. The Euclidean distance is the length of the hypotenuse.

According to the Pythagorean theorem:

```text
3² + 4² = 5²
```

Therefore, the straight-line distance is 5 units.

---

## 4. Euclidean Distance in Higher Dimensions

Machine learning datasets often contain more than two features.

For example, a customer may be represented by:

- Age
- Annual income
- Number of purchases

Each feature corresponds to one dimension of the feature space.

For two vectors with `n` dimensions:

```text
A = (a₁, a₂, ..., aₙ)
B = (b₁, b₂, ..., bₙ)
```

the Euclidean distance is:

```text
d_E(A, B) = √(Σᵢ₌₁ⁿ (aᵢ - bᵢ)²)
```

If your editor does not display the summation notation clearly, use this equivalent expanded form:

```text
d_E(A, B) = √((a₁ - b₁)² + (a₂ - b₂)² + ... + (aₙ - bₙ)²)
```

The same formula can be expressed using the L2 norm:

```text
d_E(A, B) = ||A - B||₂
```

### Example in Three Dimensions

Consider:

```text
A = (1, 2, 3)
B = (4, 6, 3)
```

The Euclidean distance is:

```text
d_E(A, B) = √((4 - 1)² + (6 - 2)² + (3 - 3)²)

          = √(3² + 4² + 0²)

          = √(9 + 16 + 0)

          = √25

          = 5
```

Therefore:

```text
Euclidean distance = 5 units
```

The same principle applies regardless of the number of dimensions.

---

## 5. Manhattan Distance

### 5.1. Definition

Manhattan distance measures the sum of the absolute differences between corresponding coordinates.

It is also called:

- City-block distance
- Taxicab distance
- L1 distance

The name comes from navigating a city whose streets form a grid.

Imagine that you are in a city where you can move only horizontally or vertically along streets. You cannot travel diagonally through buildings.

The total length of your route may be greater than the straight-line distance.

Manhattan distance models this type of movement.

### 5.2. Mathematical Formula in Two Dimensions

For two points:

```text
A = (x₁, y₁)
B = (x₂, y₂)
```

the Manhattan distance is:

```text
d_M(A, B) = |x₂ - x₁| + |y₂ - y₁|
```

The vertical bars represent absolute value.

Absolute value ensures that coordinate differences contribute nonnegative distances.

For example:

```text
|-4| = 4
|4|  = 4
```

Moving four units to the left contributes the same distance as moving four units to the right.

### 5.3. Worked Example

Consider the same points:

```text
A = (2, 3)
B = (5, 7)
```

**Step 1: Calculate the coordinate differences.**

```text
dx = 5 - 2 = 3
dy = 7 - 3 = 4
```

**Step 2: Calculate the absolute values.**

```text
|dx| = 3
|dy| = 4
```

**Step 3: Add the absolute differences.**

```text
d_M(A, B) = |5 - 2| + |7 - 3|
          = 3 + 4
          = 7
```

Therefore:

```text
Manhattan distance = 7 units
```

Unlike Euclidean distance, Manhattan distance does not take the square root of the sum of squared differences. Instead, it adds the absolute coordinate differences directly.

---

## 6. Manhattan Distance in Higher Dimensions

For two vectors with `n` dimensions:

```text
A = (a₁, a₂, ..., aₙ)
B = (b₁, b₂, ..., bₙ)
```

the Manhattan distance is:

```text
d_M(A, B) = Σᵢ₌₁ⁿ |aᵢ - bᵢ|
```

An expanded form is:

```text
d_M(A, B) = |a₁ - b₁| + |a₂ - b₂| + ... + |aₙ - bₙ|
```

The Manhattan distance is also the L1 norm of the difference between the two vectors:

```text
d_M(A, B) = ||A - B||₁
```

### Example in Three Dimensions

Consider:

```text
A = (1, 2, 3)
B = (4, 6, 3)
```

The Manhattan distance is:

```text
d_M(A, B) = |4 - 1| + |6 - 2| + |3 - 3|

          = 3 + 4 + 0

          = 7
```

Therefore:

```text
Manhattan distance = 7 units
```

Notice that the Euclidean distance is 5, whereas the Manhattan distance is 7.

The two measures describe distance differently.

---

## 7. Euclidean vs. Manhattan Distance

| Property | Euclidean Distance | Manhattan Distance |
|---|---|---|
| Alternative name | L2 distance | L1 distance |
| Geometric interpretation | Straight-line distance | Sum of coordinate-wise movements |
| Formula | `√(Σ(aᵢ - bᵢ)²)` | Σ abs(aᵢ-bᵢ) |
| Uses squared differences | Yes | No |
| Uses absolute differences | Not directly | Yes |
| Takes a square root | Yes | No |
| Typical grid movement | Straight-line geometry | Horizontal and vertical movement |
| Contribution of large differences | Squaring emphasizes large differences | Differences contribute linearly |
| Common applications | KNN, K-means, geometric problems | Grid navigation, L1-based methods, some clustering approaches |

### Important Observation

For the points:

```text
A = (2, 3)
B = (5, 7)
```

we obtained:

```text
Euclidean distance = 5
Manhattan distance = 7
```

For any two vectors with the same finite number of real-valued coordinates:

```text
Euclidean distance ≤ Manhattan distance
```

To understand why, let:

```text
u = |x₂ - x₁|
v = |y₂ - y₁|
```

Since `u` and `v` are nonnegative:

```text
√(u² + v²) ≤ u + v
```

This inequality also extends to higher dimensions.

**Important:** Euclidean and Manhattan distances are equal when at most one coordinate differs between the two points.

For example:

```text
A = (1, 2)
B = (5, 2)
```

Both distances are 4 units.

---

## 8. Python Implementation

Now, let us implement both distance measures using Python.

We will use NumPy to represent vectors and perform mathematical operations.

### 8.1. Install NumPy

If NumPy is not installed in your Python environment, run:

```bash
pip install numpy
```

### 8.2. Calculate Euclidean Distance

```python
import numpy as np

A = np.array([2, 3])
B = np.array([5, 7])

euclidean_distance = np.linalg.norm(A - B, ord=2)

print("Euclidean Distance:", euclidean_distance)
```

Output:

```text
Euclidean Distance: 5.0
```

### Explanation

The expression:

```python
A - B
```

calculates the difference between corresponding vector components.

The result is:

```text
[-3 -4]
```

The function:

```python
np.linalg.norm(A - B, ord=2)
```

calculates the Euclidean norm of the difference vector.

Therefore:

```text
√((-3)² + (-4)²) = 5
```

The Euclidean distance between two vectors is the Euclidean norm of their difference.

### 8.3. Calculate Manhattan Distance

```python
import numpy as np

A = np.array([2, 3])
B = np.array([5, 7])

manhattan_distance = np.sum(np.abs(A - B))

print("Manhattan Distance:", manhattan_distance)
```

Output:

```text
Manhattan Distance: 7
```

### Explanation

The expression:

```python
np.abs(A - B)
```

calculates the absolute differences:

```text
[3 4]
```

Then:

```python
np.sum(np.abs(A - B))
```

adds these values.

Therefore:

```text
Manhattan distance = 3 + 4 = 7
```

### 8.4. Compare Both Distances

We can calculate both measures in the same program.

```python
import numpy as np

A = np.array([2, 3])
B = np.array([5, 7])

euclidean = np.linalg.norm(A - B, ord=2)
manhattan = np.linalg.norm(A - B, ord=1)

print("Point A:", A)
print("Point B:", B)
print("Euclidean Distance:", euclidean)
print("Manhattan Distance:", manhattan)
```

Output:

```text
Point A: [2 3]
Point B: [5 7]
Euclidean Distance: 5.0
Manhattan Distance: 7.0
```

For a one-dimensional vector, NumPy's `ord=2` specifies the Euclidean norm, while `ord=1` specifies the sum of absolute vector components.

---

## 9. Applications in Machine Learning

Distance measures help machine learning algorithms compare observations, identify similarities, and organize data.

However, not every machine learning algorithm requires an explicit distance calculation. The importance of distance depends on the algorithm and the representation of the data.

### 9.1. K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a supervised learning algorithm used for classification and regression.

The basic idea is to predict the output for a new observation using nearby observations from the training dataset.

Suppose we have three training points:

```text
A = (1, 1), Class = Red
B = (2, 2), Class = Red
C = (5, 5), Class = Blue
```

A new observation is:

```text
Q = (2, 1)
```

We calculate its Euclidean distance from each training point.

**Distance from Q to A:**

```text
d_E(Q, A) = √((2 - 1)² + (1 - 1)²)
          = √(1 + 0)
          = 1
```

**Distance from Q to B:**

```text
d_E(Q, B) = √((2 - 2)² + (1 - 2)²)
          = √(0 + 1)
          = 1
```

**Distance from Q to C:**

```text
d_E(Q, C) = √((2 - 5)² + (1 - 5)²)
          = √(9 + 16)
          = 5
```

The two closest points belong to the Red class.

With `k = 1`, either A or B is a nearest neighbor, and both have the Red label. With `k = 2`, both nearest neighbors are Red.

Therefore, the KNN classifier predicts the Red class for this observation.

### Why Does the Distance Measure Matter?

Changing the distance metric can change which observations are considered nearest neighbors. This can affect the final prediction.

The choice between Euclidean and Manhattan distance should therefore be evaluated for the particular dataset and task.

### 9.2. Clustering

Clustering algorithms group observations based on similarities.

For example, a business might group customers using features such as:

- Annual income
- Purchase frequency
- Average spending

Distance measures can help determine which customers are similar.

Euclidean distance is commonly associated with K-means clustering, where cluster centers are means of the assigned points.

Manhattan distance is useful in some clustering approaches, including methods based on absolute deviations or medians.

**Important:** Standard K-means minimizes the sum of squared Euclidean distances to cluster centroids. Simply replacing Euclidean distance with Manhattan distance does not make standard K-means optimize a Manhattan-distance objective. The clustering objective and center-update rule must be chosen consistently.

### 9.3. Feature Scaling

Suppose a dataset contains two features:

- Age: values between 18 and 80.
- Annual income: values between 10,000 and 200,000.

The income feature can dominate the numerical distance because its values have a much larger scale.

This can make distance-based algorithms behave inappropriately if the features are intended to have comparable importance.

A common solution is feature scaling.

For example, standardization transforms a feature `x` into:

```text
z = (x - mean) / standard_deviation
```

where:

- `mean` is the mean of the feature.
- `standard_deviation` is its standard deviation.

After scaling, distance calculations may better reflect differences across features.

However, scaling is not automatically appropriate for every dataset. The correct approach depends on the meaning, distribution, and intended importance of each feature.

---

## 10. Applications in Reinforcement Learning

Reinforcement Learning (RL) is a machine learning framework in which an agent interacts with an environment to learn a policy that maximizes cumulative reward.

A reinforcement learning problem is commonly described using:

- States
- Actions
- Rewards
- Transition dynamics
- A policy

Distance measures can be useful in several RL settings, particularly when states have numerical or spatial representations.

### 10.1. Grid-World Navigation

Consider an agent navigating a two-dimensional grid.

The agent starts at:

```text
S = (1, 1)
```

and wants to reach the goal:

```text
G = (4, 5)
```

Assume that the agent can move only up, down, left, or right, one grid cell per action.

#### Manhattan Distance

```text
d_M(S, G) = |4 - 1| + |5 - 1|
          = 3 + 4
          = 7
```

Therefore, the Manhattan distance is 7.

If there are no obstacles and each move costs one action, the minimum number of moves is also 7.

#### Euclidean Distance

```text
d_E(S, G) = √((4 - 1)² + (5 - 1)²)
          = √(9 + 16)
          = √25
          = 5
```

The Euclidean distance is 5.

The Euclidean distance represents the straight-line separation between the start and goal. However, the agent cannot move diagonally in this environment.

Thus, Manhattan distance is a more appropriate estimate of the minimum number of moves when movement is restricted to horizontal and vertical steps of unit length.

#### Connection to Reinforcement Learning

An RL agent may use state information and rewards to learn how to reach the goal.

For example, the environment could provide:

- A positive reward for reaching the goal.
- A small negative reward for each move.
- An additional penalty for entering a hazardous cell.

Distance to the goal can help us understand the geometry of the task. It can also be used to design a heuristic, construct an auxiliary reward, or evaluate a policy.

However, **distance is not the same as expected cumulative reward**.

An obstacle, a costly transition, or an action with uncertain outcomes may make the geometrically closest route suboptimal.

### 10.2. Reward Shaping

Reward shaping modifies an RL reward function to provide additional learning signals.

One possible approach is to define a potential function based on the distance to the goal:

```text
Φ(s) = -d_M(s, G)
```

Here, `s` is the current state and `G` is the goal state.

A simple distance-based reward modification might be:

```text
r_new(s, a, s') =
    r(s, a, s') + α × [d_M(s, G) - d_M(s', G)]
```

where:

- `r(s, a, s')` is the original reward.
- `s` is the current state.
- `a` is the selected action.
- `s'` is the next state.
- `α` controls the magnitude of the additional reward.

If an action moves the agent closer to the goal, the distance-based term is positive. If an action moves the agent farther away, the term is negative.

#### An Important Theoretical Caution

An arbitrary distance-based reward modification can change the optimal policy.

A theoretically established alternative is potential-based reward shaping:

```text
r_new(s, a, s') =
    r(s, a, s') + γ × Φ(s') - Φ(s)
```

where:

- `Φ(s)` is a state-potential function.
- `γ` is the discount factor.

Under the standard assumptions of the potential-based reward-shaping result, this form preserves the optimal policy.

For example, we may define:

```text
Φ(s) = -d_M(s, G)
```

and use it in the potential-based formula.

This is an important distinction when applying mathematical ideas to reinforcement learning research.

### 10.3. Distance in State-Space Representations

In some RL problems, states are represented as numerical vectors.

For example:

```text
s = (x, y, v_x, v_y)
```

where:

- `x, y` are position coordinates.
- `v_x, v_y` are velocity components.

A distance measure can compare two states. However, ordinary Euclidean distance may not capture the true similarity between states.

For example, a difference of one unit in position may have a different meaning from a difference of one unit in velocity.

Possible approaches include:

- Scaling state features.
- Using a weighted distance.
- Learning a representation in which relevant states are close.
- Using a task-specific similarity measure.

A weighted Euclidean distance can be written as:

```text
d_W(s₁, s₂) = √(Σᵢ₌₁ⁿ wᵢ × (s₁ᵢ - s₂ᵢ)²)
```

where `wᵢ ≥ 0` represents the weight assigned to feature `i`.

When all weights equal 1, this reduces to ordinary Euclidean distance.

Distance-based methods can be useful in tabular RL, nearest-neighbor approaches, exploration methods, and state abstraction. Their usefulness depends on the representation and problem structure.

---

## 11. Practical Python Example: A Grid-World Distance Calculator

The following program calculates the Euclidean and Manhattan distances between two grid positions.

```python
import numpy as np


def euclidean_distance(point_a, point_b):
    """Calculate the straight-line distance between two points."""
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("Points must have the same dimensions.")

    return np.linalg.norm(a - b, ord=2)


def manhattan_distance(point_a, point_b):
    """Calculate the Manhattan distance between two points."""
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("Points must have the same dimensions.")

    return np.linalg.norm(a - b, ord=1)


start = (1, 1)
goal = (4, 5)

print("Start:", start)
print("Goal:", goal)

print("Euclidean distance:",
      euclidean_distance(start, goal))

print("Manhattan distance:",
      manhattan_distance(start, goal))
```

Expected output:

```text
Start: (1, 1)
Goal: (4, 5)
Euclidean distance: 5.0
Manhattan distance: 7.0
```

### Experiment 1

Change the starting and goal positions:

```python
start = (2, 2)
goal = (5, 2)
```

Both distance measures should return 3.

### Experiment 2

Now try:

```python
start = (1, 1)
goal = (4, 4)
```

Before running the program, calculate both distances.

Expected results:

```text
Euclidean distance = √18 ≈ 4.243
Manhattan distance = 6
```

This demonstrates how the distance measures differ when both coordinates change.

---

## 12. When Should You Use Each Distance Measure?

There is no universally best distance measure. The appropriate choice depends on the data representation, algorithm, and application.

### Use Euclidean Distance When:

- Straight-line geometric distance is meaningful.
- Features have appropriate scales.
- The algorithm is designed around Euclidean geometry.
- You want to measure the ordinary geometric separation between points.
- You are working with algorithms such as standard K-means or many implementations of KNN.

### Consider Manhattan Distance When:

- Movement is restricted to horizontal and vertical steps.
- You are solving grid-based navigation problems.
- You want a distance based on the sum of absolute feature differences.
- The problem or algorithm is designed around L1 geometry.
- You are investigating sparse models or methods based on absolute deviations.

### Before Choosing a Distance Measure, Ask:

1. What does each feature represent?
2. Are the features measured on comparable scales?
3. Does the problem have geometric or grid-based structure?
4. Does the algorithm assume a particular distance measure?
5. Which measure produces better results on validation data?

---

## 13. Common Mistakes

### Mistake 1: Forgetting the Square Root

Incorrect Euclidean distance:

```text
d_E(A, B) = Σᵢ₌₁ⁿ (aᵢ - bᵢ)²
```

Correct Euclidean distance:

```text
d_E(A, B) = √(Σᵢ₌₁ⁿ (aᵢ - bᵢ)²)
```

The expression without the square root is the squared Euclidean distance, not the ordinary Euclidean distance.

### Mistake 2: Forgetting Absolute Values

Incorrect Manhattan distance:

```text
d_M(A, B) = Σᵢ₌₁ⁿ (aᵢ - bᵢ)
```

Correct Manhattan distance:

```text
d_M(A, B) = Σᵢ₌₁ⁿ |aᵢ - bᵢ|
```

Without absolute values, positive and negative differences can cancel each other.

### Mistake 3: Assuming the Shortest Geometric Route Is Always the Best RL Route

A route with the smallest geometric distance may contain obstacles, high costs, or risky transitions.

An RL agent seeks to maximize expected cumulative reward, not necessarily minimize geometric distance.

### Mistake 4: Ignoring Feature Scales

A feature with large numerical values can dominate a distance calculation.

Feature scaling or a carefully designed weighted distance may be necessary.

### Mistake 5: Assuming Distance Always Represents Semantic Similarity

Two observations can be close numerically but different in meaning.

The usefulness of a distance measure depends on whether the representation captures the properties relevant to the task.

---

## 14. Exercises

Try to solve the following problems before looking at the answers.

### Exercise 1: Two-Dimensional Distance

Given:

```text
A = (1, 2)
B = (4, 6)
```

Calculate:

1. Euclidean distance.
2. Manhattan distance.

### Exercise 2: Three-Dimensional Distance

Given:

```text
A = (2, 1, 3)
B = (5, 5, 3)
```

Calculate both distances.

### Exercise 3: Python Implementation

Write two Python functions:

```python
euclidean_distance(a, b)
```

and:

```python
manhattan_distance(a, b)
```

Test them with at least three different pairs of points.

### Exercise 4: Grid Navigation

An RL agent starts at:

```text
S = (2, 3)
```

and wants to reach:

```text
G = (6, 8)
```

Assume the agent can move only horizontally or vertically, one grid cell per action.

Answer the following:

1. What is the Manhattan distance?
2. What is the Euclidean distance?
3. What is the minimum number of moves required if no obstacles block the route?
4. Can the agent always reach the goal in the number of moves predicted by the Manhattan distance if obstacles are present? Explain why.

### Exercise 5: Research Thinking

Imagine you are designing an RL agent for task scheduling in a fog-computing environment.

A state vector contains:

```text
s = (CPU utilization, memory utilization, network delay)
```

Discuss:

1. Why might Euclidean distance between two states be misleading?
2. Why might feature scaling be necessary?
3. How could a weighted distance better reflect the scheduling objective?
4. Why does a smaller state distance not necessarily imply a higher reward?

This exercise connects the mathematical concepts in this lesson to research on reinforcement learning and computing-resource scheduling.

---

## 15. Exercise Answers

### Answer 1

Given:

```text
A = (1, 2)
B = (4, 6)
```

Euclidean distance:

```text
d_E = √((4 - 1)² + (6 - 2)²)
    = √(9 + 16)
    = √25
    = 5
```

Manhattan distance:

```text
d_M = |4 - 1| + |6 - 2|
    = 3 + 4
    = 7
```

### Answer 2

Given:

```text
A = (2, 1, 3)
B = (5, 5, 3)
```

Euclidean distance:

```text
d_E = √((5 - 2)² + (5 - 1)² + (3 - 3)²)
    = √(9 + 16 + 0)
    = √25
    = 5
```

Manhattan distance:

```text
d_M = |5 - 2| + |5 - 1| + |3 - 3|
    = 3 + 4 + 0
    = 7
```

### Answer 3

One possible implementation is:

```python
import numpy as np


def euclidean_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("Points must have the same dimensions.")

    return np.linalg.norm(a - b, ord=2)


def manhattan_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError("Points must have the same dimensions.")

    return np.linalg.norm(a - b, ord=1)


print(euclidean_distance([1, 2], [4, 6]))
print(manhattan_distance([1, 2], [4, 6]))
```

Output:

```text
5.0
7.0
```

### Answer 4

Given:

```text
S = (2, 3)
G = (6, 8)
```

Manhattan distance:

```text
d_M = |6 - 2| + |8 - 3|
    = 4 + 5
    = 9
```

Euclidean distance:

```text
d_E = √((6 - 2)² + (8 - 3)²)
    = √(16 + 25)
    = √41
    ≈ 6.403
```

If there are no obstacles and each horizontal or vertical move costs one action, the minimum number of moves is 9.

If obstacles block the shortest route, the agent may require more than nine moves. If the goal is unreachable, no valid route exists.

### Answer 5

Possible answers include:

1. CPU utilization, memory utilization, and network delay have different units and scales. Ordinary Euclidean distance may therefore be dominated by the feature with the largest numerical scale.
2. Feature scaling can make the numerical contributions of different features more comparable.
3. A weighted distance can assign greater importance to features that are more relevant to the scheduling objective.
4. State distance measures numerical or representational similarity, whereas reward depends on the environment's transition dynamics, resource constraints, action choices, and reward function.

The weights and scaling method should be justified by the scheduling problem rather than selected arbitrarily.

---

## 16. Key Takeaways

- **Euclidean distance** measures the straight-line distance between two points.
- **Manhattan distance** sums the absolute differences between corresponding coordinates.
- Both measures can be extended to any finite number of dimensions.
- Distance measures are useful in KNN, clustering, and other similarity-based methods.
- Manhattan distance is particularly useful for grid navigation when movement is restricted to horizontal and vertical steps.
- Feature scaling is often important when calculating distances between machine learning observations.
- In reinforcement learning, distance can support navigation, state comparison, heuristics, and reward design.
- Geometric distance and expected cumulative reward are different concepts.
- The best distance measure depends on the representation, algorithm, and application.

---

## 17. Further Learning

After completing this lesson, study the following topics:

1. Vectors and vector operations.
2. Vector norms: L1, L2, and L-infinity.
3. Dot product and cosine similarity.
4. Matrix operations and linear transformations.
5. Feature scaling and normalization.
6. K-Nearest Neighbors and clustering.
7. State representations in reinforcement learning.
8. Reward functions and potential-based reward shaping.

These topics provide mathematical foundations for understanding more advanced machine learning and reinforcement learning methods.

---

## About This Lesson

This lesson is part of a learning series on **Mathematics for Machine Learning and Reinforcement Learning**.

The series connects mathematical concepts with Python implementations and practical applications in machine learning, reinforcement learning, and intelligent systems.

**Next lesson suggestion:** Vector Norms and Similarity Measures: L1, L2, and Cosine Similarity.
