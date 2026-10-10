# Lesson 1: Functions and Graphs

**Learning Path:** Mathematics for Machine Learning and Reinforcement Learning

## 1. What Is a Function?

A function is a mathematical rule that maps each valid input to exactly one output.

We represent a function as:

\[
y=f(x)
\]

where:
- \(x\) is the input.
- \(f(x)\) is the output.
- \(f\) represents the mapping rule.

For example:

\[
f(x)=2x+1
\]

If \(x=3\), then:

\[
f(3)=2(3)+1=7
\]

Therefore, the output is 7.

## 2. Domain and Range

The **domain** is the set of valid input values, while the **range** is the set of corresponding output values.

For \(f(x)=2x+1\), if the domain is \(\{0,1,2,3\}\), the range is \(\{1,3,5,7\}\).

## 3. Understanding Graphs

A graph visually represents the relationship between inputs and outputs.

Consider the linear function:

\[
f(x)=2x+1
\]

Its general form is:

\[
y=mx+b
\]

Here, \(m\) is the slope and \(b\) is the y-intercept.

For our example, the slope is 2 and the y-intercept is 1.

## 4. Python Example

```python
def f(x):
    return 2 * x + 1

x = 3
y = f(x)

print(f"f({x}) = {y}")
```

Output:

```text
f(3) = 7
```

## 5. Applications in Machine Learning and Reinforcement Learning

Functions are fundamental to machine learning:

- A prediction model maps input features to outputs.
- A loss function measures prediction error.
- An RL policy maps a state to an action or a probability distribution over actions.
- A value function estimates expected future return.

For example, \(V(s)\) represents the value of state \(s\), while \(Q(s,a)\) represents the value of taking action \(a\) in state \(s\).

## 6. Exercises

1. Calculate \(f(0)\), \(f(2)\), and \(f(-1)\) for \(f(x)=3x-2\).
2. Calculate \(g(3)\) and \(g(-3)\) for \(g(x)=x^2\).
3. Modify the Python example to implement \(f(x)=-x+4\).
4. Explain why an RL policy can be considered a function.

## Key Takeaway

Functions describe relationships between inputs and outputs. Understanding them provides a foundation for studying machine learning models, neural networks, and reinforcement learning.

**Next lesson:** Powers, Exponentials, and Logarithms.
