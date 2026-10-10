import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2 * x + 1

print(f"f(3) = {f(3)}")

x = np.linspace(-5, 5, 100)
y = f(x)

plt.plot(x, y, label="f(x) = 2x + 1")
plt.xlabel("Input x")
plt.ylabel("Output f(x)")
plt.title("Functions and Graphs")
plt.grid(True)
plt.legend()
plt.show()
