import matplotlib.pyplot as plt
import numpy as np

xs = np.linspace(2, 20, 1000)
plt.plot(xs, np.exp(-xs), label="e^-x")
plt.plot(xs, xs ** (-1 / 2), label="1/sqrt(x)")
plt.plot(xs, xs ** (-2), label="1/x^2")
plt.plot(xs, xs ** (-3), label="1/x^3")
plt.legend()
plt.show()
