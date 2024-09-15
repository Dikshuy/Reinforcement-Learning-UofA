import matplotlib.pyplot as plt

initializations = [-100, -10, -5, 0, 5, 10, 100]
steps_PI = [3376, 3464, 3467, 3470, 3472, 3474, 3500]
steps_VI = [8, 8, 8, 8, 168, 237, 466]
steps_GPI = [25, 35, 40, 30, 70, 215, 470]

plt.figure(figsize=(10, 6))

plt.plot(initializations, steps_PI, label="PI", marker='o')
plt.plot(initializations, steps_VI, label="VI", marker='s')
plt.plot(initializations, steps_GPI, label="GPI", marker='d')

plt.xlabel("Initial Value (V₀)")
plt.ylabel("Number of Steps to Converge")
plt.title("Steps to Bellman Error Convergence for PI, VI, and GPI")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("policy_comparison.png")