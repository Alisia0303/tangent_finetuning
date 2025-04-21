import matplotlib.pyplot as plt

# Data
slices = [1, 3, 5, 10]
accuracies = [82.13, 79.68, 78.35, 57.15]

# Plot
plt.figure(figsize=(6, 4))
plt.plot(slices, accuracies, marker='o', linestyle='-', color='royalblue', label='Accuracy')

# Labels and title
plt.xlabel("Number of Slices")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs. Number of Slices")
plt.grid(True)
plt.xticks(slices)
plt.legend()
plt.tight_layout()
plt.savefig("acc.png")
plt.show()
