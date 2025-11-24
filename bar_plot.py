import matplotlib.pyplot as plt

# Data
labels = ['Baseline', 'Auto-tune', 'Deadzone']
overshoots = [33.54, 25.36, 16.66]
accuracies = [6.25, 4.26, 4.26]

# Bar plot for Overshoot
plt.figure(figsize=(6,4))
plt.bar(labels, overshoots, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.ylabel('Overshoot (%)')
plt.title('Overshoot Comparison')
plt.ylim(0, max(overshoots)*1.2)
for i, v in enumerate(overshoots):
    plt.text(i, v + 1, f"{v:.2f}", ha='center', va='bottom')
plt.tight_layout()
plt.show()

# Bar plot for Accuracy
plt.figure(figsize=(6,4))
plt.bar(labels, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.ylabel('Accuracy (Steady-State MAE)')
plt.title('Accuracy Comparison')
plt.ylim(0, max(accuracies)*1.2)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.2, f"{v:.2f}", ha='center', va='bottom')
plt.tight_layout()
plt.show()