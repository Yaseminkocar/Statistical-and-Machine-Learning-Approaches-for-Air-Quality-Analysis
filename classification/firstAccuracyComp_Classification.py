import matplotlib.pyplot as plt

accuracies = {
    "Decision Tree": 1.0000,
    "Random Forest": 0.9979,
    "k-NN": 0.8690
}

models = list(accuracies.keys())
scores = list(accuracies.values())

plt.figure(figsize=(8, 5))
bars = plt.bar(models, scores)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.3f}",
        ha="center", va="bottom", fontsize=12
    )

plt.ylim(0, 1.1)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Classification Models")
plt.tight_layout()
plt.savefig("accuracy_comparison.pdf", dpi=300)
plt.show()

print("Saved accuracy comparison chart as accuracy_comparison.pdf")
