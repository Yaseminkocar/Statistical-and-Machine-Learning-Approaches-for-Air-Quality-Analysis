

import matplotlib.pyplot as plt


first_training = {
    "Decision Tree": 1.0000,
    "Random Forest": 0.9979,
    "k-NN": 0.8940
}

second_training = {
    "Decision Tree": 0.3638,
    "Random Forest": 0.3659,
    "k-NN": 0.3493
}



print("\n===== FIRST vs SECOND TRAINING ACCURACY COMPARISON =====\n")
print("{:<20} {:<15} {:<15}".format("Model", "First", "Second"))
print("-" * 50)

for model in first_training.keys():
    print("{:<20} {:<15} {:<15}".format(
        model,
        f"{first_training[model]:.4f}",
        f"{second_training[model]:.4f}"
    ))

print("\n=========================================================\n")



models = list(first_training.keys())
first_scores = [first_training[m] for m in models]
second_scores = [second_training[m] for m in models]

x = range(len(models))
width = 0.35

plt.figure(figsize=(10, 6))


plt.bar([i - width/2 for i in x], first_scores, width=width, label="First Training")


plt.bar([i + width/2 for i in x], second_scores, width=width, label="Second Training")


plt.xticks(x, models)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison: First vs Second Model Training")
plt.ylim(0, 1.1)
plt.legend()

plt.tight_layout()
plt.savefig("accuracy_first_vs_second.pdf")
plt.close()

print("Saved comparison chart as accuracy_first_vs_second.pdf")
