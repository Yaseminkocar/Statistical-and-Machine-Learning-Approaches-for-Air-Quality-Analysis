import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn import tree



train_df = pd.read_excel("trainingSet_Classification.xlsx")
test_df  = pd.read_excel("testSet_Classification.xlsx")


drop_cols = [
    "Kayıt No",
    "Sıra No",
    "Kayıt Yapan",
    "Kayıt Tarihi",
    "Aktif Mi_Ev",
    "Cihaz_HI1",
    "Pm10",
    "Pm1",
    "Pm2.5",
    "PM10_cat"   # target
]

feature_cols = [c for c in train_df.columns if c not in drop_cols]

print("\nFeatures used in second training:")
print(feature_cols)

X_train = train_df[feature_cols]
y_train = train_df["PM10_cat"]

X_test  = test_df[feature_cols]
y_test  = test_df["PM10_cat"]



models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "k-NN": Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=7))
    ])
}

results = {}



for name, model in models.items():
    print("\n===================================")
    print(f" Training {name}")
    print("===================================\n")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"{name} Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n-----------------------------------\n")



print("Saving decision tree visualization...")

plt.figure(figsize=(26, 14))
dt_vis = models["Decision Tree"]
tree.plot_tree(
    dt_vis,
    feature_names=feature_cols,
    class_names=["Good", "Moderate", "Poor"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.tight_layout()
plt.savefig("decision_tree_second_training.pdf")
plt.close()



print("Saving Random Forest feature importance...")

rf = models["Random Forest"]
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar([feature_cols[i] for i in indices], importances[indices])
plt.xticks(rotation=75)
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance (Second Training)")
plt.tight_layout()
plt.savefig("rf_feature_importance_second_training.pdf")
plt.close()



print("Saving k-NN confusion matrix...")

y_pred_knn = models["k-NN"].predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_knn, annot=True, cmap="Blues",
    xticklabels=["Good", "Moderate", "Poor"],
    yticklabels=["Good", "Moderate", "Poor"]
)
plt.title("k-NN Confusion Matrix (Second Training)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("knn_confusion_matrix_second_training.pdf")
plt.close()


print("Saving accuracy comparison chart...")

models_list = list(results.keys())
scores = list(results.values())

plt.figure(figsize=(8, 5))
bars = plt.bar(models_list, scores)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.3f}",
        ha="center", va="bottom", fontsize=12
    )

plt.ylim(0, 0.5)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison (Second Training)")
plt.tight_layout()
plt.savefig("accuracy_comparison_second_training.pdf")
plt.close()

print("\nAll visualizations saved as PDF files successfully!")
