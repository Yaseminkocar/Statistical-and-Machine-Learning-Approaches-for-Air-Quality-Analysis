import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_excel("trainingSet_Classification.xlsx")
test = pd.read_excel("testSet_Classification.xlsx")

y_train = train["PM10_cat"]
y_test = test["PM10_cat"]

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

cols_to_drop = [
    "PM10_cat", "Pm1", "Pm2.5", "Pm10",
    "Kayıt No", "Sıra No", "Kayıt Yapan", "Kayıt Tarihi"
]
feature_cols = [c for c in train.columns if c not in cols_to_drop]

X_train = train[feature_cols]
X_test = test[feature_cols]

print("Feature columns used:", feature_cols)


dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train_enc)
y_pred_dt = dt.predict(X_test)

dt_acc = accuracy_score(y_test_enc, y_pred_dt)
print(f"Decision Tree Accuracy: {dt_acc:.4f}")

print("\n=== Decision Tree Classification Report ===")
print(classification_report(y_test_enc, y_pred_dt, target_names=le.classes_))


ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)
ada.fit(X_train, y_train_enc)
y_pred_ada = ada.predict(X_test)

ada_acc = accuracy_score(y_test_enc, y_pred_ada)
print(f"\nAdaBoost Accuracy: {ada_acc:.4f}")

print("\n=== AdaBoost Classification Report ===")
print(classification_report(y_test_enc, y_pred_ada, target_names=le.classes_))


rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train_enc)
y_pred_rf = rf.predict(X_test)

rf_acc = accuracy_score(y_test_enc, y_pred_rf)
print(f"\nRandom Forest Accuracy: {rf_acc:.4f}")

print("\n=== Random Forest Classification Report ===")
print(classification_report(y_test_enc, y_pred_rf, target_names=le.classes_))


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(confusion_matrix(y_test_enc, y_pred_dt),
            annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[0])
axes[0].set_title("Decision Tree Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

sns.heatmap(confusion_matrix(y_test_enc, y_pred_ada),
            annot=True, fmt="d", cmap="Oranges",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[1])
axes[1].set_title("AdaBoost Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("True")

sns.heatmap(confusion_matrix(y_test_enc, y_pred_rf),
            annot=True, fmt="d", cmap="Greens",
            xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[2])
axes[2].set_title("Random Forest Confusion Matrix")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("True")

plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
plt.bar(["Decision Tree", "AdaBoost", "Random Forest"],
        [dt_acc, ada_acc, rf_acc],
        color=["blue", "orange", "green"])
plt.ylim(0, 1)
plt.ylabel("Test Accuracy")
plt.title("Base Classifier vs AdaBoost vs Random Forest")
plt.tight_layout()
plt.show()
