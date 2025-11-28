import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


train_df = pd.read_excel("trainingSet_Classification.xlsx")
test_df = pd.read_excel("testSet_Classification.xlsx")


drop_cols = [
    "Kayıt No",
    "Sıra No",
    "Kayıt Yapan",
    "Kayıt Tarihi",
    "Aktif Mi_Ev",
    "Cihaz_HI1",
    "Pm10",
    "PM10_cat"  #Target column
]

feature_cols = [c for c in train_df.columns if c not in drop_cols]

print("Feature columns being used:")
print(feature_cols)

X_train = train_df[feature_cols]
y_train = train_df["PM10_cat"]

X_test = test_df[feature_cols]
y_test = test_df["PM10_cat"]


models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),

    # k-NN with scaling
    "k-NN": Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=7))
    ])
}


results = {}

for name, model in models.items():
    print("\n===============================")
    print(f" Training {name}")
    print("===============================\n")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print(f"{name} Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n-----------------------------------\n")

print("Final Accuracy Comparison (without Pm10 as a feature):")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
