from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_excel("trainingSet_Classification.xlsx")

drop_cols = [
    "Kayıt No", "Sıra No", "Kayıt Yapan", "Kayıt Tarihi",
    "Aktif Mi_Ev", "Cihaz_HI1",
    "Pm10", "PM10_cat"
]
feature_cols = [c for c in train_df.columns if c not in drop_cols]

dt_clf_vis = DecisionTreeClassifier(random_state=42)
dt_clf_vis.fit(train_df[feature_cols], train_df["PM10_cat"])


plt.figure(figsize=(26, 14))
tree.plot_tree(
    dt_clf_vis,
    feature_names=feature_cols,
    class_names=["Good", "Moderate", "Poor"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.tight_layout()
plt.savefig("decision_tree_pm10cat.pdf", dpi=300)
plt.show()

print("Saved decision tree as decision_tree_pm10cat.pdf")
