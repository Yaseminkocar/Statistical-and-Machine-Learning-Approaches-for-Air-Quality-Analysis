import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_excel("trainingSet_Classification.xlsx")

drop_cols = [
    "Kayıt No", "Sıra No", "Kayıt Yapan", "Kayıt Tarihi",
    "Aktif Mi_Ev", "Cihaz_HI1",
    "Pm10", "PM10_cat"
]

feature_cols = [c for c in train_df.columns if c not in drop_cols]

rf_clf = RandomForestClassifier(n_estimators=150, random_state=42)
rf_clf.fit(train_df[feature_cols], train_df["PM10_cat"])

importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar([feature_cols[i] for i in indices], importances[indices])
plt.xticks(rotation=75)
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importance for PM10_cat Prediction")
plt.tight_layout()
plt.savefig("rf_feature_importance.pdf", dpi=300)
plt.show()

print("Saved rf_feature_importance.pdf")
