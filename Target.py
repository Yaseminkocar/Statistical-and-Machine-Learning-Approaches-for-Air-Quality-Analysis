import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

df = pd.read_excel("hi1_2025_09_eylul.xlsx")

# --- Rename columns to English ---
rename_map = {
    "Kayıt Tarihi": "Timestamp",
    "Pm1": "PM1",
    "Pm2.5": "PM2_5",
    "Pm10": "PM10",
    "CO2": "CO2",
    "CH2O": "CH2O",
    "Voc": "VOC",
    "Sicaklik(°C)": "Temperature",
    "Nem(%)": "Humidity",
    "His. Sıcaklık(°C)": "Apparent_Temperature"
}

df = df.rename(columns=rename_map)

# --- Time-based features ---
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek


# ==============================
# 2. Feature selection
# ==============================

y = df["O3"]  # Regression target

feature_cols = [
    "PM1", "PM2_5", "PM10",
    "CO2", "CH2O", "VOC",
    "Temperature", "Humidity", "Apparent_Temperature",
    "Hour", "DayOfWeek"
]

X = df[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_df = X_train.copy()
train_df["O3"] = y_train
test_df = X_test.copy()
test_df["O3"] = y_test
train_df.to_excel("train_split.xlsx", index=False)
test_df.to_excel("test_split.xlsx", index=False)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

rmse_lr  = rmse(y_test, y_pred_lr)
mape_lr  = mape(y_test, y_pred_lr)
rmse_rf  = rmse(y_test, y_pred_rf)
mape_rf  = mape(y_test, y_pred_rf)

print("Linear Regression  - RMSE:", rmse_lr)
print("Linear Regression  - MAPE: {:.2f}%".format(mape_lr))
print("Random Forest      - RMSE:", rmse_rf)
print("Random Forest      - MAPE: {:.2f}%".format(mape_rf))

plt.figure()
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel("True O3")
plt.ylabel("Predicted O3")
plt.title("Linear Regression: True vs Predicted O3")
plt.grid(True)
plt.savefig("plot_lr_true_vs_predicted.pdf")
plt.show()

plt.figure()
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel("True O3")
plt.ylabel("Predicted O3")
plt.title("Random Forest: True vs Predicted O3")
plt.grid(True)
plt.savefig("plot_rf_true_vs_predicted.pdf")
plt.show()

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.bar(range(len(feature_cols)), importances[indices])
plt.xticks(
    range(len(feature_cols)),
    np.array(feature_cols)[indices],
    rotation=45,
    ha="right"
)
plt.ylabel("Importance")
plt.title("Random Forest Feature Importances for O3 Regression")
plt.tight_layout()
plt.savefig("plot_rf_feature_importances.pdf")
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.title("Linear Regression")
plt.xlabel("True O3")
plt.ylabel("Predicted O3")

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.title("Random Forest")
plt.xlabel("True O3")
plt.ylabel("Predicted O3")

plt.tight_layout()
plt.savefig("plot_side_by_side_lr_rf.pdf")
plt.show()

lr_errors = y_test - y_pred_lr
rf_errors = y_test - y_pred_rf

plt.figure(figsize=(10, 6))
plt.hist(lr_errors, bins=40, alpha=0.6, label="LR Errors")
plt.hist(rf_errors, bins=40, alpha=0.6, label="RF Errors")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution Comparison")
plt.legend()
plt.savefig("plot_error_distribution.pdf")
plt.show()


print(f"Linear Regression & {rmse_lr:.3f} & {mape_lr:.2f} \\\\")
print(f"Random Forest & {rmse_rf:.3f} & {mape_rf:.2f} \\\\")
