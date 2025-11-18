from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

df = pd.read_excel("hi1_2025_09_eylul.xlsx")

df["Kayıt Tarihi"] = pd.to_datetime(df["Kayıt Tarihi"])

df["Hour"] = df["Kayıt Tarihi"].dt.hour
df["DayOfWeek"] = df["Kayıt Tarihi"].dt.dayofweek

y = df["O3"]

feature_cols = [
    "Pm1", "Pm2.5", "Pm10",
    "CO2", "CH2O", "Voc",
    "Sicaklik(°C)", "Nem(%)", "His. Sıcaklık(°C)",
    "Hour", "DayOfWeek"
]

X = df[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Train size:", X_train.shape, y_train.shape)
print("Test size :", X_test.shape, y_test.shape)

train_df = X_train.copy()
train_df["O3"] = y_train

test_df = X_test.copy()
test_df["O3"] = y_test

train_df.to_excel("train_split.xlsx", index=False)
test_df.to_excel("test_split.xlsx", index=False)

print("Train ve test Excel dosyaları oluşturuldu!")
print("train_split.xlsx ve test_split.xlsx")

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

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

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

rmse_lr = rmse(y_test, y_pred_lr)
mape_lr = mape(y_test, y_pred_lr)

rmse_rf = rmse(y_test, y_pred_rf)
mape_rf = mape(y_test, y_pred_rf)

print("Linear Regression - RMSE :", rmse_lr)
print("Linear Regression - MAPE : {:.2f}%".format(mape_lr))

print("Random Forest       - RMSE :", rmse_rf)
print("Random Forest       - MAPE : {:.2f}%".format(mape_rf))
