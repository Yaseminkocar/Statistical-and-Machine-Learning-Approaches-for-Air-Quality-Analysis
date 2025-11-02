import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

# 1) Dosyayı oku
# Excel ise:
df = pd.read_excel("hi1_2025_09_eylul.xlsx")   # dosya adını kendine göre değiştir

# 2) Virgüllü sayıları noktaya çevir (CO = '0,5' gibi olanlar için)
# Sayısal olması gereken sütunları seç
num_cols = [
    'Pm1', 'Pm2.5', 'Pm10',
    'CO', 'CO2', 'CH2O', 'O3', 'No2', 'Voc',
    'Sicaklik(°C)', 'Nem(%)', 'His. Sıcaklık(°C)'
]

for col in num_cols:
    # Bazı sütunlar zaten float olabilir, o yüzden errors='ignore'
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )

# 3) Gereksiz sütunları at
drop_cols = ['Kayıt No', 'Aktif Mi', 'Sıra No', 'Cihaz', 'Kayıt Yapan']
df = df.drop(columns=drop_cols, errors='ignore')

# 4) Tarih sütununu işler hale getir
df['Kayıt Tarihi'] = pd.to_datetime(df['Kayıt Tarihi'])
df['Hour'] = df['Kayıt Tarihi'].dt.hour
df['DayOfWeek'] = df['Kayıt Tarihi'].dt.dayofweek
df = df.drop(columns=['Kayıt Tarihi'])

# 5) X ve y'yi hazırla
# Bu sensörlerin hepsi sende var:
feature_cols = [
    'CO', 'CO2', 'CH2O', 'O3', 'No2', 'Voc',
    'Sicaklik(°C)', 'Nem(%)', 'His. Sıcaklık(°C)',
    'Hour', 'DayOfWeek'
]

X = df[feature_cols]
y = df['Pm2.5']   # hedef

# 6) Ölçekle
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7) En iyi 5 özelliği seç
selector = SelectKBest(score_func=f_regression, k=5)
selector.fit(X_scaled, y)

# 8) Sonuçları yazdır
scores = selector.scores_
selected_feats = [col for col, keep in zip(feature_cols, selector.get_support()) if keep]

print("Tüm özelliklerin F-score değerleri:")
for col, sc in zip(feature_cols, scores):
    print(f"{col:25} --> {sc:.3f}")

print("\nSeçilen en iyi 5 özellik:")
print(selected_feats)
