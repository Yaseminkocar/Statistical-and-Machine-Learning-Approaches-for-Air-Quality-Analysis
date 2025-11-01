import pandas as pd
from matplotlib import pyplot as plt

# 1. Veri setini yükle (Excel'den okuma)
file_name = "/Users/yasemin/Desktop/hi1_2025_09_eylul 2.xlsx"
df = pd.read_excel(file_name)

# --- AŞAMA 2.1: ÖN HAZIRLIK ve TEMİZLEME ---

# 2. Tür Dönüşümü
df['Kayıt Tarihi'] = pd.to_datetime(df['Kayıt Tarihi'])

# 3. Gereksiz sütunları at
columns_to_drop = ['Kayıt Yapan', 'CO', 'No2']
df_cleaned = df.drop(columns=columns_to_drop)

# 4. İndeks ataması
df_cleaned.set_index('Kayıt Tarihi', inplace=True)

# 5. Resample ve ortalama al
df_resampled = df_cleaned.resample('15T').mean(numeric_only=True)

print("Yeniden örnekleme başarıyla tamamlandı.")
df_resampled.to_excel("/Users/yasemin/Desktop/hi1_2025_09_eylul_15min.xlsx")
df_resampled.plot(figsize=(10,5), title="15-Minute Averaged Values (Resampled Data)")
plt.show()