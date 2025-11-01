# Gerekli kütüphaneleri yükle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veriyi yükle
df = pd.read_excel('/Users/yasemin/Desktop/hi1_2025_09_eylul 2.xlsx', index_col='Kayıt Tarihi')

# Sadece sayısal sütunları seç (opsiyonel ama genellikle gerekli)
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df_num = df[numerical_cols]

# Korelasyon matrisini hesapla
corr_matrix = df_num.corr()

# Korelasyon matrisini ısı haritası olarak çiz
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()
