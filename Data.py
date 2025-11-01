import pandas as pd
from matplotlib import pyplot as plt


file_name = "/Users/yasemin/Desktop/hi1_2025_09_eylul.xlsx"
df = pd.read_excel(file_name)


df['Kayıt Tarihi'] = pd.to_datetime(df['Kayıt Tarihi'])


columns_to_drop = ['Kayıt Yapan', 'CO', 'No2']
df_cleaned = df.drop(columns=columns_to_drop)


df_cleaned.set_index('Kayıt Tarihi', inplace=True)


df_resampled = df_cleaned.resample('15T').mean(numeric_only=True)

print("resampling successful")
df_resampled.to_excel("/Users/yasemin/Desktop/hi1_2025_09_eylul_15min.xlsx")
df_resampled.plot(figsize=(10,5), title="15-Minute Averaged Values (Resampled Data)")
plt.show()