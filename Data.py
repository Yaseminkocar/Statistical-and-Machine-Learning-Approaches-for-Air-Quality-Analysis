import pandas as pd
from matplotlib import pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

input_path = os.path.join(base_dir, "hi1_2025_09_eylul.xlsx")

df = pd.read_excel(input_path)


df['Kayıt Tarihi'] = pd.to_datetime(df['Kayıt Tarihi'])

columns_to_drop = ['Kayıt Yapan', 'CO', 'No2']
df_cleaned = df.drop(columns=columns_to_drop)
df_cleaned.set_index('Kayıt Tarihi', inplace=True)

df_resampled = df_cleaned.resample('15T').mean(numeric_only=True)

print(" Resampling successful")

output_path = os.path.join(base_dir, "hi1_2025_09_eylul_15min.xlsx")
df_resampled.to_excel(output_path)

df_resampled.plot(figsize=(10,5), title="15-Minute Averaged Values (Resampled Data)")
plt.tight_layout()
plt.show()
