import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "hi1_2025_09_eylul.xlsx")

df = pd.read_excel(file_path, index_col='Kayıt Tarihi')

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df_num = df[numerical_cols]

corr_matrix = df_num.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()

output_path = os.path.join(base_dir, "correlation_heatmap.pdf")
plt.savefig(output_path, format='pdf')

plt.close()
print(f"PDF created: {output_path}")
