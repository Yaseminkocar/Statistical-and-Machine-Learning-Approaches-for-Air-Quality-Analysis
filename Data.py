import pandas as pd
from matplotlib import pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_dir, "hi1_2025_09_eylul.xlsx")

df = pd.read_excel(input_path)


df['Kayıt Tarihi'] = pd.to_datetime(df['Kayıt Tarihi'])

columns_to_drop = ['Kayıt Yapan', 'CO', 'No2']
df_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

df_cleaned.set_index('Kayıt Tarihi', inplace=True)

df_resampled = df_cleaned.resample('15min').mean(numeric_only=True)

print("Columns after resampling:")
print(df_resampled.columns.tolist())

def style_plot(title):
    plt.title(title, fontsize=18)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()


pm_cols = [col for col in df_resampled.columns if 'pm' in col.lower()]
plt.figure(figsize=(14,7))
df_resampled[pm_cols].plot(ax=plt.gca(), linewidth=2)
style_plot("Particulate Matter (PM) - 15 Minute Averages")
plt.savefig(os.path.join(base_dir, "fig_pm.pdf"))
plt.close()

temp_cols = [col for col in df_resampled.columns if 'sicak' in col.lower()]
plt.figure(figsize=(14,7))
df_resampled[temp_cols].plot(ax=plt.gca(), linewidth=2)
style_plot("Temperature Variables - 15 Minute Averages")
plt.savefig(os.path.join(base_dir, "fig_temp.pdf"))
plt.close()

humidity_cols = [col for col in df_resampled.columns if 'nem' in col.lower()]
plt.figure(figsize=(14,7))
df_resampled[humidity_cols].plot(ax=plt.gca(), linewidth=2)
style_plot("Humidity - 15 Minute Averages")
plt.savefig(os.path.join(base_dir, "fig_humidity.pdf"))
plt.close()

gas_keywords = ['co2', 'ch2o', 'o3', 'voc']
gas_cols = [col for col in df_resampled.columns if any(k in col.lower() for k in gas_keywords)]

plt.figure(figsize=(14,7))


ax1 = plt.gca()

colors = ['red', 'green', 'blue', 'orange']
axes = [ax1]

gas_keywords = ['co2', 'ch2o', 'o3', 'voc']
gas_cols = [col for col in df_resampled.columns if any(k in col.lower() for k in gas_keywords)]


df_resampled[gas_cols[0]].plot(ax=ax1, color=colors[0], linewidth=2, label=gas_cols[0])

for i in range(1, len(gas_cols)):
    ax_new = ax1.twinx()
    axes.append(ax_new)

    ax_new.spines["right"].set_position(("axes", 1 + (i*0.1)))

    df_resampled[gas_cols[i]].plot(ax=ax_new, color=colors[i], linewidth=2, label=gas_cols[i])


plt.title("Gas Pollutants (Separate Y-Scales for Clarity)", fontsize=18)


lines = []
labels = []

for ax in axes:
    line, label = ax.get_legend_handles_labels()
    lines += line
    labels += label

plt.legend(lines, labels, fontsize=12, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "fig_gases.pdf"))
plt.close()


print("All good — plots created successfully!")
