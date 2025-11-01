import pandas as pd
import matplotlib.pyplot as plt

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


df = pd.read_excel("hi1_2025_09_eylul.xlsx")
df.columns = df.columns.str.strip()

print(">> Excel columns:")
for c in df.columns:
    print(" -", repr(c))



col_time = pick_col(df, ["Kayıt Tarihi", "Kayit Tarihi", "KayıtTarihi", "Tarih", "Date", "Timestamp"])
col_pm1  = pick_col(df, ["Pm1", "PM1"])
col_pm25 = pick_col(df, ["Pm2.5", "PM2.5", "Pm2_5", "PM2_5", "Pm25", "PM25"])
col_pm10 = pick_col(df, ["Pm10", "PM10"])
col_temp = pick_col(df, ["Sıcaklık(°C)", "Sicaklik(°C)", "Sıcaklık", "Sicaklik", "Sıcaklık (°C)", "Sicaklik (°C)", "Sicaklik °C"])
col_hum  = pick_col(df, ["Nem(%)", "Nem (%)", "Nem"])
col_voc  = pick_col(df, ["VOC", "Voc", "voc"])
col_dev  = pick_col(df, ["Cihaz", "Device", "cihaz"])

print("\n>> Selected columns:")
print(" time   :", col_time)
print(" pm1    :", col_pm1)
print(" pm2.5  :", col_pm25)
print(" pm10   :", col_pm10)
print(" temp   :", col_temp)
print(" humid  :", col_hum)
print(" VOC    :", col_voc)
print(" device :", col_dev)



if col_time is not None:
    df[col_time] = pd.to_datetime(df[col_time], errors="coerce")



pm_cols = [c for c in [col_pm1, col_pm25, col_pm10] if c is not None]
if pm_cols:
    plt.figure(figsize=(6,4))
    df[pm_cols].boxplot()
    plt.title("Box Plots of Particulate Matter (PM) Concentrations")
    plt.ylabel("µg/m³")
    plt.savefig("boxplots_pm.png", dpi=300, bbox_inches="tight")
    plt.close()
else:
    print("!! Warning: PM columns not found; box plot skipped.")



if col_temp is not None:
    plt.figure(figsize=(6,4))
    df[col_temp].dropna().hist(bins=15)
    plt.title("Temperature Distribution")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Frequency")
    plt.savefig("histogram_temperature.png", dpi=300, bbox_inches="tight")
    plt.close()
else:
    print("!! Warning: Temperature column not found; histogram skipped.")


if (col_time is not None) and (col_pm25 is not None):
    plt.figure(figsize=(9,4))
    plt.plot(df[col_time], df[col_pm25])
    plt.title("PM2.5 Levels Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("lineplot_pm25.png", dpi=300, bbox_inches="tight")
    plt.close()
else:
    print("!! Warning: Time or PM2.5 column missing; line plot skipped.")


if (col_temp is not None) and (col_pm25 is not None):
    plt.figure(figsize=(6,4))
    plt.scatter(df[col_temp], df[col_pm25], alpha=0.6)
    plt.title("PM2.5 vs Temperature")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.savefig("scatter_pm_temp.png", dpi=300, bbox_inches="tight")
    plt.close()
else:
    print("!! Warning: Temperature or PM2.5 column missing; scatter plot skipped.")


if col_voc is not None and df[col_voc].notna().any():
    plt.figure(figsize=(6,4))
    df[col_voc].value_counts().plot(kind="bar")
    plt.title("VOC Value Frequencies")
    plt.xlabel("VOC")
    plt.ylabel("Count")
    plt.savefig("bar_voc.png", dpi=300, bbox_inches="tight")
    plt.close()
elif col_dev is not None:
    plt.figure(figsize=(6,4))
    df[col_dev].value_counts().plot(kind="bar")
    plt.title("Device Frequency Distribution")
    plt.xlabel("Device")
    plt.ylabel("Count")
    plt.savefig("bar_device.png", dpi=300, bbox_inches="tight")
    plt.close()
else:
    print("!! Warning: Neither VOC nor device column found; bar chart skipped.")



print("\n The following visualizations were saved (if available):")
print(" - boxplots_pm.png")
print(" - histogram_temperature.png")
print(" - lineplot_pm25.png")
print(" - scatter_pm_temp.png")
print(" - bar_voc.png or bar_device.png")
