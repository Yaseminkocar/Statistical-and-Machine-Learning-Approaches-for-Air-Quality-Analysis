import pandas as pd
from sklearn.model_selection import train_test_split

#encoded data has to be used
df = pd.read_excel("hi1_2025_09_eylul_encoded_with_PM10cat.xlsx")


print(df["PM10_cat"].value_counts())

#split 80% train, 20% test
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["PM10_cat"]
)


train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


train_df.to_excel("trainingSet_Classification.xlsx", index=False)
test_df.to_excel("testSet_Classification.xlsx", index=False)

print("Done! Saved:")
print(" - trainingSet_Classification.xlsx")
print(" - testSet_Classification.xlsx")
