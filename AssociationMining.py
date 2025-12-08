import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_excel("hi1_2025_09_eylul_discretized.xlsx")

cat_cols = ["Temp_cat", "Humid_cat", "PM25_cat", "PM10_cat"]
df_cat = df[cat_cols].copy()

transactions = pd.get_dummies(df_cat)

print("Transaction matrix shape:", transactions.shape)
print(transactions.head())

freq_itemsets = apriori(
    transactions,
    min_support=0.05,
    use_colnames=True,
    max_len=3
)

freq_itemsets = freq_itemsets.sort_values("support", ascending=False)
print("\nTop 10 frequent itemsets:")
print(freq_itemsets.head(10))

rules = association_rules(
    freq_itemsets,
    metric="confidence",
    min_threshold=0.6
)

rules = rules.sort_values(["lift", "confidence"], ascending=False)

interesting = rules[
    rules["consequents"].astype(str).str.contains("PM10_cat")
    | rules["consequents"].astype(str).str.contains("PM25_cat")
]

cols = ["antecedents", "consequents", "support", "confidence", "lift"]
print("\nTop 10 interesting rules (related to PM10/PM25):")
print(interesting[cols].head(10))

interesting[cols].to_excel("association_rules_pm.xlsx", index=False)
print("\nDone. 'association_rules_pm.xlsx' has been saved.")
