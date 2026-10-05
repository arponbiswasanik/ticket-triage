import pandas as pd

df = pd.read_csv("data/raw/tickets_raw.csv")

print("Shape:", df.shape)
print("\nColumns and dtypes:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isna().sum())

for col in df.columns:
    if df[col].nunique() < 25:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts(dropna=False))

print("\nSample rows:")
print(df.head(2).T)