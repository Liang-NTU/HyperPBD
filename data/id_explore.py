import csv
import pandas as pd

path = "./cosmetics_future.csv"
df = pd.read_csv(path)
ids = set(df[["CUSTOMER_ID"]].values.flatten())
gt_ids = set([i for i in range(24909)])
missing = list(set(gt_ids) - set(ids))

print(missing)