import pandas as pd
import matplotlib.pyplot as plt

dfs = []

for i in range(5, 9+1):
    df = pd.read_parquet(f"raw_ais/parquets_fish_only/2022-01-{i:02d}_fish_only.parquet", engine="pyarrow")
    dfs.append(df)

df_all_days = pd.concat(dfs, ignore_index=True)

mmsi_russian_trawler = 273418680

df_russian_trawler = df_all_days[df_all_days["mmsi"] == mmsi_russian_trawler].copy()
print(df_russian_trawler.head())
#df_russian_trawler.to_parquet("russian_svalbard_trawler.parquet", index=False)

plt.scatter(df_russian_trawler["lon"], df_russian_trawler["lat"], s=4)
plt.show()


df_cleaned_russian = pd.read_parquet("russian_svalbard_trawler_cleaned.parquet", engine="pyarrow")

plt.scatter(df_cleaned_russian["lon"], df_cleaned_russian["lat"], s=4)
plt.show()

