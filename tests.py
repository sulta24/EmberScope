import pandas as pd
import numpy as np
df = pd.read_parquet("DATA/Atlas_nerm_with_meteo.parquet")
pd.set_option('display.max_columns', None)
print(df.describe())
df['lat'] = (df['lat'] - min(df['lat'])) / (max(df['lat']) - min(df['lat']))
df['lon'] = (df['lon'] - min(df['lon'])) / (max(df['lon']) - min(df['lon']))
df['landcover'] = np.minimum(1, df['landcover'] / 10)
df['duration'] = np.minimum(1, df['duration'] / 31)
df['expansion'] = np.minimum(df['expansion'] / 10,1)
df['speed'] = np.minimum(df['speed'] / 10,1)
df['direction'] = df['direction'] / 8
df['avg_wspd'] = np.minimum(df['avg_wspd'] / 10, 1)
df['avg_wdir'] = (df['avg_wdir'] // 45) / 8
df['avg_hum'] = df['avg_hum'] / 100
print("AFTER NORMALIZING")
print(df.head())
df.to_parquet("with_meteo/Atlas_nerm_with_meteo_normalaized.parquet")

