import pandas as pd
df = pd.read_parquet("combined_atlas.parquet")
import numpy as np

print('-' * 300)
pd.set_option('display.max_columns', None)
print(df[['lat', 'lon', 'duration', 'expansion', 'speed', 'landcover']].head())

print(df.head())

print(df.describe())
print('-' * 300)
#Размер





# Преобразование колонки start_date в тип datetime
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')

# Проверка преобразования




# Извлекаем временные характеристики
df['year'] = df['start_date'].dt.year
df['month'] = df['start_date'].dt.month
df['day'] = df['start_date'].dt.day

# Циклическое кодирование можно использовать синусоидальное
# представление для учета их цикличности
# Это полезно, что бы избежать внезапных скачков в значениях
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)

# Удаляем оригинальную колонку start_date
df = df.drop(columns=['start_date'])
df = df.drop(columns=['year'])
df = df.drop(columns=['month'])
df = df.drop(columns=['day'])



# Новый порядок колонок
new_order = df.columns[[0,1,6,7,8,2,3,4,5]]  # Поменяем местами 'A' и 'C'

# Применяем новый порядок
df = df.reindex(columns=new_order)

#Колонки



# Применение преобразований к указанным столбцам единички
df['lat'] = (df['lat'] - min(df['lat'])) / (max(df['lat']) - min(df['lat']))
df['lon'] = (df['lon'] - min(df['lon'])) / (max(df['lon']) - min(df['lon']))
df['landcover'] = np.minimum(1, df['landcover'] / 10)
df['duration'] = np.minimum(1, df['duration'] / 31)
df['expansion'] = np.minimum(df['expansion'] / 10,1)
df['speed'] = np.minimum(df['speed'] / 10,1)
df['direction'] = df['direction'] / 8



print('+' * 300)
print(df.describe())
print('+' * 300)



# Проверка наличия пропущенных значений
print(df.isnull().sum())


df.to_parquet("Atlas_nerm.parquet")

