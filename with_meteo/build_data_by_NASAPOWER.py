
import pandas as pd
import numpy as np
from datetime import timedelta
import requests
import json

# Загрузка данных
df = pd.read_parquet("combined_atlas.parquet")[0:-1: 1200]
pd.set_option('display.max_columns', None)
# Проверка типов данных и приведение к нужным
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
counter = 0
base_url = (
    "https://power.larc.nasa.gov/api/temporal/daily/point?"
    "parameters=WS10M,WD10M,RH2M&community=RE&longitude={longitude}&latitude={latitude}"
    "&start=20230101&end=20230131&format=JSON"
)
# Функция для получения метео-данных за период
import requests
import json
from datetime import datetime, timedelta

def get_weather_data(latitude, longitude, start_date, duration):
    global counter
    """
    Функция для получения данных о погоде с сайта NASA POWER
    по координатам, стартовой дате и продолжительности периода.

    Parameters:
    - latitude: широта (float)
    - longitude: долгота (float)
    - start_date: стартовая дата (datetime)
    - duration: продолжительность в днях (int)

    Returns:
    - dict с данными о скорости ветра, направлении ветра и влажности
    """

    # Вычисление конечной даты
    end_date = start_date + timedelta(days=duration)

    # Преобразование дат в формат YYYYMMDD
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    # Формируем URL для запроса
    base_url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point?"
        "parameters=WS10M,WD10M,RH2M&community=RE&longitude={longitude}&latitude={latitude}"
        "&start={start}&end={end}&format=JSON"
    )

    api_request_url = base_url.format(longitude=longitude, latitude=latitude, start=start_str, end=end_str)

    # Выполняем запрос к API
    response = requests.get(api_request_url, timeout=30)
    print(f"URL запроса: {api_request_url}")
    print(f"Статус возвращения: {response.status_code}")
    response.raise_for_status()  # Проверка на ошибки

    # Загружаем данные из ответа
    data = response.json()

    # Извлекаем данные о погоде
    wind_speed = data["properties"]["parameter"]["WS10M"]  # Скорость ветра
    wind_direction = data["properties"]["parameter"]["WD10M"]  # Направление ветра
    humidity = data["properties"]["parameter"]["RH2M"]  # Влажность
    print('Текущая итерация:',counter)
    counter += 1

    return wind_speed, wind_direction, humidity




# Пример использования функции
  # Продолжительность в днях







# Итерация по строкам в DataFrame и добавление метео-данных
weather_data = []
for _, row in df.iterrows():
    lat = row['lat']
    lon = row['lon']
    start_date = row['start_date']
    duration = row['duration']

    # Получаем метео-данные за период
    data_wspd, data_wdir, data_hum = get_weather_data(lat, lon, start_date, duration)
    avg_wspd = sum(list(data_wspd.values())) / len(list(data_wspd.values()))

    avg_wdir = sum(list(data_wdir.values())) / len(list(data_wdir.values()))
    avg_hum = sum(list(data_hum.values())) / len(list(data_hum.values()))
    print('Полученные значения:',avg_wspd, avg_wdir, avg_hum)
    # Добавляем данные в список
    weather_data.append([avg_wspd, avg_wdir, avg_hum])



weather_df = pd.DataFrame(weather_data, columns=['avg_wspd', 'avg_wdir', 'avg_hum'])
print('Дата с резуьтатами:')
print(weather_df.head())
# Преобразуем список в DataFrame и добавляем к исходному DataFrame
df.index = [i for i in range(len(weather_df))]
df = pd.concat([df, weather_df], axis=1)

# Проверка результата
print("Проверка результата:")
print(df.head())
print(df.describe())

#Нормалировка
df['year'] = df['start_date'].dt.year
df['month'] = df['start_date'].dt.month
df['day'] = df['start_date'].dt.day

# Циклическое кодирование можно использовать синусоидальное
# представление для учета их цикличности
# Это полезно, что бы избежать внезапных скачков в значениях
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)


df = df.drop(columns=['year'])
df = df.drop(columns=['month'])
df = df.drop(columns=['day'])

print(df.head())

df.to_parquet("Atlas_nerm_with_meteo.parquet")



import os
import time
# Ваш код программы
print("Программа выполнена.")
time.sleep(10)
# Для Windows
if os.name == 'nt':
    os.system("shutdown /s /f /t 1")