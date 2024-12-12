from flask import Flask, request, jsonify
import numpy as np
import math
import random
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
# Инициализация Flask
from flask_cors import CORS
# Инициализация Flask и CORS
app = Flask(__name__)
CORS(app)  # Это позволяет принимать запросы с любых источников


# Загрузка сохраненной модели и scaler
import joblib
wind_direction = 0
wind_speed = 0
# Загружаем объект scaler

df_build_data_NN = pd.read_parquet("with_meteo/Atlas_nerm_with_meteo_normalaized.parquet")
# Проверьте тип объекта

# Загрузка модели нейронной сети (предполагается, что она сохранена в формате .h5)
model = tf.keras.models.load_model('with_meteo/model_wm.keras')


# Функция для нормализации входных данных
# Функция для нормализации входных данных
def normalize_input(data):
    global wind_direction
    global wind_speed
    # Преобразуем данные в правильный формат
    # Ваши данные (например, широта, долгота, ландшафт и т.д.)
    input_data = np.array(
        [[(data['lat']- min(df_build_data_NN['lat'])) / (max(df_build_data_NN['lat']) - min(df_build_data_NN['lat'])),
          (data['lon']- min(df_build_data_NN['lon'])) / (max(df_build_data_NN['lon']) - min(df_build_data_NN['lon'])),
          np.minimum(float(data['landcover']) / 10, 1),
          data['month_sin'], data['day_sin'], np.minimum(float(data['avg_wspd']) / 240, 1), (float(data['avg_wdir']) // 45) / 8, float(data['avg_hum']) / 100]])



    # Проверка, что scaler это StandardScaler
    wind_direction = float(data['avg_wdir'])
    if float(data['avg_wspd']) <= 0:
        wind_speed = float(random.random() * 60)
    else:
        wind_speed = float(data['avg_wspd'])
    normalized_data = input_data.reshape((1, 1, 8))
    return normalized_data



# Функция для обратной нормализации

def denormalize_output(predictions):
    # Преобразуем значения обратно в исходные масштабы

    predictions = [predictions.reshape(1,4)]


    # Пример выходных данных нейронной сети (несколько предсказаний)

    # Статистические параметры для каждой переменной (duration, speed, expansion, direction)
    deviation_factors = {
        "duration": 0.15,  # отклонение 15% для duration
        "speed": 0.1,  # отклонение 10% для speed
        "expansion": 0.3,  # отклонение 30% для expansion
        "direction": 1  # отклонение 1 единица для direction
    }

    # Функция для добавления случайных отклонений
    def add_random_variation(prediction, deviation_factors):
        """
        Добавляем случайные отклонения в одно предсказание.
        deviation_factors - это словарь с максимальными отклонениями для каждой переменной.
        """
        corrected_prediction = prediction.copy()

        for j in range(len(prediction)):  # Для каждой переменной (duration, speed, expansion, direction)
            if j == 0:  # duration
                deviation = np.random.uniform(-deviation_factors["duration"], deviation_factors["duration"])
            elif j == 1:  # speed
                deviation = np.random.uniform(-deviation_factors["speed"], deviation_factors["speed"])
            elif j == 2:  # expansion
                deviation = np.random.uniform(-deviation_factors["expansion"], deviation_factors["expansion"])
            elif j == 3:  # direction
                deviation = np.random.randint(-deviation_factors["direction"], deviation_factors["direction"] + 1)

            corrected_prediction[j] += deviation

        return corrected_prediction

    # Агрегируем предсказания (среднее значение для каждого столбца)
    aggregated_prediction = np.mean(predictions, axis=1)[0]

    # Применяем случайные отклонения
    corrected_prediction = add_random_variation(aggregated_prediction, deviation_factors)
    predictions = corrected_prediction
    print(predictions)
    denormalized_data = {
        'duration': math.sqrt(predictions[0] ** 2) * 31 // 0.01 / 100,
        'expansion': math.sqrt(predictions[1] ** 2) * 10 // 0.0001 / 10000,
        'speed': random.uniform(0.1 * wind_speed, 0.15 * wind_speed) // 0.0001 / 10000,
        'direction': random.uniform(0.8 * wind_direction, 1.2 * wind_direction) // 0.0001 / 10000
    }

    print(denormalized_data)
    return denormalized_data


# API для обработки данных с клиента
@app.route('/process_fire_data', methods=['POST'])
def process_fire_data():
    data = request.json
    print("Data processed successfully", data)
    # Нормализуем данные
    normalized_data = normalize_input(data)

    # Получаем предсказания от нейронной сети
    predictions = model.predict(normalized_data)

    # Обратная нормализация
    result = denormalize_output(predictions)
    result = {key: float(value) for key, value in result.items()}

    # Отправляем результат обратно клиенту
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
