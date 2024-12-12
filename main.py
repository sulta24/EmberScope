import numpy as np
import tensorflow as tf
import pandas as pd
data = pd.read_parquet('with_meteo/Atlas_nerm_with_meteo_normalaized.parquet')
# Загрузка сохранённой модели
model = tf.keras.models.load_model('with_meteo/model_wm.keras')
input_features = ['lat', 'lon', 'landcover', 'month_sin', 'day_sin', 'avg_wspd', 'avg_wdir', 'avg_hum']
target_outputs = ['duration', 'expansion', 'speed', 'direction']
X = data[input_features].values
y = data[target_outputs].values

# Входные данные (вектор)
input_data = np.array([X[100]])
input_data_2 = np.array([X[1]])
print(input_data)
# Преобразование данных в ожидаемую форму (batch_size, sequence_length, num_features)
input_data = input_data.reshape((1, 1, 8))  # batch_size=1, sequence_length=1, num_features=8
input_data_2 = input_data_2.reshape((1, 1, 8))
out_data1 = np.array([y[100]])
out_data2 = np.array([y[1]])
# Убедимся, что размерность данных правильная
print("Shape of input data:", input_data.shape)

# Предсказание
predictions = model.predict(input_data)
predictions_2 = model.predict(input_data_2)
# Вывод предсказаний
print("Predictions:", predictions, predictions_2)
print(out_data1, out_data2)



# Реальные предсказания


# Пример выходных данных (предсказания нейронной сети)
predictions = [out_data1]




# Пример выходных данных нейронной сети (несколько предсказаний)


# Статистические параметры для каждой переменной (duration, speed, expansion, direction)
deviation_factors = {
    "duration": 0.15,   # отклонение 15% для duration
    "speed": 0.1,       # отклонение 10% для speed
    "expansion": 0.3,   # отклонение 30% для expansion
    "direction": 1      # отклонение 1 единица для direction
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
            deviation = np.random.randint(-deviation_factors["direction"], deviation_factors["direction"] + 10)

        corrected_prediction[j] += deviation

    return corrected_prediction

# Агрегируем предсказания (среднее значение для каждого столбца)
aggregated_prediction = np.mean(predictions, axis=1)[0]

# Применяем случайные отклонения
corrected_prediction = add_random_variation(aggregated_prediction, deviation_factors)

print("Откорректированное единственное предсказание:")
print(corrected_prediction[-1])
