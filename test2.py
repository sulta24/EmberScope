import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import MeanAbsoluteError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_parquet('with_meteo/Atlas_nerm_with_meteo_normalaized.parquet')

# Масштабируем числовые признаки
input_features = ['lat', 'lon', 'landcover', 'month_sin', 'day_sin', 'avg_wspd', 'avg_wdir', 'avg_hum']
X = data[input_features].values  # Входные признаки

# Определяем целевые переменные
target_outputs = ['duration', 'expansion', 'speed', 'direction']
y = data[target_outputs].values  # Целевые переменные

# Масштабируем данные
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Преобразуем данные в форму для LSTM: (samples, time_steps, features)
# Поскольку у нас один временной шаг, данные будут иметь форму (samples, 1, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Разделение на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Гиперпараметры
lstm_units = 160
dropout_rate = 0.4
learning_rate = 0.005650955718010664
delta = 1.0

# Создание модели
model = Sequential()

# Входной слой (для данных формы (samples, 1, 8))
model.add(LSTM(lstm_units, return_sequences=True, input_shape=(1, 8), activation='tanh',
               recurrent_activation='sigmoid', use_bias=True, dropout=0.0, recurrent_dropout=0.0))

# Слой Dropout
model.add(Dropout(dropout_rate))

# Полносвязный слой с 64 нейронами
model.add(Dense(64, activation='relu'))

# Выходной слой с 4 нейронами (для 4 целевых переменных)
model.add(Dense(4, activation='linear'))

# Компиляция модели с использованием оптимизатора Adam и потерь Huber
optimizer = Adam(learning_rate=learning_rate)
loss = Huber(delta=delta)

model.compile(optimizer=optimizer, loss=loss, metrics=[MeanAbsoluteError(name="mae")])

# Вывод информации о модели
model.summary()

# Обучение модели
history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_split=0.2, verbose=1)

# Оценка модели
val_loss, val_mae = model.evaluate(X_test, y_test, verbose=1)
print(f'Validation Loss: {val_loss}, Validation MAE: {val_mae}')

# График обучения
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Прогнозы на тестовых данных
y_pred = model.predict(X_test)

# Обратное масштабирование (если нужно)
y_pred_rescaled = scaler_y.inverse_transform(y_pred)
y_test_rescaled = scaler_y.inverse_transform(y_test)

# Пример вывода предсказанных и реальных значений
print("Predicted Values: ", y_pred_rescaled[:5])
print("True Values: ", y_test_rescaled[:5])
