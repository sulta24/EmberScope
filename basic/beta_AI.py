import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib

# Загрузка данных из файла parquet
data = pd.read_parquet('Atlas_nerm.parquet')

# Масштабируем числовые признаки
input_features = ['lat', 'lon', 'landcover', 'month_sin', 'day_sin']
#отключаем скалер
#scaler = StandardScaler()
#data[input_features] = scaler.fit_transform(data[input_features])

# Определяем целевые переменные (targets)
target_outputs = ['duration', 'expansion', 'speed', 'direction']

# Разделим данные на X и y
X = data[input_features].values
y = data[target_outputs].values

# Преобразуем X в форму для LSTM: (samples, timesteps, features)
X = X.reshape(X.shape[0], 1, X.shape[1])  # 1 временной шаг

# Преобразуем данные в тип float32 для TensorFlow
X = X.astype(np.float32)
y = y.astype(np.float32)

# Устанавливаем конкретные гиперпараметры
lstm_units = 128
dropout_rate = 0.2
learning_rate = 0.01
delta = 1.0

# Создание модели
def create_model(lstm_units, dropout_rate, learning_rate, delta):
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(target_outputs)))  # Выходной слой

    optimizer = Adam(learning_rate=learning_rate)
    loss = Huber(delta=delta)
    model.compile(optimizer=optimizer, loss=loss)
    return model

model = create_model(lstm_units, dropout_rate, learning_rate, delta)

# Early stopping для предотвращения переобучения
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Обучение модели
history = model.fit(
    X, y,
    epochs=10,  # Увеличь количество эпох для более качественного обучения
    validation_split=0.2,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# График потерь
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Сохранение модели и scaler
model.save('model.keras')
#joblib.dump(scaler, 'scaler.pkl')

# Сохранение архитектуры модели
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
