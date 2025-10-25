import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Подготовка данных
numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=int)
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)  # 1 - нечётное, 0 - чётное

# --- Преобразуем числа в двоичный формат (4 бита на вход) ---
binary_numbers = np.array([[int(b) for b in format(n, '04b')] for n in numbers], dtype=float)

# Создание модели
model = Sequential([
    Dense(8, input_dim=4, activation='relu'),  # теперь вход — 4 бита
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(binary_numbers, labels, epochs=200, verbose=1)

# --- Тестовые данные ---
test_numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=int)
test_binary = np.array([[int(b) for b in format(n, '04b')] for n in test_numbers], dtype=float)
predictions = model.predict(test_binary)

# --- Вывод результата ---
for num, bits, pred in zip(test_numbers, test_binary, predictions):
    print(f"Число {int(num):2d} ({''.join(map(str, bits))}): {'нечётное' if pred >= 0.5 else 'чётное'} ({pred[0]:.2f})")

xyz = input("end")
