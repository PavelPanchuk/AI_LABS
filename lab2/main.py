import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Подготовка данных
numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)  # 1 - нечётное, 0 - чётное

# Нормализация данных
numbers = numbers / 10

# Создание модели
model = Sequential([
    Dense(8, input_dim=1, activation='relu'),  # Первый скрытый слой с 8 нейронами
    Dense(4, activation='relu'),              # Второй скрытый слой с 4 нейронами
    Dense(1, activation='sigmoid')            # Выходной слой с сигмоидальной активацией
])

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(numbers, labels, epochs=100, verbose=1)

# Тестовые данные
test_numbers = np.array([6, 7, 11, 12], dtype=float) / 10  # Нормализация тестовых данных
predictions = model.predict(test_numbers)

# Вывод результата
for num, pred in zip(test_numbers * 10, predictions):
    print(f"Число {int(num)}: {'нечётное' if pred > 0.5 else 'чётное'}")

xyz=input("end")