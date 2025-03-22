import numpy as np

class HebbianNetwork:
    def __init__(self, input_size, bias=True, learning_rate=0.1):
        """
        Инициализация сети.
        :param input_size: число входов (без учета bias)
        :param bias: использовать ли дополнительный вход для смещения (bias)
        :param learning_rate: скорость обучения
        """
        self.bias = bias
        self.learning_rate = learning_rate
        # Если bias=True, добавляем дополнительный вес для bias
        self.weights = np.random.randn(input_size + int(bias)) * 0.01

    def train(self, X, y, epochs=20):
        """
        Обучает сеть с использованием модифицированного правила Хебба (дельта-правило).
        :param X: матрица входных данных (n_samples, input_size)
        :param y: вектор целевых значений (n_samples,) (для четности: 1 - четное, 0 - нечетное)
        :param epochs: количество эпох обучения
        """
        for epoch in range(epochs):
            for x, target in zip(X, y):
                # Если используется bias, дополняем вектор x единицей
                x_aug = np.insert(x, 0, 1) if self.bias else x
                # Прогноз (0 или 1)
                output = self.predict(x)
                # Вычисляем ошибку
                error = target - output
                # Обновляем веса по правилу дельты: Δw = η * error * x
                self.weights += self.learning_rate * error * x_aug
            print(f"Эпоха {epoch+1}/{epochs}, веса: {self.weights}")

    def predict(self, x):
        """
        Делает предсказание для одного входного вектора.
        :param x: вектор входных данных
        :return: 1, если предсказано «четное» (активация > 0), иначе 0
        """
        x_aug = np.insert(x, 0, 1) if self.bias else x
        activation = np.dot(self.weights, x_aug)
        # Пороговая функция: если активация положительна, считаем, что число четное
        return 1 if activation > 0 else 0

def generate_data():
    """
    Генерирует данные:
      - X: бинарное представление чисел от 0 до 9 (4 бита)
      - y: метки четности (1, если число четное, 0 если нечетное)
    """
    X = np.array([[int(b) for b in f"{i:04b}"] for i in range(10)])
    y = np.array([1 if i % 2 == 0 else 0 for i in range(10)])
    return X, y

if __name__ == "__main__":
    # Фиксируем случайное состояние для воспроизводимости
    np.random.seed(42)

    # Генерация данных и нормализация: преобразуем бинарное представление из {0,1} в диапазон [-1, 1]
    X, y = generate_data()
    X_normalized = 2 * X - 1

    # Инициализируем сеть с bias и задаем learning_rate
    hebb_net = HebbianNetwork(input_size=4, bias=True, learning_rate=0.1)
    hebb_net.train(X_normalized, y, epochs=20)

    # Тестирование сети
    print("\nТестирование сети:")
    for i, x in enumerate(X):
        x_normalized = 2 * x - 1  # нормализация входа
        prediction = hebb_net.predict(x_normalized)
        result = "Четное" if prediction == 1 else "Нечетное"
        print(f"Число {i} ({x.tolist()}): {result}")
