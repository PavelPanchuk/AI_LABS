import numpy as np

class HebbianNetwork:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * 0.01  # Инициализация весов малыми случайными значениями

    def train(self, X, y):
        for x, target in zip(X, y):
            self.weights += x * target  # Обновление весов с учетом цели

    def predict(self, x):
        # Прогнозируем, четное ли число
        return np.dot(self.weights, x) > 0  # Четное или нечетное

# Данные: бинарное представление чисел от 0 до 9
def generate_data():
    return np.array([[int(b) for b in f"{i:04b}"] for i in range(10)]), np.array([i % 2 for i in range(10)])

X, y = generate_data()

# Нормализация данных (входы: [-1, 1])
X_normalized = 2 * X - 1  # Преобразуем данные в диапазон [-1, 1]

hebb_net = HebbianNetwork(input_size=4)
hebb_net.train(X_normalized, y)

# Тестирование
for i, x in enumerate(X):
    x_normalized = 2 * x - 1  # Нормализуем вход для предсказания
    prediction = hebb_net.predict(x_normalized)
    print(f"Число {i}: {'Четное' if prediction else 'Нечетное'}")
