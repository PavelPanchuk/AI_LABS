import numpy as np

class HebbianNetwork:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * 0.01  # Инициализация весов малыми случайными значениями

    def train(self, X):
        for x in X:
            self.weights += x * x  # Правило Хэбба

    def predict(self, x):
        return np.dot(self.weights, x) > 0  # Четное или нечетное

# Данные: бинарное представление чисел от 0 до 9
def generate_data():
    return np.array([[int(b) for b in f"{i:04b}"] for i in range(10)]), np.array([i % 2 for i in range(10)])

X, y = generate_data()
hebb_net = HebbianNetwork(input_size=4)
hebb_net.train(X)

# Тестирование
for i, x in enumerate(X):
    print(f"Число {i}: {'Четное' if hebb_net.predict(x) else 'Нечетное'}")
