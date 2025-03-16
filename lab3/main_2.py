import numpy as np


class KohonenNetwork:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(output_size, input_size) * 0.01  # Малые случайные значения весов

    def train(self, X, epochs=100, learning_rate=0.1):
        for _ in range(epochs):
            for x in X:
                winner = np.argmin(np.linalg.norm(self.weights - x, axis=1))  # Нейрон-победитель
                self.weights[winner] += learning_rate * (x - self.weights[winner])

    def predict(self, x):
        return np.argmin(np.linalg.norm(self.weights - x, axis=1))
# Генерация бинарных данных (числа 0-9 в двоичном формате)
def generate_data():
    return np.array([[int(b) for b in f"{i:04b}"] for i in range(10)])

X = generate_data()

koh_net = KohonenNetwork(input_size=4, output_size=2)
koh_net.train(X)

# Тестирование
for i, x in enumerate(X):
    print(f"Число {i}: Класс {koh_net.predict(x)}")
