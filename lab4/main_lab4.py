import numpy as np
import matplotlib.pyplot as plt

# Функция активации (сигмоид) и его производная
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Генерация данных (полином 3-й степени)
np.random.seed(42)
X = np.linspace(-1, 1, 100).reshape(-1, 1)  # Входные данные
Y = 2*X**3 - 3*X**2 + 0.5*X + 1  # Полином: y = 2x^3 - 3x^2 + 0.5x + 1
Y += np.random.normal(0, 0.1, size=Y.shape)  # Добавляем шум

# Параметры нейросети
input_size = 1  # Входной слой (X)
hidden_size = 10  # Скрытый слой
output_size = 1  # Выходной слой
learning_rate = 0.1  # Скорость обучения
epochs = 5000  # Количество итераций

# Инициализация весов
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

# Обучение сети
for epoch in range(epochs):
    # Прямое распространение
    hidden_input = np.dot(X, W1)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2)
    final_output = final_input  # Линейный выход

    # Вычисление ошибки
    error = Y - final_output

    # Обратное распространение
    d_output = error  # Производная ошибки
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)

    # Обновление весов
    W2 += hidden_output.T.dot(d_output) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate

# Предсказание
X_test = np.linspace(-1, 1, 100).reshape(-1, 1)
hidden_test = sigmoid(np.dot(X_test, W1))
Y_pred = np.dot(hidden_test, W2)

# График
plt.scatter(X, Y, label="Исходные данные")
plt.plot(X_test, Y_pred, color="red", label="Предсказания НС")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Интерполяция полиномом нейросетью")
plt.show()
