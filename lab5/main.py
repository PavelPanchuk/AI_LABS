import numpy as np

class HammingNetwork:
    def __init__(self, patterns):
        """
        Инициализация сети Хэмминга.

        Args:
            patterns (dict): Словарь, где ключи - это названия букв,
                             а значения - это бинарные векторы (+1/-1)
                             представляющие эти буквы.
        """
        self.patterns = patterns
        self.num_patterns = len(patterns)
        self.pattern_size = len(list(patterns.values())[0])
        self.weights_layer1 = np.array(list(patterns.values())) / 2
        self.biases_layer1 = np.ones(self.num_patterns) * self.pattern_size / 2
        self.epsilon = 0.1  # Параметр для Maxnet

    def predict(self, input_vector):
        """
        Распознавание входного вектора.

        Args:
            input_vector (np.array): Бинарный вектор (+1/-1) входного сигнала.

        Returns:
            str or None: Название распознанной буквы или None, если распознавание не удалось.
        """
        if len(input_vector) != self.pattern_size:
            raise ValueError("Размер входного вектора не соответствует размеру образцов.")

        # Первый слой: вычисление расстояния Хэмминга (в обратной интерпретации)
        output_layer1 = np.dot(self.weights_layer1, input_vector) + self.biases_layer1

        # Второй слой (Maxnet): нахождение нейрона с максимальной активацией
        output_layer2 = output_layer1.copy()
        while True:
            previous_output = output_layer2.copy()
            for i in range(self.num_patterns):
                sum_inhibitory = 0
                for j in range(self.num_patterns):
                    if i != j:
                        sum_inhibitory += max(0, output_layer2[j])
                output_layer2[i] = max(0, output_layer1[i] - self.epsilon * sum_inhibitory)

            # Проверка на сходимость
            if np.array_equal(output_layer2, previous_output):
                break

        # Определение победителя
        winner_index = np.argmax(output_layer2)
        if output_layer2[winner_index] > 0:
            return list(self.patterns.keys())[winner_index]
        else:
            return None

if __name__ == '__main__':
    # Пример представления букв (простые шаблоны 5x3)
    # +1 представляет "включенный" пиксель, -1 - "выключенный"

    letter_patterns = {
        'A': np.array([+1, +1, +1,
                       +1, -1, +1,
                       +1, +1, +1,
                       +1, -1, +1,
                       +1, -1, +1]),
        'B': np.array([+1, +1, -1,
                       +1, -1, +1,
                       +1, +1, -1,
                       +1, -1, +1,
                       +1, +1, -1]),
        'C': np.array([+1, +1, +1,
                       +1, -1, -1,
                       +1, -1, -1,
                       +1, -1, -1,
                       +1, +1, +1]),
        'D': np.array([+1, +1, -1,
                       +1, -1, +1,
                       +1, -1, +1,
                       +1, -1, +1,
                       +1, +1, -1])
    }

    # Создание сети Хэмминга
    hamming_net = HammingNetwork(letter_patterns)

    # Тестовые входные данные
    test_inputs = {
        'Чистая A': np.array([+1, +1, +1, +1, -1, +1, +1, +1, +1, +1, -1, +1, +1, -1, +1]),
        'Зашумленная A': np.array([+1, +1, -1, +1, -1, +1, +1, +1, +1, +1, +1, +1, +1, -1, +1]),
        'Чистая B': np.array([+1, +1, -1, +1, -1, +1, +1, +1, -1, +1, -1, +1, +1, +1, -1]),
        'Чистая C': np.array([+1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, -1, +1, +1, +1]),
        'Неизвестный символ': np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    }

    # Тестирование сети
    for name, input_data in test_inputs.items():
        prediction = hamming_net.predict(input_data)
        print(f"Вход: {name}, Распознано: {prediction}")
xyz=input("end")