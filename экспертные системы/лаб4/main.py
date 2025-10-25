import numpy as np

class HopfieldNetwork:
    def __init__(self, patterns):
        """
        Инициализация сети Хопфилда.

        Args:
            patterns (dict): Словарь, где ключи - названия образцов,
                             а значения - биполярные векторы (+1/-1).
        """
        self.patterns = patterns
        self.pattern_size = len(list(patterns.values())[0])
        self.num_patterns = len(patterns)
        
        # Инициализация весов по правилу Хебба
        self.weights = np.zeros((self.pattern_size, self.pattern_size))
        for pattern in patterns.values():
            x = np.array(pattern).reshape(-1, 1)  # Преобразуем в вектор-столбец
            self.weights += np.dot(x, x.T)
        self.weights /= self.pattern_size  # Нормализация
        np.fill_diagonal(self.weights, 0)  # Обнуление диагонали

    def predict(self, input_vector, max_iter=100):
        """
        Восстановление образца из входного вектора.

        Args:
            input_vector (np.array): Бинарный вектор (+1/-1) для распознавания.
            max_iter (int): Максимальное количество итераций обновления.

        Returns:
            str or None: Название ближайшего образца или None.
        """
        y = np.array(input_vector.copy())
        prev_y = np.zeros_like(y)
        
        # Асинхронное обновление с проверкой стабильности
        for _ in range(max_iter):
            prev_y[:] = y[:]
            # Обновление нейронов в случайном порядке
            order = np.random.permutation(self.pattern_size)
            for i in order:
                activation = np.dot(self.weights[i, :], y)
                y[i] = 1 if activation >= 0 else -1
            if np.array_equal(y, prev_y):
                break
        
        # Поиск ближайшего образца
        min_dist = np.inf
        best_name = None
        for name, pattern in self.patterns.items():
            dist = np.sum(y != pattern)
            if dist < min_dist:
                min_dist = dist
                best_name = name
        return best_name if min_dist <= self.pattern_size else None

if __name__ == '__main__':
    # Примеры образцов (те же, что и для сети Хэмминга)
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

    # Создание сети
    hopfield_net = HopfieldNetwork(letter_patterns)

    # Тестовые данные
    test_inputs = {
        'Чистая A': np.array([+1, +1, +1, +1, -1, +1, +1, +1, +1, +1, -1, +1, +1, -1, +1]),
        'Зашумленная A': np.array([+1, +1, -1, +1, -1, +1, +1, +1, +1, +1, +1, +1, +1, -1, +1]),
        'Чистая B': np.array([+1, +1, -1, +1, -1, +1, +1, +1, -1, +1, -1, +1, +1, +1, -1]),
        'Чистая C': np.array([+1, +1, +1, +1, -1, -1, +1, -1, -1, +1, -1, -1, +1, +1, +1]),
        'Неизвестный символ': np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    }

    # Проверка работы сети
    for name, input_data in test_inputs.items():
        prediction = hopfield_net.predict(input_data)
        print(f"Вход: {name}, Распознано: {prediction}")
