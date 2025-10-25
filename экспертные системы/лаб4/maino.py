import numpy as np

class HopfieldNetwork:
    def __init__(self, patterns):
        self.patterns = patterns
        self.pattern_size = len(list(patterns.values())[0])
        self.num_patterns = len(patterns)

        # Правило Хебба с нормировкой
        self.weights = np.zeros((self.pattern_size, self.pattern_size))
        for pattern in patterns.values():
            x = np.array(pattern).reshape(-1, 1)
            self.weights += np.dot(x, x.T)
        self.weights /= self.pattern_size
        np.fill_diagonal(self.weights, 0)

    def energy(self, state):
        return -0.5 * np.dot(state, np.dot(self.weights, state))

    def predict(self, input_vector, max_iter=100):
        y = np.array(input_vector.copy())

        for _ in range(max_iter):
            prev_y = y.copy()
            for i in np.random.permutation(self.pattern_size):
                activation = np.dot(self.weights[i, :], y)
                y[i] = 1 if activation >= 0 else -1
            if np.array_equal(y, prev_y):  # стабильность
                break

        # Проверка на совпадение с образцами
        for name, pattern in self.patterns.items():
            if np.array_equal(y, pattern):
                return name

        # Если точного совпадения нет — ищем ближайший
        best_name = None
        max_similarity = -np.inf
        for name, pattern in self.patterns.items():
            similarity = np.dot(y, pattern)
            if similarity > max_similarity:
                max_similarity = similarity
                best_name = name

        return best_name if max_similarity > 0 else None


if __name__ == '__main__':
    letter_patterns = {
        'A': np.array([+1, +1, +1,
                       +1, -1, +1,
                       +1, +1, +1,
                       +1, -1, +1,
                       +1, -1, +1]),

        '3': np.array([+1, +1, +1,
                       -1, -1, +1,
                       +1, +1, +1,
                       -1, -1, +1,
                       +1, +1, +1]),

        # 'C': np.array([+1, +1, +1,
        #                +1, -1, -1,
        #                +1, -1, -1,
        #                +1, -1, -1,
        #                +1, +1, +1]),

        # 'D': np.array([+1, +1, -1,
        #                +1, -1, +1,
        #                +1, -1, +1,
        #                +1, -1, +1,
        #                +1, +1, -1])
    }

    hopfield_net = HopfieldNetwork(letter_patterns)

    test_inputs = {
        'Чистая A': np.array([+1, +1, +1,
                              +1, -1, +1,
                              +1, +1, +1,
                              +1, -1, +1,
                              +1, -1, +1]),
        'Зашумленная A': np.array([+1, +1, -1,
                                   +1, -1, +1,
                                   +1, +1, +1,
                                   +1, -1, +1,
                                   +1, -1, +1]),
        'Чистая 3': np.array([+1, +1, +1,
                              -1, -1, +1,
                              +1, +1, +1,
                              -1, -1, +1,
                              +1, +1, +1]),
        'Чистая C': np.array([+1, +1, +1,
                              +1, -1, -1,
                              +1, -1, -1,
                              +1, -1, -1,
                              +1, +1, +1]),
        'Неизвестный символ': np.array([-1]*15)
    }

    for name, input_data in test_inputs.items():
        prediction = hopfield_net.predict(input_data)
        print(f"Вход: {name}, Распознано: {prediction}")
