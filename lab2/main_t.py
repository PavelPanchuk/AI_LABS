class Perceptron:
    def __init__(self, num_of_inputs, threshold, learning_rate):
        self.weights = [0.0] * num_of_inputs
        self.threshold = threshold
        self.learning_rate = learning_rate

    def predict(self, inputs):
        if len(inputs) != len(self.weights):
            raise ValueError("mismatched number of inputs and weights")
        acc = sum(i * w for i, w in zip(inputs, self.weights))
        return 0.0 if acc < self.threshold else 1.0

    def update(self, updates):
        if len(updates) != len(self.weights):
            raise ValueError("mismatched number of updates and weights")
        for i in range(len(self.weights)):
            self.weights[i] += updates[i]

    def train(self, inputs, targets):
        for input_vec, target in zip(inputs, targets):
            output = self.predict(input_vec)
            error = target - output
            updates = [self.learning_rate * error * x for x in input_vec]
            self.update(updates)


def to_binary_list(x, length=4):
    #Преобразует десятичное число в двоичное, в список.
    binary_str = bin(x)[2:].zfill(length)
    return [int(digit) for digit in binary_str]


if __name__ == "__main__":
    # Инициализация перцептрона с 4 входами
    perceptron = Perceptron(num_of_inputs=4, threshold=0.5, learning_rate=1)
    training_inputs = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
    ]
    targets = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    # Обучение
    for _ in range(10):
        perceptron.train(training_inputs, targets)

    x = 4
    binary_vector = to_binary_list(x, length=4)
    print("Предсказание метки класса {} {}: {}".format(x,binary_vector, perceptron.predict(binary_vector)))
