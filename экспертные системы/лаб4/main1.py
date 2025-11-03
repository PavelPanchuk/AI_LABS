import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np

# --- Оригинальный класс сети Хэмминга ---
class HammingNetwork:
    def __init__(self, patterns):
        self.patterns = patterns
        self.num_patterns = len(patterns)
        self.pattern_size = len(list(patterns.values())[0])

        # Слой 1
        self.weights_layer1 = np.array(list(patterns.values())) / 2
        self.biases_layer1 = np.ones(self.num_patterns) * self.pattern_size / 2

        # Слой 2 (Maxnet)
        self.epsilon = 1 / (2 * self.num_patterns)

    def predict(self, input_vector):
        if len(input_vector) != self.pattern_size:
            raise ValueError("Размер входного вектора не соответствует размеру образцов.")

        output_layer1 = np.dot(self.weights_layer1, input_vector) + self.biases_layer1
        output_layer2 = output_layer1.copy()

        while True:
            prev = output_layer2.copy()
            for i in range(self.num_patterns):
                inhibition = np.sum(np.maximum(0, output_layer2)) - max(0, output_layer2[i])
                output_layer2[i] = max(0, output_layer1[i] - self.epsilon * inhibition)
            if np.allclose(output_layer2, prev, atol=1e-6):
                break

        winner_index = np.argmax(output_layer2)
        return list(self.patterns.keys())[winner_index] if output_layer2[winner_index] > 0 else None

# --- GUI функция ---
def run_hamming():
    train_path = filedialog.askopenfilename(title="Выбери обучающий Excel", filetypes=[("Excel", "*.xlsx *.xls")])
    if not train_path:
        return
    test_path = filedialog.askopenfilename(title="Выбери тестовый Excel", filetypes=[("Excel", "*.xlsx *.xls")])
    if not test_path:
        return

    try:
        # --- Загружаем обучающие данные ---
        df_train = pd.read_excel(train_path, header=None)
        labels = df_train.iloc[-1].dropna().values
        df_train = df_train.iloc[:-1]

        patterns = {}
        for idx, label in enumerate(labels):
            # Преобразуем в числовые значения и +1/-1
            col_values = df_train.iloc[:, idx*3:(idx+1)*3].apply(pd.to_numeric, errors='coerce').fillna(0).values.flatten()
            vector = np.where(col_values == 0, -1, +1)
            patterns[label] = vector

        # Создаем сеть
        net = HammingNetwork(patterns)

        # --- Загружаем тестовый символ ---
        df_test = pd.read_excel(test_path, header=None)
        test_vector = np.where(df_test.values.flatten() == 0, -1, +1)

        # --- Распознаем ---
        result = net.predict(test_vector)

        # --- Очистка окна ---
        for widget in frame_main.winfo_children():
            widget.destroy()

        # --- Вывод результатов ---
        tk.Label(frame_main, text="Результат распознавания:", font=("Arial", 14)).pack(pady=10)
        text_box = tk.Text(frame_main, width=70, height=20, font=("Consolas", 12))
        text_box.pack(pady=10)

        # Тестовая последовательность
        seq_str = ''.join(['1' if x==1 else '0' for x in test_vector])
        text_box.insert(tk.END, f"Распознанная буква: {result if result else 'Неизвестно'}   Последовательность: {seq_str}\n\n")

        # Отображение обучающих шаблонов
        text_box.insert(tk.END, "Обучающие шаблоны:\n")
        for k, v in patterns.items():
            pattern_str = ''.join(['1' if x==1 else '0' for x in v])
            text_box.insert(tk.END, f"{k}: {pattern_str}\n")

        # Кнопка сброса
        def reset_app():
            for widget in frame_main.winfo_children():
                widget.destroy()
            tk.Label(frame_main, text="Сеть Хэмминга — загрузите обучающую и тестовую выборку", font=("Arial", 14)).pack(pady=20)
            tk.Button(frame_main, text="Загрузить файлы и распознать", command=run_hamming, bg="lightblue", font=("Arial", 12)).pack(pady=10)

        tk.Button(frame_main, text="Сбросить", command=reset_app, bg="lightgray").pack(pady=10)

    except Exception as e:
        messagebox.showerror("Ошибка", f"Ошибка при обработке файлов:\n{e}")

# --- GUI ---
root = tk.Tk()
root.title("Распознавание символов (Сеть Хэмминга)")
root.geometry("700x500")

frame_main = tk.Frame(root)
frame_main.pack(expand=True, fill="both")

label_title = tk.Label(frame_main, text="Сеть Хэмминга — загрузите обучающую и тестовую выборку", font=("Arial", 14))
label_title.pack(pady=20)

button_run = tk.Button(frame_main, text="Загрузить файлы и распознать", command=run_hamming, bg="lightblue", font=("Arial", 12))
button_run.pack(pady=10)

root.mainloop()
