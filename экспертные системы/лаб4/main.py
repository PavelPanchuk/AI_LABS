import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np

# --- Класс Hopfield (ваш оригинальный код) ---
class HopfieldNetwork:
    def __init__(self, patterns):
        self.patterns = patterns
        self.pattern_size = len(list(patterns.values())[0])
        self.num_patterns = len(patterns)
        self.weights = np.zeros((self.pattern_size, self.pattern_size))
        for pattern in patterns.values():
            x = np.array(pattern).reshape(-1, 1)
            self.weights += np.dot(x, x.T)
        self.weights /= self.pattern_size
        np.fill_diagonal(self.weights, 0)

    def predict(self, input_vector, max_iter=100):
        y = np.array(input_vector.copy())
        prev_y = np.zeros_like(y)
        for _ in range(max_iter):
            prev_y[:] = y[:]
            order = np.random.permutation(self.pattern_size)
            for i in order:
                activation = np.dot(self.weights[i, :], y)
                y[i] = 1 if activation >= 0 else -1
            if np.array_equal(y, prev_y):
                break

        # Поиск ближайшего шаблона
        min_dist = np.inf
        best_name = None
        for name, pattern in self.patterns.items():
            dist = np.sum(y != pattern)
            if dist < min_dist:
                min_dist = dist
                best_name = name
        return best_name, y

# --- GUI функция ---
def run_hopfield():
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
            # Берем 5 строк подряд и 3 столбца для буквы
            rows = df_train.iloc[:, idx*3:(idx+1)*3].values
            vector = rows.flatten()
            vector = np.where(vector == 0, -1, +1)
            patterns[label] = vector

        net = HopfieldNetwork(patterns)

        # --- Загружаем тестовый символ ---
        df_test = pd.read_excel(test_path, header=None)
        test_vector = df_test.values.flatten()
        test_vector = np.where(test_vector == 0, -1, +1)

        # --- Распознаем ---
        result, stabilized_vector = net.predict(test_vector)

        # --- Очистка окна ---
        for widget in frame_main.winfo_children():
            widget.destroy()

        # --- Вывод результатов ---
        tk.Label(frame_main, text="Результат распознавания:", font=("Arial", 14)).pack(pady=10)
        text_box = tk.Text(frame_main, width=70, height=20, font=("Consolas", 12))
        text_box.pack(pady=10)

        seq_str = ''.join(['1' if x==1 else '0' for x in test_vector])
        text_box.insert(tk.END, f"Распознанная буква: {result}   Последовательность: {seq_str}\n\n")

        # Расстояния до всех шаблонов
        text_box.insert(tk.END, "Расстояния Хэмминга до обучающих шаблонов:\n")
        for k, v in patterns.items():
            dist = np.sum(stabilized_vector != v)
            text_box.insert(tk.END, f"{k}: {dist} различий\n")

        # Обучающие шаблоны
        text_box.insert(tk.END, "\nОбучающие шаблоны:\n")
        for k, v in patterns.items():
            pattern_str = ''.join(['1' if x==1 else '0' for x in v])
            text_box.insert(tk.END, f"{k}: {pattern_str}\n")

        # Кнопка сброса
        # Кнопка сброса
        def reset_app():
            for widget in frame_main.winfo_children():
                widget.destroy()
            tk.Label(frame_main, text="Сеть Хопфилда — загрузите обучающую и тестовую выборку", font=("Arial", 14)).pack(pady=20)
            tk.Button(frame_main, text="Загрузить файлы и распознать", command=run_hopfield, bg="lightblue", font=("Arial", 12)).pack(pady=10)

        tk.Button(frame_main, text="Сбросить", command=reset_app, bg="lightgray").pack(pady=10)

    except Exception as e:
        messagebox.showerror("Ошибка при обработке файлов", f"{e}")

# --- GUI ---
root = tk.Tk()
root.title("Распознавание символов (Сеть Хопфилда)")
root.geometry("700x500")

frame_main = tk.Frame(root)
frame_main.pack(expand=True, fill="both")

label_title = tk.Label(frame_main, text="Сеть Хопфилда — загрузите обучающую и тестовую выборку", font=("Arial", 14))
label_title.pack(pady=20)

button_run = tk.Button(frame_main, text="Загрузить файлы и распознать", command=run_hopfield, bg="lightblue", font=("Arial", 12))
button_run.pack(pady=10)

root.mainloop()
