import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# --- функция обучения модели ---
def train_model(X, y):
    model = Sequential([
        Dense(16, input_dim=8, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=200, verbose=0)
    return model

# --- функция преобразования числа в двоичный вектор ---
def to_binary_array(n):
    return np.array([int(b) for b in format(int(n), '08b')], dtype=float)

# --- основное окно ---
root = tk.Tk()
root.title("Определение чётности числа")
root.geometry("500x400")

frame_main = tk.Frame(root)
frame_main.pack(expand=True, fill="both")

label_title = tk.Label(frame_main, text="Загрузка Excel с числами", font=("Arial", 14))
label_title.pack(pady=20)

def load_excel():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
    if not file_path:
        return

    try:
        df = pd.read_excel(file_path)
        if not {"число", "признак четности"}.issubset(df.columns):
            messagebox.showerror("Ошибка", "Файл должен содержать столбцы: 'число' и 'признак четности'")
            return

        # Очистка данных
        df = df.dropna(subset=["число", "признак четности"])  # убираем пустые строки
        numbers = df["число"].astype(int).values.tolist()
        labels = df["признак четности"].astype(float).values

        # Преобразуем числа в 4-битные двоичные вектора
        X = np.array([to_binary_array(n) for n in numbers], dtype=float)

        # # Проверка формы
        # if X.ndim != 2 or X.shape[1] != 4:
        #     messagebox.showerror("Ошибка", "Ошибка формирования бинарных данных. Проверь содержимое столбца 'число'.")
        #     return

        # Обучение модели
        model = train_model(X, labels)

        # Предсказания
        # Предсказания (на тех же данных из Excel)
        predictions = model.predict(X)

        all_numbers = np.arange(1, 32)
        X_all = np.array([to_binary_array(n) for n in all_numbers], dtype=float)
        pred_all = model.predict(X_all) 

        # Очистка интерфейса
        for widget in frame_main.winfo_children():
            widget.destroy()

        # Вывод результатов
        tk.Label(frame_main, text="Результаты предсказания:", font=("Arial", 14)).pack(pady=10)
        text_box = tk.Text(frame_main, width=40, height=15, font=("Consolas", 10))
        text_box.pack(pady=10)

        for n, pred in zip(all_numbers, pred_all):
            parity = "нечётное" if pred >= 0.5 else "чётное"
            text_box.insert(tk.END, f"{n:2d} ({format(n, '08b')}): {parity} ({pred[0]:.2f})\n")


        # Кнопка "Сбросить"
        def reset_app():
            for widget in frame_main.winfo_children():
                widget.destroy()
            label_title.pack(pady=20)
            button_load.pack(pady=10)

        tk.Button(frame_main, text="Сбросить", command=reset_app, bg="lightgray").pack(pady=10)

    except Exception as e:
        messagebox.showerror("Ошибка", f"Не удалось обработать файл:\n{e}")


button_load = tk.Button(frame_main, text="Загрузить Excel", command=load_excel, bg="lightblue", font=("Arial", 12))
button_load.pack(pady=10)

root.mainloop()
