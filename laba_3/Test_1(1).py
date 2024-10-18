import hashlib
import subprocess
import concurrent.futures
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import codecs


def read_hashed_numbers(file_path):
    hashed_numbers = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Удаляем лишние пробелы и символы новой строки
                hashed_number = line.strip()
                if hashed_number:
                    hashed_numbers.append(hashed_number)
    except FileNotFoundError:
        print(f"Файл {file_path} не найден.")
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {e}")

    return hashed_numbers

#file_path = 'hashed_numbers.txt'
#numbers = read_hashed_numbers(file_path)
#phones = [89689561079, 89857375777, 89869850697, 89991192494, 89687419987]

file_path = None
phones = None
numbers = None
is_file_loaded = False


def compute_salt(phones, numbers):
    for phone in phones:
        salt = int(phone) - int(numbers[0])
        if salt < 0:
            continue
        i = 1
        while (str(int(numbers[i]) + salt)) in phones:
            i += 1
            if i == 5:
                return salt
    return 0


def sha1(phones):
    phones_sha1 = [hashlib.sha1(phone.encode()).hexdigest() for phone in phones]
    with open('sha1.txt', 'w') as f:
        for phone in phones_sha1:
            f.write(phone + '\n')

    os.system("hashcat -a 3 -m 100 -o output_sha1.txt sha1.txt ?d?d?d?d?d?d?d?d?d?d?d")


def sha256(phones):
    phones_sha256 = [hashlib.sha256(phone.encode()).hexdigest() for phone in phones]
    with open('sha256.txt', 'w') as f:
        for phone in phones_sha256:
            f.write(phone + '\n')

    os.system("hashcat -a 3 -m 1400 -o output_sha256.txt sha256.txt ?d?d?d?d?d?d?d?d?d?d?d")


def md5(phones):
    phones_md5 = [hashlib.md5(phone.encode()).hexdigest() for phone in phones]
    with open('md5.txt', 'w') as f:
        for phone in phones_md5:
            f.write(phone + '\n')

    os.system("hashcat -a 0 -m 0 -o output_md5.txt md5.txt ?d?d?d?d?d?d?d?d?d?d?d")


def load_file():
    global file_path, is_file_loaded
    file_path = filedialog.askopenfilename()
    if file_path:
        is_file_loaded = True
        button_deidentify["state"] = tk.NORMAL


def identify():
    global file_path, phones, numbers
    df = pd.read_excel(file_path)
    hashes = df["Номер телефона"]
    numbers = [number[:-2] for number in df["Unnamed: 2"].astype(str).tolist()][:5]
    with open('hashes.txt', 'w') as f:
        for HASH in hashes:
            f.write(HASH + "\n")
    os.system("hashcat -a 3 -m 0 -o output.txt hashes.txt ?d?d?d?d?d?d?d?d?d?d?d")

    with open(r'output.txt') as r:
        phones = [line.strip()[33:] for line in r]

    with open('phones.txt', 'w') as file:
        for phone in phones:
            file.write(phone + '\n')

    # Удаление первых 8 строк из phones.txt
    with open('phones.txt', 'r') as file:
        lines = file.readlines()

    # Пропуск первых 8 строк
    lines = lines[8:]

    # Запись оставшихся строк обратно в файл
    with open('phones.txt', 'w') as file:
        for line in lines:
            file.write(line)

    messagebox.showinfo("Готово", "Таблица успешно расшифрована. Данные сохранены в файле 'phones.txt'.")

def find_salt():
    global phones, numbers
    salt = compute_salt(phones, numbers)
    messagebox.showinfo("Готово", f"Значение соли: {salt}")


def encrypt(algorithm):
    global is_file_loaded, phones
    if not is_file_loaded:
        return
    if algorithm == "sha1":
        sha1(phones)
        messagebox.showinfo("Готово", "Результат сохранен в файле output_sha1.")
    elif algorithm == "sha256":
        sha256(phones)
        messagebox.showinfo("Готово", "Результат сохранен в файле output_sha256.")
    else:
        md5(phones)
        messagebox.showinfo("Готово", "Результат сохранен в файле output_md5.")


root = tk.Tk()
root.title("Добро пожаловать в программу!")

frame_actions = tk.Frame(root)
frame_encryption = tk.Frame(root)

# Размещение фреймов на главном окне
frame_actions.grid(row=0, column=0, padx=10, pady=10, sticky="w")
frame_encryption.grid(row=0, column=1, padx=10, pady=10, sticky="e")

# Элементы для выбора действия с таблицей
label_action = tk.Label(frame_actions, text="Выберите действие с таблицей:")
button_load = tk.Button(frame_actions, text="Загрузить", command=load_file)
button_deidentify = tk.Button(frame_actions, text="Деобезличить", command=identify, state=tk.DISABLED)
button_compute_salt = tk.Button(frame_actions, text="Вычислить соль", command=find_salt)

# Размещение элементов в фрейме действий
label_action.grid(row=0, column=0, padx=10, pady=10, sticky="w")
button_load.grid(row=1, column=0, padx=10, pady=5, sticky="w")
button_deidentify.grid(row=2, column=0, padx=10, pady=5, sticky="w")
button_compute_salt.grid(row=3, column=0, padx=10, pady=5, sticky="w")

# Элементы для выбора алгоритма шифрования
label_encryption = tk.Label(frame_encryption, text="Выберите алгоритм шифрования:")
button_encrypt_md5 = tk.Button(frame_encryption, text="Зашифровать MD5", command=lambda: encrypt("md5"))
button_encrypt_sha1 = tk.Button(frame_encryption, text="Зашифровать SHA-1", command=lambda: encrypt("sha1"))
button_encrypt_sha256 = tk.Button(frame_encryption, text="Зашифровать SHA-256", command=lambda: encrypt("sha256"))
label_encryption.grid(row=0, column=0, padx=10, pady=10, sticky="w")
button_encrypt_sha1.grid(row=6, column=0, padx=10, pady=5, sticky="w")
button_encrypt_sha256.grid(row=5, column=0, padx=10, pady=5, sticky="w")
button_encrypt_md5.grid(row=4, column=0, padx=10, pady=5, sticky="w")

root.mainloop()
