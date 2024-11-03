import os
import subprocess

import pandas as pd
import hashlib
import pyopencl as cl



# Функция для вычисления соли
# def compute_salt(phones, numbers):
#     for phone in phones:
#         salt = int(phone) - int(numbers[0])
#         if salt < 0:
#             continue
#         i = 1
#         while (str(int(numbers[i]) + salt)) in phones:
#             i += 1
#             if i == 5:
#                 return salt
#     return


def sha1(phones):
    phones_sha1 = [hashlib.sha1(phone.encode()).hexdigest() for phone in phones]
    with open('sha1.txt', 'w') as f:
        for phone in phones_sha1:
            f.write(phone + '\n')

    # os.remove('C:/IT/lab/lab_3/hashcat-6.2.6/hashcat.potfile')
    os.system("hashcat -a 3 -m 100 -o output_sha1.txt sha1.txt 8?d?d?d?d?d?d?d?d?d?d --potfile-disable")

def sha256(phones):
    phones_sha256 = [hashlib.sha256(phone.encode()).hexdigest() for phone in phones]
    with open('sha256.txt', 'w') as f:
        for phone in phones_sha256:
            f.write(phone + '\n')

    # os.remove('C:/IT/lab/lab_3/hashcat-6.2.6/hashcat.potfile')
    os.system("hashcat -a 3 -m 1400 -o output_sha256.txt sha256.txt 8?d?d?d?d?d?d?d?d?d?d --potfile-disable")


def sha512(phones):
    phones_sha512 = [hashlib.sha512(phone.encode()).hexdigest() for phone in phones]
    with open('sha512.txt', 'w') as f:
        for phone in phones_sha512:
            f.write(phone + '\n')

    # os.remove('C:/IT/lab/lab_3/hashcat-6.2.6/hashcat.potfile')
    os.system("hashcat -a 3 -m 1700 -o output_sha512.txt sha512.txt 8?d?d?d?d?d?d?d?d?d?d --potfile-disable")


# Функция для деобезличивания данных
def identify():
    df = pd.read_excel('scoring_data_v.1.4.xlsx')
    hashes = df["Номер телефона"].tolist()  # Извлекаем номера телефонов
    numbers = [number[:-2] for number in df["Unnamed: 2"].astype(str).tolist()][:5]
    with open('hashes.txt', 'w') as f:
        for HASH in hashes:
            f.write(HASH + "\n")
    # os.system("C:/IT/lab/lab_3/hashcat-6.2.6/hashcat.exe -a 3 -m 0 -o output.txt hashes.txt ?d?d?d?d?d?d?d?d?d?d?d")

    try:
        subprocess.run(
            ["hashcat", "-a", "3", "-m", "0", "-o", "output.txt", "hashes.txt", "8?d?d?d?d?d?d?d?d?d?d", "--potfile-disable"],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Ошибка в hashcat: {e}")
        return
    with open('output.txt') as r:
        phones = [line.strip()[33:] for line in r.readlines()]

def main():
    # identify()

    with open ("phones.txt", 'r') as f:
        phones = f.readlines()

    # salt = 100476480

    encrypt_algorithm = input("Введите алгоритм (sha1/sha256/sha512): ")
    if encrypt_algorithm == "sha1":
        sha1(phones)
    elif encrypt_algorithm == "sha256":
        sha256(phones)
    else:
        sha512(phones)

if __name__ == "__main__":
        main()