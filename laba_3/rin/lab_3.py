import subprocess
import hashlib
import time

def dehashing(output_file, hash_file, hash_type=0):
    command = [
        "hashcat",
        "-m",
        str(hash_type),
        "-a",
        "3",
        "-o",
        output_file,
        hash_file,
        "8?d?d?d?d?d?d?d?d?d?d",
        "--potfile-disable",
    ]
    try:
        start_time = time.time()
        subprocess.run(command, check=True)
        end_time = time.time()

        return round(end_time - start_time, 2)
    
    except subprocess.CalledProcessError as e:
        print(f"Ошибка: {e}")

def find_salt():
    with open("output.txt", "r", encoding="utf-8") as f:
        d = dict()
        phones = [89156311602, 89678395615, 89859771985, 89109807351, 89108471943]

        for el in f:
            el = el.split(":")[-1]
            for i in range(5):
                if (int(el) - phones[i]) not in d.keys():
                    d[(int(el) - phones[i])] = 1
                else:
                    d[(int(el) - phones[i])] += 1
        for key, value in d.items():
            if value >= 2:
                salt = key
                return salt

def hash_sha512(source_file, salt):
    with open("new_hashes_sha512.txt", "w", encoding="utf-8") as md5:
        source_file = open(source_file, "r", encoding="utf-8")
        for el in source_file:
            el = int(el.split(":")[-1])
            string = str(el + salt)
            md5.write(str(hashlib.sha512(string.encode()).hexdigest()) + "\n")
        print("done")
        source_file.close()

def hash_sha1(source_file, salt: int):
    with open("new_hashes_sha1.txt", "w", encoding="utf-8") as sha1:
        source_file = open(source_file, "r", encoding="utf-8")
        for el in source_file:
            el = int(el.split(":")[-1])
            string = str(el + salt)
            sha1.write(str(hashlib.sha1(string.encode()).hexdigest()) + "\n")
        print("done")
        source_file.close()

def hash_sha256(source_file, salt: int):
    with open("new_hashes_sha256.txt", "w", encoding="utf-8") as sha256:
        source_file = open(source_file, "r", encoding="utf-8")
        for el in source_file:
            el = int(el.split(":")[-1])
            string = str(el + salt)
            sha256.write(str(hashlib.sha256(string.encode()).hexdigest()) + "\n")
        print("done")
        source_file.close()

def write_right_phones(salt):
    with open("phones.txt", "w", encoding="utf-8") as f:
        lst = []
        with open("output.txt", "r", encoding="utf-8") as ff:
            for el in ff:
                el = int(el.split(":")[-1])
                lst.append(str(el - salt))
        s = "\n".join(lst)
        f.write(s)

if __name__ == "__main__":

    # hash_file = "hashes.txt"

    # time_dehashing_starting_md5 = dehashing("output.txt", hash_file)

    # print(time_dehashing_starting_md5)
    # salt = find_salt()
    # write_right_phones(salt)
    # print (salt)

    salt = 100476480
    source_file = "phones.txt"

    hash_sha512(source_file, salt)
    hash_sha1(source_file, salt)
    hash_sha256(source_file, salt)

    time_sha1 = dehashing("output_sha1.txt", "new_hashes_sha1.txt", 100)
    print("done_sha1")
    time_sha256 = dehashing("output_sha256.txt", "new_hashes_sha256.txt", 1400)
    print("done_sha256")
    time_sha512 = dehashing("output_sha512.txt", "new_hashes_sha512.txt", 1700)
    print("done_sha512")
    
    print(
    f"время для sha512: {time_sha512}\nвремя для sha1: {time_sha1}\nвремя для sha256:{time_sha256}"
    )
