import time

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
                
def write_right_phones(salt):
    with open("phones.txt", "w", encoding="utf-8") as f:
        lst = []
        with open("output.txt", "r", encoding="utf-8") as ff:
            # salt = 100476480
            for el in ff:
                el = int(el.split(":")[-1])
                lst.append(str(el - salt))
        s = "\n".join(lst)
        f.write(s)

if __name__ == "__main__":
    salt = find_salt()
    write_right_phones(salt)
    print (salt)
