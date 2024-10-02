import random
import time
import pandas as pd

start_exec_time = time.time()

stores = [line.strip() for line in open('./laba_1/stores.txt', 'r', encoding="utf8")]
categories = [line.strip() for line in open('./laba_1/categories.txt', 'r', encoding="utf8")]
brands = [line.strip() for line in open('./laba_1/brands.txt', 'r', encoding="utf8")]
coordinates_dict = {}
with open('./laba_1/coordinates.txt', 'r', encoding="utf8") as file:
    for line in file:
        shop_name, coords = line.strip().split(':')
        if shop_name in coordinates_dict:
            coordinates_dict[shop_name].append(coords)
        else:
            coordinates_dict[shop_name] = [coords]

def create_card(bank_name, card_variant):
    card_num = str(random.randint(123456789000, 999999999999))
    bank_prefixes = {
        'VTB': {'mastercard': '4274', 'visa': '4272'},
        'Sberbank': {'mastercard': '5469', 'visa': '4276'},
        'Alfa-Bank': {'mastercard': '5106', 'visa': '4279'},
        'T-Bank': {'mastercard': '5189', 'visa': '5213'},
        'Raiffeisen': {'mastercard': '5404', 'visa': '4273'}
    }
    
    if bank_name not in bank_prefixes:
        raise ValueError('Invalid bank')
    
    card_prefix = bank_prefixes[bank_name].get(card_variant)
    full_card_num = card_prefix + card_num
    return full_card_num

def generate_random_data():
    shop_idx = random.randint(0, 29)
    if shop_idx <= 9:
        topic_idx = random.randint(0, 14)
        brand_idx = random.randint(0, 149)
    elif 10 <= shop_idx <= 19:
        topic_idx = random.randint(15, 29)
        brand_idx = random.randint(150, 299)
    else:
        topic_idx = random.randint(30, 49)
        brand_idx = random.randint(300, 499)
    
    return shop_idx, topic_idx, brand_idx

def get_random_coordinates(store_name):
    # Если магазин есть в словаре координат
    if store_name in coordinates_dict:
        return random.choice(coordinates_dict[store_name])
    else:
        return "Координаты не найдены"

def pick_random_time():
    hour = random.randint(10, 21)
    minute = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}"

def random_item_count():
    return random.randint(5,100)

def realistic_pricing(product_type):
    price_ranges = {
        "Смартфон": (10000, 70000),
        "Ноутбук": (30000, 120000),
        "Телевизор": (15000, 80000),
        "Планшет": (7000, 35000),
        "Фотокамера": (5000, 30000),
        "Принтер": (3000, 15000),
        "Кондиционер": (10000, 40000),
        "Холодильник": (15000, 60000),
        "Пылесос": (2000, 10000),
        "Стиральная машина": (10000, 50000),
        "Плеер и аудиотехника": (1000, 5000),
        "Микроволновая печь": (3000, 10000),
        "Видеокамера": (8000, 35000),
        "Монитор": (5000, 25000),
        "Гарнитура": (500, 3000),
        "Крем": (200, 1500),
        "Лосьон": (100, 1000),
        "Пудра": (300, 2000),
        "Тональник": (400, 2500),
        "Маскара": (150, 1000),
        "Тушь": (100, 800),
        "Помада": (200, 1500),
        "Блеск": (150, 1000),
        "Парфюм": (1000, 8000),
        "Гель": (100, 800),
        "Консилер": (300, 2000),
        "Румяна": (200, 1500),
        "Линер": (100, 800),
        "Шампунь": (100, 1000),
        "Основа": (300, 2000),
        "Футболка": (500, 3000),
        "Джинсы": (1000, 5000),
        "Платье": (800, 4000),
        "Куртка": (1500, 8000),
        "Блузка": (600, 3000),
        "Шорты": (400, 2000),
        "Юбка": (600, 3000),
        "Пальто": (1500, 8000),
        "Свитер": (800, 4000),
        "Штаны": (800, 4000),
        "Жилет": (500, 2500),
        "Рубашка": (600, 3000),
        "Шарф": (200, 1500),
        "Плавки": (300, 2000),
        "Ветровка": (1000, 5000),
        "Шапка": (200, 1500),
        "Перчатки": (100, 800),
        "Бикини": (300, 2000),
        "Шляпа": (200, 1500),
        'Костюм': (1500, 8000)
    }

    base_price, max_price = price_ranges.get(product_type, (100, 50))
    price = round(random.gauss(base_price, max_price), 2)
    price = round(price / 100) * 100 + random.choice([99, 5, 0])
    return abs(price)

def select_bank_and_variant(bank_probs, card_probs):
    banks = ['VTB', 'Sberbank', 'Alfa-Bank', 'T-Bank', 'Raiffeisen']
    bank_name = random.choices(banks, weights=bank_probs, k=1)[0]
    
    card_variants = ['mastercard', 'visa']
    card_variant = random.choices(card_variants, weights=card_probs, k=1)[0]
    
    return bank_name, card_variant

# Запрашиваем количество строк для генерации
num_rows = int(input("Введите количество строк для генерации (минимум 50000): "))

# Запрашиваем вероятности для банков
bank_probs = []
print("Введите вероятности для банков (сумма должна быть > 0):")
for bank in ['VTB', 'Sberbank', 'Alfa-Bank', 'T-Bank', 'Raiffeisen']:
    prob = float(input(f"Введите вероятность для {bank}: "))
    bank_probs.append(prob)

# Запрашиваем вероятности для типов карт
card_probs = []
print("Введите вероятности для типов карт (сумма должна быть > 0):")
for card in ['mastercard', 'visa']:
    prob = float(input(f"Введите вероятность для {card}: "))
    card_probs.append(prob)

# Собираем координаты и время
shop_location_data = []
for i in range(len(stores)):
    loc_name = stores[i]
    loc_coordinates = get_random_coordinates(loc_name)
    loc_time = pick_random_time()
    shop_location_data.append((loc_coordinates, loc_time))

df_combined = pd.DataFrame()
for i in range(num_rows):
    store_index, topic_index, brand_index = generate_random_data()
    
    category = categories[topic_index]
    item_count = random_item_count()

    bank, card_variant = select_bank_and_variant(bank_probs, card_probs)  # Передаем вероятности
    card_number = create_card(bank, card_variant)

    # данные о покупке
    purchase_info = {
        'Магазин': [stores[store_index]],
        'Координаты и время': [shop_location_data[store_index]],
        'Категория': [category],
        'Бренд': [brands[brand_index]],
        'Номер карты': [card_number],
        'Количество товаров': [item_count],
        'Цена': [realistic_pricing(category)]
    }

    df_purchase = pd.DataFrame(purchase_info)
    df_combined = pd.concat([df_combined, df_purchase], ignore_index=True)

#print(df_combined)
df_combined.to_excel('./laba_1/output.xlsx', index=False)

end_exec_time = time.time()
print(f"Программа выполнена за {end_exec_time - start_exec_time} секунд")
