import pandas as pd
import xml.etree.ElementTree as ET
import ast

# Загрузите данные из Excel
usecols = ['Магазин', 'Координаты и время', 'Категория', 'Бренд', 'Номер карты', 'Количество товаров', 'Цена']
excel_data_df = pd.read_excel('./laba_1/output.xlsx', usecols=usecols)

# Создаем корневой элемент
dataset = ET.Element('dataset')

# Обрабатываем каждую строку в DataFrame
for index, row in excel_data_df.iterrows():
    store = ET.SubElement(dataset, 'store')

    # Название магазина
    name = ET.SubElement(store, 'name')
    name.text = str(row['Магазин'])

# Координаты и время — создаем элемент coordinates
    coordinates = ET.SubElement(store, 'coordinates')

    # Разбор координат и времени
    try:
        coord_time = ast.literal_eval(str(row['Координаты и время']))  # Преобразуем строку в кортеж
        lat_long = coord_time[0] if isinstance(coord_time[0], str) else None  # Проверяем, что это строка
        time = coord_time[1]  # Время
    except (ValueError, SyntaxError, TypeError):  # Обработка некорректных данных
        lat_long = None
        time = None

    # Вставляем время, если оно есть
    if time is not None:
        datetime = ET.SubElement(coordinates, 'datetime')  # Используем созданный элемент coordinates
        datetime.text = str(time)

    # Вставляем широту и долготу, если они не None
    location = ET.SubElement(coordinates, 'location')

    if lat_long is not None:
        latitude, longitude = lat_long.split(',')  # Разделяем строку на широту и долготу

        # Создаем элементы для широты и долготы
        latitude_elem = ET.SubElement(location, 'latitude')
        latitude_elem.text = latitude.strip()  # Убираем лишние пробелы

        longitude_elem = ET.SubElement(location, 'longitude')
        longitude_elem.text = longitude.strip()  # Убираем лишние пробелы
    else:
        # Если координаты отсутствуют, добавляем сообщение об отсутствии данных
        missing_coords = ET.SubElement(location, 'missing_coordinates')
        missing_coords.text = 'Coordinates not available'

    # Категория
    category = ET.SubElement(store, 'category')
    category.text = str(row['Категория'])

    # Бренд
    brand = ET.SubElement(store, 'brand')
    brand.text = str(row['Бренд'])

    # Номер карты
    card_number = ET.SubElement(store, 'card_number')
    card_number.text = str(row['Номер карты'])

    # Количество товаров
    item_count = ET.SubElement(store, 'item_count')
    item_count.text = str(row['Количество товаров'])

    # Цена
    price = ET.SubElement(store, 'price')
    price.text = str(row['Цена'])

# Создаем дерево XML
tree = ET.ElementTree(dataset)

# Записываем XML-документ в файл с добавлением строки стилей
with open('./laba_1/data.xml', 'w', encoding='utf-8') as f:
    f.write('<?xml version="1.0" encoding="utf-8"?>\n')
    f.write('<?xml-stylesheet type="text/xsl" href="styles.xsl"?>\n')
    tree.write(f, encoding='unicode')
