import requests
import time

# Функция для получения координат через Nominatim API (максимум 5 результатов)
def get_coordinates(query, city, max_results=5):
    url = f'https://nominatim.openstreetmap.org/search'
    params = {'q': f'{query}, {city}', 'format': 'json', 'limit': max_results}
    
    try:
        response = requests.get(url, params=params)
        
        # Проверяем статус ответа
        if response.status_code != 200:
            print(f"Ошибка запроса: {response.status_code}. Сообщение: {response.text}")
            return None
        
        # Попробуем получить JSON
        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"Ошибка декодирования JSON для {query} в {city}. Ответ: {response.text}")
            return None
        
        # Если данных нет
        if not data:
            print(f"Нет данных для {query} в {city}.")
            return None

        # Собираем координаты
        coordinates = []
        for item in data[:max_results]:
            latitude = round(float(item['lat']),6)
            longitude = round(float(item['lon']),6)
            coordinates.append((latitude, longitude))
        
        return coordinates

    except requests.exceptions.RequestException as e:
        print(f"Ошибка запроса: {e}")
        return None

# Функция для чтения магазинов из файла
def load_shops(file_path):
    with open(file_path, 'r', encoding="utf8") as f:
        shops = [line.strip() for line in f]
    return shops

# Функция для сохранения координат в файл
def save_coordinates(file_path, shop, coordinates_list):
    with open(file_path, 'a', encoding="utf8") as f:
        for coordinates in coordinates_list:
            f.write(f"{shop}:{coordinates[0]},{coordinates[1]}\n")

# Функция для загрузки уже сохраненных координат
def load_saved_coordinates(file_path):
    saved_coordinates = {}
    try:
        with open(file_path, 'r', encoding="utf8") as f:
            for line in f:
                shop, coords = line.strip().split(':')
                lat, lon = coords.split(',')
                if shop not in saved_coordinates:
                    saved_coordinates[shop] = []
                saved_coordinates[shop].append((float(lat), float(lon)))
    except FileNotFoundError:
        pass  # Если файл не найден, просто продолжаем
    return saved_coordinates

# Основная функция для парсинга и сохранения координат
def main():
    shop_file = './laba_1/stores.txt'
    coordinates_file = './laba_1/coordinates.txt'
    city = 'Санкт-Петербург'

    shops = load_shops(shop_file)
    saved_coordinates = load_saved_coordinates(coordinates_file)

    for shop in shops:
        if shop in saved_coordinates and len(saved_coordinates[shop]) >= 5:
            print(f"Для {shop} уже сохранены 5 координат.")
        else:
            coordinates = get_coordinates(shop, city, max_results=5)
            if coordinates:
                print(f"Получены координаты для {shop}: {coordinates}")
                save_coordinates(coordinates_file, shop, coordinates)
            else:
                print(f"Не удалось получить координаты для {shop}")
            # Добавляем паузу, чтобы не нарушить лимит запросов (макс 1 запрос в секунду)
            time.sleep(1)

if __name__ == "__main__":
    main()
