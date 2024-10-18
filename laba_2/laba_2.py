import pandas as pd
import random
import sys
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QPushButton, QLabel, QComboBox
from PyQt5.QtWidgets import QListWidget

st = [line.strip() for line in open('./laba_1/brands.txt', 'r', encoding="utf8")]
shop_list = [line.strip() for line in open('./laba_1/stores.txt', 'r', encoding="utf8")]
topic_list = [line.strip() for line in open('./laba_1/categories.txt', 'r', encoding="utf8")]

excel_data_df = pd.read_excel('./laba_2/output.xlsx', sheet_name='Sheet1')
columns = ['Магазин', 'Координаты и время', 'Категория', 'Бренд', 'Номер карты', 'Количество товаров', 'Цена']

def delete_rows_with_lowest_uniqueness(df, qi_identifiers, max_rows_to_delete=2500):
    # Сортировка строк по значению count (показателю уникальности)
    df_sorted = df.sort_values(by='count')
    
    # Удаление первых max_rows_to_delete строк
    removed_rows = df_sorted.head(max_rows_to_delete)
    remaining_rows = df_sorted.iloc[max_rows_to_delete:]
    
    return remaining_rows, removed_rows

def mask_shop(data):
    masked_data = ''
    for i in range(len(shop_list)):
        if shop_list[i] == data:
            index = i
            if index <= 9:
                masked_data = 'Техники'
            elif 10 <= index <= 19:
                masked_data = 'Косметики и мед.товаров'
            elif index > 19:
                masked_data = "Одежды"

    return masked_data

def mask_categories(data):
    masked_data = ''
    for i in range(len(topic_list)):
        if topic_list[i] == data:
            index = i
            if index <= 14:
                masked_data = 'Техника'
            elif 15 <= index <= 29:
                masked_data = 'Косметика и медицина'
            elif index >= 30:
                masked_data = "Одежда,обувь и аксессуары"

    return masked_data

def mask_price(data):
    if int(data) < 5000:
        price = 'меньше 5000'
    elif int(data) >= 5000 & int(data)<50000:
        price = 'от 5000 до 50000'
    else:
        price = 'больше 50000'
    return price

def mask_card(data):
    #masked_data_1 = "*" * 12
    masked_data=''
    pref = str(data)[:1]
    if pref =='5':
        masked_data = 'Mastercard'
    elif pref == '4':
        masked_data='Visa'
    #masked_data = pref + masked_data_1
    return masked_data

def mask_num(data):
    if int(data) >= 50:
        masked_data = '(50, 100)'
    else:
        masked_data = '(5, 50)'
    return masked_data

def mask_brand(data):
    masked_data = ''
    for i in range(len(st)):
        if st[i] == data:
            index = i
            if index <= 149:
                masked_data = 'Техника'
            elif 150 <= index <= 299:
                masked_data = 'Косметика и медицина'
            elif index >= 300:
                masked_data = "Одежда,обувь и аксессуары"

    return masked_data

def mask_coordinates(data):
    data = str(data)
    start_index = data.find('.') + 2
    startstart_index = data.find('.', start_index) + 2
    end_index = data.find(',', start_index)
    ednend_index = data.find("'")
    time_end = data.find("'", ednend_index)

    first_cord = data[start_index:end_index]
    second_cord = data[startstart_index:ednend_index]

    perturbed_data1 = ''.join(char if random.random() > 0.2 else random.choice("1234567890") for char in first_cord)
    perturbed_data2 = ''.join(char if random.random() > 0.2 else random.choice("1234567890") for char in second_cord)

    first_cord = data[1:4] + perturbed_data1
    second_cord = data[end_index:startstart_index] + perturbed_data2
    cord = first_cord[:2] + second_cord[:4]

    return cord

def calculate_k_anonymity(df, column):
    group_counts = df.groupby(column).size().reset_index(name='count')
    return min(group_counts['count']), max(group_counts['count']), group_counts


class AnonymizationApp(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.actionLabel = QLabel('Выберите действие:')
        self.actionComboBox = QComboBox()
        self.actionComboBox.addItem('Обезличить датасет')
        self.actionComboBox.addItem('Вычислить k-анонимность')

        columns = ['Магазин', 'Координаты и время', 'Категория', 'Бренд', 'Номер карты', 'Количество товаров',
                   'Цена']
        self.qiLabel = QLabel('Выберите квази-идентификаторы:')
        self.qiListWidget = QListWidget()
        self.qiListWidget.addItems(columns)
        self.qiListWidget.setSelectionMode(QListWidget.MultiSelection)

        self.executeButton = QPushButton('Выполнить')
        self.executeButton.clicked.connect(self.executeAction)

        layout.addWidget(self.actionLabel)
        layout.addWidget(self.actionComboBox)
        layout.addWidget(self.qiLabel)
        layout.addWidget(self.qiListWidget)
        layout.addWidget(self.executeButton)

        self.setLayout(layout)
        self.setWindowTitle('Anonymization App')
        self.show()

    def executeAction(self):
        action = self.actionComboBox.currentText()
        qi_identifiers = [item.text() for item in self.qiListWidget.selectedItems()]
        excel_data_df = pd.read_excel('./laba_2/output.xlsx', sheet_name='Sheet1')

        if action == 'Обезличить датасет':
            if 'Магазин' in qi_identifiers:
                excel_data_df['Магазин'] = excel_data_df['Магазин'].apply(mask_shop)
            if "Цена" in qi_identifiers:
                excel_data_df['Цена'] = excel_data_df['Цена'].apply(mask_price)
            if "Категория" in qi_identifiers:
                excel_data_df['Категория'] = excel_data_df['Категория'].apply(mask_categories)
            if "Номер карты" in qi_identifiers:
                excel_data_df['Номер карты'] = excel_data_df['Номер карты'].apply(mask_card)
            if "Координаты и время" in qi_identifiers:
                excel_data_df['Координаты и время'] = excel_data_df['Координаты и время'].apply(mask_coordinates)
            if "Количество товаров" in qi_identifiers:
                excel_data_df['Количество товаров'] = excel_data_df['Количество товаров'].apply(mask_num)
            if "Бренд" in qi_identifiers:
                excel_data_df['Бренд'] = excel_data_df['Бренд'].apply(mask_brand)
            excel_data_df.to_excel('./laba_2/anonimized.xlsx', index=False)
            print("done")

            pass

        elif action == 'Вычислить k-анонимность':
            excel_data1_df = pd.read_excel('./laba_2/anonimized.xlsx', sheet_name='Sheet1')
            k_anon, k_max, df_1 = calculate_k_anonymity(excel_data1_df, qi_identifiers)
            k = k_anon
            df_1.to_excel('./laba_2/unique_lines.xlsx', index=False)
            print(k_anon, k_max)

            counten = df_1['count'].tolist()
            sort_counten = sorted(counten)
            
            # Удаление строк с самым низким показателем уникальности
            remaining_rows, removed_rows = delete_rows_with_lowest_uniqueness(excel_data1_df, qi_identifiers)
            
            # Сохранение удаленных строк в новый файл
            removed_rows.to_excel('./laba_2/anonimize_removed.xlsx', index=False)
            
            # Сохранение оставшихся строк в исходный файл
            remaining_rows.to_excel('./laba_2/anonimized.xlsx', index=False)

            print(f"\nПлохие значения K-анонимности (первые 5):", sort_counten[:5])

            unique_rows_count = len(remaining_rows.groupby(qi_identifiers))

            print(f"\nКоличество уникальных строк по квази-идентификаторам: {unique_rows_count}")
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AnonymizationApp()
    sys.exit(app.exec_())




