import pandas

usecols=['Магазин', 'Координаты  и время', 'Категория', 'Бренд', 'Номер карты', 'Количество товаров', 'Цена']

excel_data_df = pandas.read_excel('./laba_1/output.xlsx', sheet_name='Sheet1')
excel_data_df.to_csv('./laba_1/data.csv')