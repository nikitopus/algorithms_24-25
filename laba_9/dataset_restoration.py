import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

class DatasetRestorationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Восстановление датасета")
        self.root.geometry("1200x800")
        
        self.dataFrame = pd.DataFrame()
        self.originalDataFrame = pd.DataFrame()
        
        # Создаем главный Notebook (вкладки)
        self.main_notebook = ttk.Notebook(root)
        self.main_notebook.pack(fill=tk.BOTH, expand=True)
        
        # 1. Вкладка "Датасет" (исходный интерфейс)
        self.dataset_tab = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.dataset_tab, text="Датасет")
        self.init_dataset_ui()
        
        # 2. Вкладка "Числовая статистика"
        self.stats_tab = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.stats_tab, text="Числовая статистика")
        self.init_stats_ui()
        
        # 3. Вкладка "Распределения"
        self.dist_tab = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.dist_tab, text="Распределения")
        self.init_dist_ui()
    
    def init_dataset_ui(self):
        """Инициализация вкладки с датасетом (исходный интерфейс)"""
        main_frame = tk.Frame(self.dataset_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Панель управления (левая часть)
        control_panel = tk.Frame(main_frame, width=240, relief=tk.RIDGE, borderwidth=1)
        control_panel.pack(side=tk.LEFT, fill=tk.Y)
        control_panel.pack_propagate(False)
        
        # Все элементы управления из исходного кода
        ttk.Label(control_panel, text="Удалить (%):").pack(padx=5, anchor=tk.W)
        self.remove_percent_entry = ttk.Entry(control_panel)
        self.remove_percent_entry.pack(fill=tk.X, padx=5)
        
        self.remove_btn = ttk.Button(control_panel, text="Удалить", command=self.removeRandomData)
        self.remove_btn.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_panel, text="Метод:").pack(padx=5, anchor=tk.W)
        self.method_combo = ttk.Combobox(control_panel, values=[
            "Попарное удаление",
            "Метод заполнения моды",
            "Стохастическая линейная регрессия"
        ])
        self.method_combo.pack(fill=tk.X, padx=5)
        self.method_combo.current(0)
        
        self.restore_btn = ttk.Button(control_panel, text="Восстановить", command=self.restoreData)
        self.restore_btn.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=5)
        
        self.load_btn = ttk.Button(control_panel, text="Загрузить CSV", command=self.loadCSV)
        self.load_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Статистика (упрощенная версия)
        stats_frame = tk.LabelFrame(control_panel, text="Быстрая статистика", padx=5, pady=5)
        stats_frame.pack(fill=tk.BOTH, padx=5, pady=10, expand=True)
        
        self.stats_text = ScrolledText(stats_frame, height=10, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.stats_text.insert(tk.END, "Записей: 0\nПропусков: 0%")
        self.stats_text.config(state=tk.DISABLED)

        # Таблица данных (правая часть)
        table_frame = tk.Frame(main_frame)
        table_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.style = ttk.Style()
        self.style.configure("NaN.Treeview", background="red")

        # Treeview для отображения таблицы
        self.tree = ttk.Treeview(table_frame)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Полосы прокрутки
        scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scroll_y.set)
        
        scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.configure(xscrollcommand=scroll_x.set)
    
    def init_stats_ui(self):
        """Инициализация вкладки с числовой статистикой"""
        # Текстовое поле для статистики
        self.num_stats_text = ScrolledText(self.stats_tab, wrap=tk.WORD)
        self.num_stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Кнопка обновления
        update_btn = ttk.Button(self.stats_tab, text="Обновить статистику", 
                              command=self.update_numeric_stats)
        update_btn.pack(side=tk.BOTTOM, pady=5)
    
    def init_dist_ui(self):
        """Инициализация вкладки с распределениями"""
        # График распределения
        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.dist_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Кнопка обновления
        update_btn = ttk.Button(self.dist_tab, text="Обновить графики", 
                              command=self.update_distributions)
        update_btn.pack(side=tk.BOTTOM, pady=5)
    
    def removeRandomData(self):
        if self.dataFrame.empty:
            messagebox.showwarning("Предупреждение", "Нет данных для обработки")
            return

        try:
            removal_percentage = float(self.remove_percent_entry.get())
            if not (0 < removal_percentage <= 100):
                raise ValueError
        except ValueError:
            messagebox.showwarning("Ошибка", "Введите процент от 0 до 100")
            return

        data_frame_copy = self.dataFrame.copy()
        total_cells = data_frame_copy.size
        cells_to_remove = int((removal_percentage / 100) * total_cells)

        # Получаем список непустых ячеек
        non_nan_cells = [
            (row_idx, col_name)
            for row_idx in data_frame_copy.index
            for col_name in data_frame_copy.columns
            if pd.notna(data_frame_copy.at[row_idx, col_name])
        ]

        if len(non_nan_cells) < cells_to_remove:
            messagebox.showwarning("Ошибка", "Недостаточно непустых значений для удаления")
            return

        # Вычисляем вероятности удаления для каждой ячейки
        probabilities = []
        for row_idx, col_name in non_nan_cells:
            row_factor = (row_idx + 1) / len(data_frame_copy)
            col_factor = (data_frame_copy.columns.get_loc(col_name) + 1) / len(data_frame_copy.columns)
            probability = (row_factor + col_factor) / 2
            probabilities.append(probability)

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()  # Нормализуем вероятности

        # Выбираем случайные ячейки для удаления
        cells_to_delete_indices = np.random.choice(
            len(non_nan_cells), 
            cells_to_remove, 
            replace=False, 
            p=probabilities
        )

        # Удаляем выбранные ячейки
        for cell_idx in cells_to_delete_indices:
            row_idx, col_name = non_nan_cells[cell_idx]
            data_frame_copy.at[row_idx, col_name] = np.nan

        self.dataFrame = data_frame_copy
        self.updateTable()
    
    def updateStatistics(self):
        totalCells = self.dataFrame.size
        missingCells = self.dataFrame.isna().sum().sum()
        missingPercentage = (missingCells / totalCells) * 100 if totalCells > 0 else 0
        
        # Включаем редактирование
        self.stats_text.config(state=tk.NORMAL)
        
        # Полностью очищаем поле
        self.stats_text.delete(1.0, tk.END)
        
        # Вставляем обновленный текст
        self.stats_text.insert(tk.END, f"Записей: {len(self.dataFrame)}\nПропусков: {missingPercentage:.2f}%")
        
        # Снова отключаем редактирование
        self.stats_text.config(state=tk.DISABLED)

    def update_all_stats(self):
        """Обновляет всю статистику и графики"""
        self.update_numeric_stats()
        self.update_distributions()

    def update_numeric_stats(self):
        """Вычисляет и отображает числовую статистику"""
        if self.dataFrame.empty:
            return
            
        self.num_stats_text.config(state=tk.NORMAL)
        self.num_stats_text.delete(1.0, tk.END)
        
        cols_to_analyze = [col for col in self.dataFrame.columns if col != "Unnamed: 0"]

        # Для числовых столбцов (исключаем 'Номер карты')
        numeric_cols = [col for col in self.dataFrame[cols_to_analyze].select_dtypes(include=['number']).columns 
                    if col != "Номер карты"]
                
        for col in numeric_cols:
            data = self.dataFrame[col].dropna()
            if len(data) == 0:
                continue
                
            stats = [
                f"Столбец: {col}",
                f"Среднее: {data.mean():.2f}",
                f"Медиана: {data.median():.2f}",
                f"Станд. отклонение: {data.std():.2f}",
                f"Минимум: {data.min():.2f}",
                f"Максимум: {data.max():.2f}",
                f"Количество: {len(data)}",
                "-"*30 + "\n"
            ]
            self.num_stats_text.insert(tk.END, "\n".join(stats))
        
        # Для категориальных столбцов (включая 'Номер карты')
        cat_cols = self.dataFrame[cols_to_analyze].select_dtypes(include=['object']).columns.union(["Номер карты"])

        for col in cat_cols:
            data = self.dataFrame[col].dropna()
            if len(data) == 0:
                continue
                
            mode = data.mode()
            mode_str = mode.iloc[0] if not mode.empty else "Нет данных"
            
            stats = [
                f"Столбец: {col}",
                f"Мода: {mode_str}",
                f"Уникальных значений: {data.nunique()}",
                f"Количество: {len(data)}",
                "-"*30 + "\n"
            ]
            self.num_stats_text.insert(tk.END, "\n".join(stats))
            
        self.num_stats_text.config(state=tk.DISABLED)
    
    def update_distributions(self):
        """Строит графики распределений для числовых столбцов"""
        if self.dataFrame.empty:
            return
            
        self.figure.clear()

        cols_to_analyze = [col for col in self.dataFrame.columns if col != "Unnamed: 0"]
        
        # Для числовых столбцов (исключаем 'Номер карты')
        numeric_cols = [col for col in self.dataFrame[cols_to_analyze].select_dtypes(include=['number']).columns 
                    if col != "Номер карты"]

        if len(numeric_cols) == 0:
            return
            
        # Создаем подграфики
        n_cols = min(2, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        axes = self.figure.subplots(n_rows, n_cols)
        if len(numeric_cols) == 1:
            axes = np.array([axes])
            
        for ax, col in zip(axes.flatten(), numeric_cols):
            data = self.dataFrame[col].dropna()
            if len(data) == 0:
                continue
                
            # Гистограмма
            ax.hist(data, bins='auto', alpha=0.7, edgecolor='black')
            ax.set_title(f'Распределение {col}')
            ax.set_xlabel('Значения')
            ax.set_ylabel('Частота')
            
            # Линии для среднего и медианы
            ax.axvline(data.mean(), color='red', linestyle='--', label='Среднее')
            ax.axvline(data.median(), color='green', linestyle=':', label='Медиана')
            ax.legend()
            
        # Удаляем лишние оси
        for ax in axes.flatten()[len(numeric_cols):]:
            ax.remove()
            
        self.figure.tight_layout()
        self.canvas.draw()

    def restoreData(self):
        if self.dataFrame.empty:
            return
        
        method = self.method_combo.get()
        if method == "Попарное удаление":
            self.pairwiseDeletion()
        elif method == "Метод заполнения моды":
            self.modeImputation()
        elif method == "Стохастическая линейная регрессия":
            self.stochasticRegressionImputation()
        self.updateTable()

        errorText = self.calculateRelativeError()
        if errorText is not None:
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.insert(tk.END, f"\n{errorText}")
            self.stats_text.config(state=tk.DISABLED)

    def pairwiseDeletion(self):
        df = self.dataFrame
        columns_with_missing = df.columns[df.isna().any()].tolist()
        rows_before = len(df)
        df_cleaned = df.dropna(subset=columns_with_missing)
        rows_after = len(df_cleaned)
        
        # Обновляем DataFrame
        self.dataFrame = df_cleaned
        
        # Показываем статистику удаления
        removed_count = rows_before - rows_after

        messagebox.showinfo(
            "Попарное удаление выполнено",
            f"Удалено строк: {removed_count}\n"
            f"Осталось строк: {rows_after}\n"
            f"Удалено {removed_count/rows_before*100:.1f}% данных"
        )

    def modeImputation(self):
        df = self.dataFrame
        for col in df.columns:
            if df[col].dropna().empty:
                continue
            modeVal = df[col].mode().iloc[0]
            df[col] = df[col].fillna(modeVal)
        self.dataFrame = df

    def calculateRelativeError(self):
        if self.originalDataFrame.empty or self.dataFrame.empty:
            return None

        def preprocessDataFrame(dataFrame):
            processedFrame = dataFrame.copy()

            # Кодируем названия магазинов
            codes, _ = pd.factorize(processedFrame['Магазин'])
            processedFrame['storeCode'] = codes.astype(float)

            # Разбираем координаты и время
            coord_time = processedFrame['Координаты и время'].str.extract(r"\('([^']+)', '([^']+)'\)")
            coords = coord_time[0].str.split(',', expand=True)
            processedFrame['latitude'] = pd.to_numeric(coords[0], errors='coerce')
            processedFrame['longitude'] = pd.to_numeric(coords[1], errors='coerce')
            
            time_parts = coord_time[1].str.split(':', expand=True)
            processedFrame['hour'] = pd.to_numeric(time_parts[0], errors='coerce')
            processedFrame['minute'] = pd.to_numeric(time_parts[1], errors='coerce')
            processedFrame['timeCode'] = processedFrame['hour'] * 60 + processedFrame['minute']

            # Кодируем категории и бренды
            codes, _ = pd.factorize(processedFrame['Категория'])
            processedFrame['categoryCode'] = codes.astype(float)
            
            codes, _ = pd.factorize(processedFrame['Бренд'])
            processedFrame['brandCode'] = codes.astype(float)

            # Обрабатываем номер карты
            processedFrame['cardCode'] = pd.to_numeric(processedFrame['Номер карты'], errors='coerce')

            # Количество товаров и цена
            processedFrame['quantity'] = pd.to_numeric(processedFrame['Количество товаров'], errors='coerce')
            processedFrame['price'] = pd.to_numeric(processedFrame['Цена'], errors='coerce')

            return processedFrame

        originalProcessed = preprocessDataFrame(self.originalDataFrame)
        currentProcessed = preprocessDataFrame(self.dataFrame)

        numericColumns = [
            'storeCode', 'latitude', 'longitude', 'timeCode',
            'categoryCode', 'brandCode', 'cardCode', 'quantity', 'price'
        ]

        readableNames = {
            'storeCode': 'Магазин',
            'latitude': 'Координаты и время',
            'longitude': 'Координаты и время',
            'timeCode': 'Координаты и время',
            'categoryCode': 'Категория',
            'brandCode': 'Бренд',
            'cardCode': 'Номер карты',
            'quantity': 'Количество товаров',
            'price': 'Цена'
        }

        originalSubset = originalProcessed[numericColumns]
        currentSubset = currentProcessed[numericColumns]

        originalAligned, currentAligned = originalSubset.align(currentSubset, join='inner', axis=0)
        originalAligned, currentAligned = originalAligned.align(currentAligned, join='inner', axis=1)

        nonZeroMask = originalAligned != 0
        relativeErrors = ((originalAligned - currentAligned).abs() / originalAligned)[nonZeroMask]

        columnErrors = relativeErrors.mean() * 100
        groupedErrors = {}

        for col, err in columnErrors.items():
            readable = readableNames.get(col, col)
            groupedErrors.setdefault(readable, []).append(err)

        result = "\n".join([f"{col}: {sum(errs)/len(errs):.2f}%" for col, errs in groupedErrors.items()])
        result += f"\nСуммарная ошибка: {columnErrors.sum():.2f}%"

        return result

    def stochasticRegressionImputation(self):
        dataFrame = self.dataFrame.copy()
        
        # Сохраняем индексы для обратного преобразования
        self.storeIndex = None
        self.categoryIndex = None
        self.brandIndex = None
        
        # 1. Преобразование категориальных переменных
        # Магазины
        codes, self.storeIndex = pd.factorize(dataFrame['Магазин'])
        dataFrame['storeCode'] = codes.astype(float)
        dataFrame.loc[dataFrame['Магазин'].isna(), 'storeCode'] = np.nan
        
        # Координаты и время - более надежная обработка
        try:
            coord_time = dataFrame['Координаты и время'].str.extract(r"\('([^']+)', '([^']+)'\)")
            coords = coord_time[0].str.split(',', expand=True)
            dataFrame['latitude'] = pd.to_numeric(coords[0], errors='coerce').fillna(0)
            dataFrame['longitude'] = pd.to_numeric(coords[1], errors='coerce').fillna(0)
            
            time_parts = coord_time[1].str.split(':', expand=True)
            dataFrame['hour'] = pd.to_numeric(time_parts[0], errors='coerce').fillna(0)
            dataFrame['minute'] = pd.to_numeric(time_parts[1], errors='coerce').fillna(0)
            dataFrame['timeCode'] = (dataFrame['hour'] * 60 + dataFrame['minute']).fillna(0)
        except:
            dataFrame['latitude'] = 0.0
            dataFrame['longitude'] = 0.0
            dataFrame['timeCode'] = 0.0
        
        # Категории и бренды
        codes, self.categoryIndex = pd.factorize(dataFrame['Категория'])
        dataFrame['categoryCode'] = codes.astype(float)
        dataFrame.loc[dataFrame['Категория'].isna(), 'categoryCode'] = np.nan
        
        codes, self.brandIndex = pd.factorize(dataFrame['Бренд'])
        dataFrame['brandCode'] = codes.astype(float)
        dataFrame.loc[dataFrame['Бренд'].isna(), 'brandCode'] = np.nan
        
        # 2. Числовые переменные с обработкой ошибок
        dataFrame['cardCode'] = pd.to_numeric(dataFrame['Номер карты'], errors='coerce').fillna(0)
        dataFrame['quantity'] = pd.to_numeric(dataFrame['Количество товаров'], errors='coerce').fillna(0)
        dataFrame['price'] = pd.to_numeric(dataFrame['Цена'], errors='coerce').fillna(0)
        
        numericCols = [
            'storeCode', 'latitude', 'longitude', 'timeCode',
            'categoryCode', 'brandCode', 'cardCode', 'quantity', 'price'
        ]
        
        # 3. Стохастическая регрессия с проверкой данных
        for col in numericCols:
            if dataFrame[col].isna().any():
                known = dataFrame[dataFrame[col].notna()]
                if len(known) < 2:
                    # Заменяем inplace заполнение на прямое присваивание
                    median_val = dataFrame[col].median()
                    dataFrame.loc[dataFrame[col].isna(), col] = median_val
                    continue
                    
                predictors = [c for c in numericCols if c != col and not dataFrame[c].isna().all()]
                if not predictors:
                    median_val = dataFrame[col].median()
                    dataFrame.loc[dataFrame[col].isna(), col] = median_val
                    continue
                    
                try:
                    model = LinearRegression()
                    model.fit(known[predictors], known[col])
                    
                    unknown = dataFrame[dataFrame[col].isna()]
                    if not unknown.empty:
                        pred = model.predict(unknown[predictors])
                        residuals = known[col] - model.predict(known[predictors])
                        noise = np.random.choice(residuals, size=len(unknown), replace=True)
                        dataFrame.loc[dataFrame[col].isna(), col] = pred + noise
                except:
                    median_val = dataFrame[col].median()
                    dataFrame.loc[dataFrame[col].isna(), col] = median_val
        
        # 4. Обратное преобразование с защитой от NaN
        def safe_map(values, index):
            return values.round().fillna(-1).astype(int).map(
                lambda i: index[i] if 0 <= i < len(index) else ''
            )
        
        dataFrame['Магазин'] = safe_map(dataFrame['storeCode'], self.storeIndex)
        
        # Координаты и время с защитой
        lat = dataFrame['latitude'].round(6).fillna(0)
        lon = dataFrame['longitude'].round(6).fillna(0)
        hours = (dataFrame['timeCode'] // 60).fillna(0).astype(int)
        minutes = (dataFrame['timeCode'] % 60).fillna(0).astype(int).map(lambda x: f"{x:02d}")
        dataFrame['Координаты и время'] = "('" + lat.astype(str) + "," + lon.astype(str) + "', '" + hours.astype(str) + ":" + minutes + "')"
        
        dataFrame['Категория'] = safe_map(dataFrame['categoryCode'], self.categoryIndex)
        dataFrame['Бренд'] = safe_map(dataFrame['brandCode'], self.brandIndex)
        
        dataFrame['Номер карты'] = dataFrame['cardCode'].round().fillna(0).astype(np.int64).astype(str)
        dataFrame['Количество товаров'] = dataFrame['quantity'].round().fillna(0).astype(int)
        dataFrame['Цена'] = dataFrame['price'].round().fillna(0).astype(int)
        
        self.dataFrame = dataFrame[[
            'Магазин', 'Координаты и время', 'Категория', 
            'Бренд', 'Номер карты', 'Количество товаров', 'Цена'
        ]]

    def updateTable(self):
        if self.dataFrame.empty:
            return
        
        # Очищаем текущее содержимое таблицы
        self.tree.delete(*self.tree.get_children())
        
        # Игнорируем первый столбец (индекс) и берем только остальные колонки
        columns = list(self.dataFrame.columns[1:]) if len(self.dataFrame.columns) > 1 else []
        
        # Настраиваем колонки (используем первый столбец для нумерации)
        self.tree["columns"] = columns
        self.tree.heading("#0", text="#")  # Используем встроенный столбец для нумерации
        self.tree.column("#0", width=50, stretch=tk.NO)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.W)
        
        # Добавляем данные с автоматической нумерацией строк
        for i, (_, row) in enumerate(self.dataFrame.iterrows(), start=1):
            values = list(row[1:]) if len(self.dataFrame.columns) > 1 else []
            item = self.tree.insert("", tk.END, text=str(i), values=values)
            
            # Проверяем каждое значение на np.nan и устанавливаем цвет фона
            for j, val in enumerate(values, start=1):
                if isinstance(val, float) and np.isnan(val):
                    self.tree.set(item, column=f"#{j}", value="NaN")  # Можно заменить на str(val)
                    self.tree.tag_configure('nan', background='red')
                    self.tree.item(item, tags=('nan',))

        # Настраиваем растягивание колонок
        for i, col in enumerate(columns):
            if i in [1]:  # Индексы колонок, которые должны растягиваться
                self.tree.column(col, stretch=tk.YES)
            else:
                self.tree.column(col, stretch=tk.NO)
        
        self.updateStatistics()
    
    def loadCSV(self):
        # path, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV файл", "", "CSV Files (*.csv)")
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return  # Если файл не выбран, выходим

        if path:
            self.dataFrame = pd.read_csv(path)
            self.originalDataFrame= self.dataFrame
            self.updateTable()
            self.update_all_stats()

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetRestorationApp(root)
    root.mainloop()
