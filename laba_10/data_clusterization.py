import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class DatasetClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Кластеризация датасета")
        self.root.geometry("2000x1000")
        
        self.dataFrame = pd.DataFrame()
        self.originalDataFrame = pd.DataFrame()
        self.clusteredDataFrame = pd.DataFrame()
        self.selectedFeaturesDataFrame = pd.DataFrame()
        self.anonymizedDataFrame = pd.DataFrame()
        
        self.initUI()
        self.setupPlots()
    
    def initUI(self):
        # Главный контейнер
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Панель управления (левая часть)
        control_panel = tk.Frame(main_frame, width=240, relief=tk.RIDGE, borderwidth=1)
        control_panel.pack(side=tk.LEFT, fill=tk.Y)
        control_panel.pack_propagate(False)
        
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=5)
        
        self.cluster_btn = ttk.Button(control_panel, text="1. Кластеризовать (полный датасет)", command=self.clusterFullData)
        self.cluster_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.select_features_btn = ttk.Button(control_panel, text="2. Выбрать информативные признаки", command=self.selectInformativeFeatures)
        self.select_features_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.cluster_selected_btn = ttk.Button(control_panel, text="3. Кластеризовать (отобранные признаки)", command=self.clusterSelectedFeatures)
        self.cluster_selected_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.anonymize_btn = ttk.Button(control_panel, text="4. Обезличить датасет", command=self.anonymizeData)
        self.anonymize_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.cluster_anonymized_btn = ttk.Button(control_panel, text="5. Кластеризовать (обезличенные данные)", command=self.clusterAnonymizedData)
        self.cluster_anonymized_btn.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=5)
        
        self.load_btn = ttk.Button(control_panel, text="Загрузить CSV", command=self.loadCSV)
        self.load_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Статистика
        stats_frame = tk.LabelFrame(control_panel, text="Результаты кластеризации", padx=5, pady=5)
        stats_frame.pack(fill=tk.BOTH, padx=5, pady=10, expand=True)
        
        self.stats_text = ScrolledText(stats_frame, height=10, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.stats_text.insert(tk.END, "Загрузите датасет и выполните кластеризацию")
        self.stats_text.config(state=tk.DISABLED)

        # Таблица данных (правая часть)
        table_frame = tk.Frame(main_frame)
        table_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Графики (нижняя часть)
        self.plot_frame = tk.Frame(table_frame)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        # Создаем Treeview для отображения таблицы
        self.tree = ttk.Treeview(table_frame)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Добавляем полосу прокрутки
        scroll_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scroll_y.set)
        
        scroll_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.configure(xscrollcommand=scroll_x.set)
    
    def setupPlots(self):
        # Создаем область для графиков
        self.figure, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self.figure.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def updatePlots(self, data, labels, method_name):
        # Очищаем графики
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Если данных слишком много, берем выборку
        if len(data) > 1000:
            sample_idx = np.random.choice(len(data), 1000, replace=False)
            data_sample = data.iloc[sample_idx]
            labels_sample = labels[sample_idx]
        else:
            data_sample = data
            labels_sample = labels
        
        # Преобразуем данные в 2D для визуализации
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data_sample)
        
        # График 1: Улучшенный scatter plot кластеров
        unique_labels = np.unique(labels_sample)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        # Создаем scatter plot с подписями
        for label, color in zip(unique_labels, colors):
            mask = labels_sample == label
            self.ax1.scatter(data_2d[mask, 0], 
                            data_2d[mask, 1],
                            c=[color],
                            label=f'Кластер {label}',
                            alpha=0.6,
                            edgecolors='w',
                            linewidths=0.5)
        
        # Добавляем центроиды кластеров
        for label in unique_labels:
            mask = labels_sample == label
            if sum(mask) > 0:  # Защита от пустых кластеров
                centroid = data_2d[mask].mean(axis=0)
                self.ax1.scatter(centroid[0], centroid[1],
                                marker='o',
                                c='white',
                                s=200,
                                edgecolors='k',
                                alpha=0.9)
                self.ax1.scatter(centroid[0], centroid[1],
                                marker='$%d$' % label,
                                c='black',
                                s=50)
        
        # Настройки графика
        self.ax1.set_title(f'2D проекция кластеров ({method_name})', pad=20)
        self.ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсии)')
        self.ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсии)')
        self.ax1.legend(title="Кластеры", bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax1.grid(True, linestyle='--', alpha=0.6)

        # График 2: дендрограмма (для небольшого набора данных)
        # if len(data) <= 100:
        Z = linkage(data_sample, method='average', metric='chebyshev')
        dendrogram(Z, truncate_mode="level", p=3, ax=self.ax2)
        self.ax2.set_xlabel("Точки данных (или объединённые кластеры)")
        self.ax2.set_title('Дендрограмма')
        
        # График 3: гистограмма распределения по кластерам
        unique, counts = np.unique(labels_sample, return_counts=True)
        self.ax3.bar(unique, counts)
        self.ax3.set_title('Распределение по кластерам')
        self.ax3.set_xlabel('Номер кластера')
        self.ax3.set_ylabel('Количество объектов')
        self.ax3.set_xticks(np.arange(min(unique), max(unique)+1, 1))
        
        plt.tight_layout()
        self.canvas.draw()
    
    def updateTable(self, df=None):
        if df is None:
            df = self.dataFrame
        
        # Очищаем текущее содержимое таблицы
        self.tree.delete(*self.tree.get_children())
        
        if df.empty:
            return
        
        # Игнорируем первый столбец (индекс) и берем только остальные колонки
        columns = list(df.columns)
        
        # Настраиваем колонки
        self.tree["columns"] = columns
        self.tree.heading("#0", text="#")
        self.tree.column("#0", width=50, stretch=tk.NO)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.W)
        
        # Добавляем данные с автоматической нумерацией строк
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            values = list(row)
            item = self.tree.insert("", tk.END, text=str(i), values=values)
    
    def updateStatistics(self, text):
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, text)
        self.stats_text.config(state=tk.DISABLED)
    
    def name_clusters(self, labels, df):
        # Проверка совпадения размеров
        if len(labels) != len(df):
            raise ValueError(f"Длина labels ({len(labels)}) не совпадает с длиной df ({len(df)})")
        
        cluster_names = {}
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_data = df[cluster_mask]
            
            # Проверка, что кластер не пустой
            if len(cluster_data) == 0:
                cluster_names[cluster_id] = f"Empty_Cluster_{cluster_id}"
                continue
                
            # Анализ ключевых характеристик
            try:
                avg_range = cluster_data['Electric Range'].mean()
                make_counts = cluster_data['Make'].value_counts().idxmax()
                geo_mode = cluster_data['County'].value_counts().idxmax()[:3]  # Берем первые 3 буквы
                
                # Формирование имени
                if avg_range > 200:
                    range_type = 'LongRange'
                elif avg_range > 100:
                    range_type = 'MidRange'
                else:
                    range_type = 'ShortRange'
                    
                ev_type = cluster_data['Electric Vehicle Type'].mode()[0]
                if 'PHEV' in str(ev_type):
                    ev_suffix = '_Hybrid'
                else:
                    ev_suffix = '_FullElectric'
                    
                name = f"{make_counts[:4]}_{range_type}_{geo_mode}{ev_suffix}"
                cluster_names[cluster_id] = name
                
            except Exception as e:
                print(f"Ошибка при обработке кластера {cluster_id}: {str(e)}")
                cluster_names[cluster_id] = f"Unknown_Cluster_{cluster_id}"
        
        return cluster_names

    def preprocessData(self, df):
        # 1. Удаляем столбцы с уникальными значениями
        df = df.drop(columns=[col for col in df.columns 
                            if df[col].nunique() == len(df)], errors='ignore')
        
        # 2. Разделяем признаки по типам
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        # 3. Заполняем пропуски перед кодированием
        for col in categorical_cols:
            df[col] = df[col].fillna('MISSING')  # Заменяем NaN специальным значением
        
        # 4. Кодируем категориальные переменные
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        
        # 5. Заполняем пропуски в числовых признаках
        if len(numeric_cols) > 0:
            # Заменяем NaN медианными значениями
            imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
            # Масштабируем только числовые признаки
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # 6. Проверяем на оставшиеся пропуски
        if df.isna().any().any():
            raise ValueError("Обнаружены пропущенные значения после препроцессинга")
        
        return df
    
    def clusterData(self, data, n_clusters=4):
        # Перебор разного количества кластеров
        # best_score = -1
        # best_n = 2

        # for n in range(2, 10):
        #     model = AgglomerativeClustering(n_clusters=n, metric='chebyshev', linkage='average')
        #     labels = model.fit_predict(data)
        #     score = silhouette_score(data, labels)
        #     if score > best_score:
        #         best_score = score
        #         best_n = n

        # print(f"Оптимальное количество кластеров: {best_n} (score={best_score:.2f})")
        
        # Иерархическая кластеризация с расстоянием Чебышева
        clustering = AgglomerativeClustering(
            # distance_threshold=0.3, 
            # n_clusters=None,
            n_clusters=n_clusters, 
            metric='chebyshev', 
            linkage='average'
        )
        labels = clustering.fit_predict(data)

        # Оценка качества кластеризации (отделимость кластеров)
        silhouette = silhouette_score(data, labels)
        
        return labels, silhouette
    
    def clusterFullData(self):
        if self.dataFrame.empty:
            messagebox.showerror("Ошибка", "Сначала загрузите датасет!")
            return
        
        try:
            # Предварительная обработка данных
            processed_data = self.preprocessData(self.dataFrame.copy())
            
            # Кластеризация
            # n_clusters = 3  # Можно добавить выбор количества кластеров
            labels, silhouette = self.clusterData(processed_data)
            n_clusters = len(np.unique(labels))
            
            # Сохраняем результаты
            self.clusteredDataFrame = self.dataFrame.copy()
            self.clusteredDataFrame['Cluster'] = labels
            
            # Генерация имен кластеров
            cluster_names = self.name_clusters(labels, self.clusteredDataFrame)
            
            # Добавляем имена в DataFrame (убедитесь, что размеры совпадают)
            if len(labels) == len(self.clusteredDataFrame):
                self.clusteredDataFrame['Cluster_Name'] = [cluster_names[label] for label in labels]
            else:
                raise ValueError("Несоответствие размеров labels и DataFrame")

            # Обновляем таблицу и графики
            self.updateTable(self.clusteredDataFrame)
            self.updatePlots(processed_data, labels, "полный датасет")
            
            # # Выводим статистику
            # stats_text = f"Кластеризация полного датасета:\n"
            # stats_text += f"- Количество кластеров: {n_clusters}\n"
            # stats_text += f"- Оценка отделимости (silhouette score): {silhouette:.3f}\n"
            # stats_text += f"- Размеры кластеров:\n"
            # for cluster in range(n_clusters):
            #     size = sum(labels == cluster)
            #     stats_text += f"  Кластер {cluster}: {size} объектов ({size/len(labels):.1%})\n"
            
            # Формируем отчет
            stats_text = self.generate_cluster_report(labels, silhouette, cluster_names)
            self.updateStatistics(stats_text)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при кластеризации:\n{str(e)}")
            print ("Произошла ошибка при кластеризации:\n", str(e))
    
    def generate_cluster_report(self, labels, silhouette, cluster_names):
        report = [
            f"Кластеризация полного датасета:",
            f"- Количество кластеров: {len(cluster_names)}",
            f"- Оценка отделимости (silhouette score): {silhouette:.3f}",
            "\nДетализация кластеров:"
        ]
        
        for cluster_id, name in cluster_names.items():
            cluster_data = self.clusteredDataFrame[self.clusteredDataFrame['Cluster'] == cluster_id]
            if len(cluster_data) == 0:
                continue
                
            report.extend([
                f"\nКластер {name} (ID: {cluster_id}):",
                f"- Размер: {len(cluster_data)} объектов ({len(cluster_data)/len(labels):.1%})",
                # f"- Средний запас хода: {cluster_data['Electric Range'].mean():.1f} миль",
                # f"- Тип ТС: {cluster_data['Electric Vehicle Type'].mode()[0]}",
                # f"- Типичный регион: {cluster_data['County'].mode()[0]}",
                # f"- Популярные марки: {', '.join(cluster_data['Make'].value_counts().head(3).index.tolist())}"
            ])
        
        return "\n".join(report)

    def selectInformativeFeatures(self):
        if self.dataFrame.empty:
            messagebox.showerror("Ошибка", "Сначала загрузите датасет!")
            return
        
        try:
            # Предварительная обработка данных
            processed_data = self.preprocessData(self.dataFrame.copy())
            
            # variances = np.var(processed_data, axis=0)

            # # Создаем DataFrame для наглядности
            # variance_df = pd.DataFrame({
            #     'Признак': processed_data.columns,
            #     'Дисперсия': variances
            # }).sort_values('Дисперсия', ascending=False)

            # print("Дисперсия признаков до отбора:")
            # print(variance_df)

            # Выбор признаков на основе компактности (дисперсии)
            selector = VarianceThreshold(threshold=1.1)  # Можно настроить порог
            selected_data = selector.fit_transform(processed_data)
            
            # Получаем индексы выбранных признаков
            selected_indices = selector.get_support(indices=True)
            selected_features = processed_data.columns[selected_indices]
            
            # Сохраняем выбранные признаки
            self.selectedFeaturesDataFrame = self.dataFrame[selected_features]
            
            # Выводим информацию
            stats_text = "Выбраны наиболее информативные признаки:\n"
            stats_text += ", ".join(selected_features) + "\n"
            stats_text += f"Всего выбрано {len(selected_features)} из {len(processed_data.columns)} признаков"
            
            self.updateStatistics(stats_text)
            self.updateTable(self.selectedFeaturesDataFrame)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при выборе признаков:\n{str(e)}")
    
    def clusterSelectedFeatures(self):
        if self.selectedFeaturesDataFrame.empty:
            messagebox.showerror("Ошибка", "Сначала выберите информативные признаки!")
            return
        
        try:
            # Предварительная обработка данных
            processed_data = self.preprocessData(self.selectedFeaturesDataFrame.copy())
            
            # Кластеризация
            # n_clusters = 3
            labels, silhouette = self.clusterData(processed_data)
            n_clusters = len(np.unique(labels))
            
            # Сохраняем результаты
            clustered_df = self.selectedFeaturesDataFrame.copy()
            clustered_df['Cluster'] = labels
            
            # Обновляем таблицу и графики
            self.updateTable(clustered_df)
            self.updatePlots(processed_data, labels, "отобранные признаки")
            
            # Выводим статистику
            stats_text = f"Кластеризация по отобранным признакам:\n"
            stats_text += f"- Количество кластеров: {n_clusters}\n"
            stats_text += f"- Оценка отделимости (silhouette score): {silhouette:.3f}\n"
            stats_text += f"- Размеры кластеров:\n"
            for cluster in range(n_clusters):
                size = sum(labels == cluster)
                stats_text += f"  Кластер {cluster}: {size} объектов ({size/len(labels):.1%})\n"
            
            self.updateStatistics(stats_text)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при кластеризации:\n{str(e)}")
            print ("Произошла ошибка при кластеризации:\n", str(e))

    def anonymizeData(self):
        if self.dataFrame.empty:
            messagebox.showerror("Ошибка", "Сначала загрузите датасет!")
            return
        
        try:
            anonymized_df = self.dataFrame.copy()

            anonymized_df['VIN (1-10)'] = anonymized_df['VIN (1-10)'].apply(lambda x: x[:1] + '********')
            
            # 1. Обработка идентификаторов (полное удаление)
            anonymized_df = anonymized_df.drop(columns=[ 
                'DOL Vehicle ID',
                '2020 Census Tract',
                'State'
            ], errors='ignore')
            
            # 2. Географические данные - агрегируем до более высокого уровня
            anonymized_df['County'] = anonymized_df['County'].apply(lambda x: x[:3] + '***')
            anonymized_df['City'] = 'City_' + anonymized_df['City'].astype('category').cat.codes.astype(str)
            
            # 3. Координаты - округляем до 1 знака после запятой
            if 'Vehicle Location' in anonymized_df:
                anonymized_df['Vehicle Location'] = anonymized_df['Vehicle Location'].str.replace(
                    r'(-?\d+\.\d{2})\d+', 
                    lambda m: f"{float(m.group(1)):.1f}",
                    regex=True
                )
            
            # 4. Технические характеристики - группируем в категории
            def range_category(x):
                if x == 0: return 'No_Range'
                elif x < 50: return 'Short_Range'
                elif x < 200: return 'Medium_Range'
                else: return 'Long_Range'
                
            anonymized_df['Electric Range'] = anonymized_df['Electric Range'].apply(range_category)
            
            # 5. Производитель и модель - обобщаем
            anonymized_df['Make'] = anonymized_df['Make'].apply(lambda x: 'Manufacturer_' + str(hash(x) % 10))
            anonymized_df['Model'] = anonymized_df['Model'].apply(lambda x: 'Model_' + str(hash(x) % 20))
            
            # 6. Почтовый индекс - оставляем только первые 3 цифры
            anonymized_df['Postal Code'] = anonymized_df['Postal Code'].astype(str).str[:3]
            
            # 7. Год модели - группируем по 3-летним периодам
            anonymized_df['Model Year'] = (anonymized_df['Model Year'] // 3 * 3).astype(str) + 's'
            
            # 8. Электрические параметры - сохраняем как есть (важно для кластеризации)
            # CAFV Eligibility и Electric Vehicle Type не изменяем
            
            # 9. Числовые данные - добавляем контролируемый шум
            numeric_cols = anonymized_df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                noise = np.random.normal(0, anonymized_df[col].std() * 0.05, size=len(anonymized_df))  # Меньше шума
                anonymized_df[col] = anonymized_df[col] + noise
                anonymized_df[col] = anonymized_df[col].round(2)  # Округляем для читаемости
            
            self.anonymizedDataFrame = anonymized_df
            self.updateStatistics("Данные оптимизированы для кластеризации")
            self.updateTable(self.anonymizedDataFrame)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при обезличивании:\n{str(e)}")
            print("Error:", str(e))    
            
    def clusterAnonymizedData(self):
        if self.anonymizedDataFrame.empty:
            messagebox.showerror("Ошибка", "Сначала обезличьте данные!")
            return
        
        try:
            # Предварительная обработка данных
            processed_data = self.preprocessData(self.anonymizedDataFrame.copy())
            
            # Кластеризация
            # n_clusters = 3
            labels, silhouette = self.clusterData(processed_data)
            n_clusters = len(np.unique(labels))

            # Сохраняем результаты
            clustered_df = self.anonymizedDataFrame.copy()
            clustered_df['Cluster'] = labels
            
            # Обновляем таблицу и графики
            self.updateTable(clustered_df)
            self.updatePlots(processed_data, labels, "обезличенные данные")
            
            # Выводим статистику
            stats_text = f"Кластеризация обезличенных данных:\n"
            stats_text += f"- Количество кластеров: {n_clusters}\n"
            stats_text += f"- Оценка отделимости (silhouette score): {silhouette:.3f}\n"
            stats_text += f"- Размеры кластеров:\n"
            for cluster in range(n_clusters):
                size = sum(labels == cluster)
                stats_text += f"  Кластер {cluster}: {size} объектов ({size/len(labels):.1%})\n"
            
            self.updateStatistics(stats_text)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при кластеризации:\n{str(e)}")
            print ("Произошла ошибка при кластеризации:\n", str(e))
    
    def loadCSV(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        
        try:
            self.dataFrame = pd.read_csv(path)
            self.originalDataFrame = self.dataFrame.copy()
            self.updateTable()
            self.updateStatistics(f"Данные успешно загружены\nЗаписей: {len(self.dataFrame)}\nСтолбцов: {len(self.dataFrame.columns)}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetClusteringApp(root)
    root.mainloop()