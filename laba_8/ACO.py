import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
import networkx as nx
import json
import math
import random, time

class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Поиск кратчайшего гамильтонова цикла")

        self.graph = nx.DiGraph()
        self.nodes = []
        self.r = 15
        self.edges = []
        self.edge_labels = {}
        self.edge_ids = {}
        self.start_node = None

        self.text_frame = ttk.Frame(self.root, width=700)
        self.text_frame.pack(side="left", fill=tk.Y, padx=5, pady=5)

        self.label_text_path = tk.Label(self.text_frame, text="Полученный путь")
        self.label_text_path.pack()

        self.cycle_text = tk.Text(self.text_frame, width=30, wrap=tk.WORD)
        self.cycle_text.pack()

        self.label_text_length = tk.Label(self.text_frame, text="Длина пути")
        self.label_text_length.pack()

        self.length_text = tk.Entry(self.text_frame, width=30)
        self.length_text.pack()

        self.label_time = tk.Label(self.text_frame, text="Время выполнения")
        self.label_time.pack()

        self.time_text = tk.Entry(self.text_frame, width=30)
        self.time_text.pack()

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(side="left", anchor='nw')

        self.label_0 = tk.Label (self.main_frame, text="Стартовый граф")
        self.label_0.grid(row=0, column=0, padx=5, pady=5)

        self.canvas = tk.Canvas(self.main_frame, width=600, height=400, bg="white")
        self.canvas.grid(row=1, column=0, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.add_node)
        self.canvas.bind("<Button-3>", self.start_edge)

        self.label_1 = tk.Label (self.main_frame, text="Полученный цикл")
        self.label_1.grid(row=2, column=0, padx=5, pady=5)

        self.canvas_cycle = tk.Canvas(self.main_frame, width=600, height=400, bg="white")
        self.canvas_cycle.grid(row=3, column=0, padx=5, pady=5)

        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(side="left", anchor="n")

        self.setup_table()

        self.find_cycle_button = ttk.Button(self.button_frame, text="Поиск цикла", command=self.draw_cycle)
        self.find_cycle_button.pack(anchor="w", pady=5)

        self.clear_button = ttk.Button(self.button_frame, text="Очистить", command=self.clear_canvas)
        self.clear_button.pack(anchor="w", pady=5)
        
        self.load_button = ttk.Button(self.button_frame, text="Загрузить граф из JSON", command=self.load_json)
        self.load_button.pack(anchor="w", pady=5)

        # Переменная для хранения состояния чек-бокса
        self.mode = tk.IntVar(value=1)

        # Создаем чек-бокс
        self.checkbox = ttk.Checkbutton(
            self.button_frame,
            text="Использовать модификацию элитных муравьев",
            variable=self.mode,
            onvalue=1,
            offvalue=0
        )
        self.checkbox.pack(anchor="w", pady=5)

        self.label_ants_count = tk.Label(self.button_frame, text="Количество муравьев")
        self.label_ants_count.pack()

        self.ants_entry = self.create_int_entry(default="10")

        self.label_iterations = tk.Label(self.button_frame, text="Количество итераций")
        self.label_iterations.pack()

        self.iterations_entry = self.create_int_entry(default="100")

        self.label_rho = tk.Label(self.button_frame, text="Интенсивность испарения")
        self.label_rho.pack()

        self.rho_entry = self.create_int_entry(default="1")

        self.label_q = tk.Label(self.button_frame, text="Количество добавляемого феромона")
        self.label_q.pack()

        self.q_entry = self.create_int_entry(default="100")
        
        self.label_alpha = tk.Label(self.button_frame, text="Коэффицент влияния феромона")
        self.label_alpha.pack()

        self.alpha_entry = self.create_int_entry(default="1")

        self.label_beta = tk.Label(self.button_frame, text="Коэффицент влияния расстояния")
        self.label_beta.pack()

        self.beta_entry = self.create_int_entry(default="2")

        self.node_count = 1

    def add_node(self, event):
        x, y = event.x, event.y
        node = f"({x}, {y})"
        if node not in self.nodes:
            self.graph.add_node(node, order=self.node_count)
            self.nodes.append(node)
            self.canvas.create_oval(x-self.r, y-self.r, x+self.r, y+self.r, fill="red")
            self.canvas.create_text(x, y, text=str(self.node_count), fill="white")
            self.node_count += 1

    def clear_canvas(self):
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()
        self.canvas.delete("all")
        self.canvas_cycle.delete("all")
        self.table.delete(*self.table.get_children())
        self.node_count = 1
        self.edge_labels.clear()
        self.cycle_text.delete("1.0", tk.END)
        self.length_text.delete(0, "end")
        self.time_text.delete(0, "end")

    def create_int_entry(self, default):
        entry = ttk.Entry(
            self.button_frame,
            validate='key',
            validatecommand=(self.root.register(self.validate_int), '%P')
        )
        entry.insert(0, default)
        entry.pack()
        return entry

    def delete_edge(self, event):
        # Определяем, была ли нажата ячейка таблицы
        region = self.table.identify_region(event.x, event.y)
        if region == "cell":
            # Получаем выбранный элемент таблицы
            item = self.table.selection()[0]
            values = self.table.item(item, "values")

            # Извлекаем начальную и конечную вершины из таблицы
            start_order = int(values[0])  # Порядковый номер начальной вершины
            end_order = int(values[1])    # Порядковый номер конечной вершины

            # Находим соответствующие вершины в графе
            start_node = self.nodes[start_order - 1]  # Порядковые номера начинаются с 1
            end_node = self.nodes[end_order - 1]

            # Удаляем ребро из графа
            if self.graph.has_edge(start_node, end_node):
                self.graph.remove_edge(start_node, end_node)

            # Удаляем ребро из списка edges
            edge_to_remove = None
            for edge in self.edges:
                if (edge[0] == start_node and edge[1] == end_node) or (edge[0] == end_node and edge[1] == start_node):
                    edge_to_remove = edge
                    break
            if edge_to_remove:
                self.edges.remove(edge_to_remove)

            # Удаляем ребро из таблицы
            self.table.delete(item)

            # Удаляем визуальное отображение ребра с canvas
            if (start_node, end_node) in self.edge_ids:
                line_id, text_id = self.edge_ids[(start_node, end_node)]
                self.canvas.delete(line_id)
                self.canvas.delete(text_id)
                del self.edge_ids[(start_node, end_node)]

    def distance(self, node1, node2):
        # Используем вес ребра из графа, если он задан
        if self.graph.has_edge(node1, node2):
            return self.graph[node1][node2]['weight']
        else:
            return float('inf')  # Если ребра нет, возвращаем бесконечность

    def draw_cycle (self):
        startTime = time.perf_counter()
        cycle = self.find_cycle()
        endTime = time.perf_counter()

        if cycle:
            min_hamiltonian_cycle = cycle[0]
            min_cycle_length = cycle[1]

            # Преобразуем цикл в строку
            cycle_order = [self.graph.nodes[node]['order'] for node in min_hamiltonian_cycle]
            cycle_str = "-".join(map(str, cycle_order)) + '-' + str(cycle_order[0])

            self.update_text(self.cycle_text, cycle_str)
            self.update_entry(self.length_text, f"{min_cycle_length}")
            self.update_entry(self.time_text, f"{endTime-startTime:.4f}")

        else:
            return

        self.canvas_cycle.delete("cycle")
        for i in range(len(min_hamiltonian_cycle) - 1):
            x1, y1 = map(int, min_hamiltonian_cycle[i].strip("()").split(", "))
            x2, y2 = map(int, min_hamiltonian_cycle[i+1].strip("()").split(", "))

            weight = self.distance(min_hamiltonian_cycle[i], min_hamiltonian_cycle[i+1])

            start_x, start_y = self.get_edge_point(x1, y1, x2, y2)
            end_x, end_y = self.get_edge_point(x2, y2, x1, y1)

            self.canvas_cycle.create_line(start_x, start_y, end_x, end_y, fill="red", arrow=tk.LAST, width=2, arrowshape=(10, 15, 4), tags="cycle")         
            self.canvas_cycle.create_oval(x1-self.r, y1-self.r, x1+self.r, y1+self.r, fill="black")
            self.canvas_cycle.create_text(x1, y1, text=str(self.graph.nodes[min_hamiltonian_cycle[i]]['order']), fill="white")
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            self.canvas_cycle.create_text(mid_x, mid_y, text=f"{weight}", fill="blue", tags="cycle")

        # Рисуем замыкающее ребро
        x1, y1 = map(int, min_hamiltonian_cycle[-1].strip("()").split(", "))
        x2, y2 = map(int, min_hamiltonian_cycle[0].strip("()").split(", "))
        weight = self.distance(min_hamiltonian_cycle[-1], min_hamiltonian_cycle[0])

        start_x, start_y = self.get_edge_point(x1, y1, x2, y2)
        end_x, end_y = self.get_edge_point(x2, y2, x1, y1)

        self.canvas_cycle.create_line(start_x, start_y, end_x, end_y, fill="red", arrow=tk.LAST, width=2, arrowshape=(10, 15, 4), tags="cycle")         

        self.canvas_cycle.create_oval(x1-self.r, y1-self.r, x1+self.r, y1+self.r, fill="black")
        self.canvas_cycle.create_text(x1, y1, text=str(self.graph.nodes[min_hamiltonian_cycle[-1]]['order']), fill="white")
        self.canvas_cycle.create_oval(x2-self.r, y2-self.r, x2+self.r, y2+self.r, fill="black")
        self.canvas_cycle.create_text(x2, y2, text=str(self.graph.nodes[min_hamiltonian_cycle[0]]['order']), fill="white")
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        self.canvas_cycle.create_text(mid_x, mid_y, text=f"{weight}", fill="blue", tags="cycle")
    
    def get_int_value_from_entry(self, entry):
        """Возвращает int значение из указанного Entry или 0 по умолчанию"""
        text = entry.get()
        try:
            return int(text) if text else 0
        except ValueError:
            return 0
    
    def find_cycle_with_ant_colony(self, use_elitism):
        # Параметры алгоритма
        ants_count = self.get_int_value_from_entry(self.ants_entry) or 10
        iterations = self.get_int_value_from_entry(self.iterations_entry) or 100
        alpha = self.get_int_value_from_entry(self.alpha_entry) or 1  # влияние феромона
        beta = self.get_int_value_from_entry(self.beta_entry) or 2   # влияние расстояния
        rho = self.get_int_value_from_entry(self.rho_entry)*0.1 or 0.1  # коэффициент испарения
        q = self.get_int_value_from_entry(self.q_entry) or 100    # количество феромона от одного муравья

        # Проверка, что граф содержит хотя бы 3 вершины
        if len(self.nodes) < 3:
            self.update_text(self.cycle_text, "Недостаточно вершин (нужно минимум 3)")
            return None

        # Проверка, что граф связный (хотя бы один возможный путь)
        if not self._has_possible_path():
            self.update_text(self.cycle_text, "Нет возможных путей между вершинами")
            return None

        # Инициализация феромонов
        pheromone = {}
        for node1 in self.nodes:
            for node2 in self.nodes:
                if node1 != node2 and self.graph.has_edge(node1, node2):
                    pheromone[(node1, node2)] = 1.0

        best_path = None
        best_length = float('inf')
        
        for _ in range(iterations):
            paths = []
            
            # Каждый муравей строит путь
            for _ in range(ants_count):
                path = self._construct_ant_path(pheromone, alpha, beta)
                if not path or len(path) != len(self.nodes):
                    continue  # Пропускаем некорректные пути
                
                length = self._calculate_path_length(path)
                if math.isinf(length):
                    continue  # Пропускаем пути с бесконечной длиной
                    
                paths.append((path, length))
                
                if length < best_length:
                    best_length = length
                    best_path = path
            
            # Если ни один муравей не нашел путь, пропускаем обновление
            if not paths:
                continue
                
            # Обновление феромонов
            self._update_pheromones(pheromone, paths, rho, q)
            
            # Оптимизация элитных муравьев (только если есть лучший путь)
            if use_elitism and best_path is not None:
                self._update_pheromones(pheromone, [(best_path, best_length)], rho, q, weight=5)
        
        if best_path is None:
            self.update_text(self.cycle_text, "Гамильтонов цикл не найден")
            return None
                            
        return best_path, best_length

    def _construct_ant_path(self, pheromone, alpha, beta):
        start_node = random.choice(self.nodes)
        path = [start_node]
        unvisited = set(self.nodes)
        unvisited.remove(start_node)
        
        while unvisited:
            current = path[-1]
            next_node = self._select_next_node(current, unvisited, pheromone, alpha, beta)
            path.append(next_node)
            unvisited.remove(next_node)
        
        return path

    def _select_next_node(self, current, unvisited, pheromone, alpha, beta):
        probabilities = []
        total = 0.0
        
        for node in unvisited:
            if self.graph.has_edge(current, node):
                ph = pheromone.get((current, node), 1e-10)
                dist = self.distance(current, node)
                attractiveness = (ph ** alpha) * ((1/dist) ** beta)
                probabilities.append((node, attractiveness))
                total += attractiveness
        
        # Нормализация вероятностей
        if total > 0:
            probabilities = [(node, prob/total) for node, prob in probabilities]
        else:
            # Если все вероятности нулевые, выбираем случайный узел
            return random.choice(list(unvisited))
        
        # Рулеточный выбор
        r = random.random()
        cumulative = 0.0
        for node, prob in probabilities:
            cumulative += prob
            if r <= cumulative:
                return node
        
        return probabilities[-1][0]

    def _calculate_path_length(self, path):
        length = 0
        for i in range(len(path)-1):
            length += self.distance(path[i], path[i+1])
        length += self.distance(path[-1], path[0])  # Замыкаем цикл
        return length

    def _has_possible_path(self):
        """Проверяет, существует ли хотя бы один возможный путь между вершинами."""
        for node1 in self.nodes:
            has_edge = False
            for node2 in self.nodes:
                if node1 != node2 and self.graph.has_edge(node1, node2):
                    has_edge = True
                    break
            if not has_edge:
                return False
        return True

    def _update_pheromones(self, pheromone, paths, rho, q, weight=1):
        """Обновление феромонов с проверкой входных данных."""
        if not paths:
            return
            
        # Испарение феромонов
        for edge in pheromone:
            pheromone[edge] *= (1 - rho)
        
        # Добавление нового феромона
        for path, length in paths:
            if path is None or len(path) != len(self.nodes) or math.isinf(length):
                continue
                
            delta_pheromone = q / length * weight
            for i in range(len(path)-1):
                edge = (path[i], path[i+1])
                if edge in pheromone:
                    pheromone[edge] += delta_pheromone
            # Замыкаем цикл
            edge = (path[-1], path[0])
            if edge in pheromone:
                pheromone[edge] += delta_pheromone
            
    def find_cycle(self):
        use_elitism = (self.mode.get() == 1)
        return self.find_cycle_with_ant_colony(use_elitism)
    
    def get_edge_point(self, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance == 0:
            return x1, y1
        scale = self.r / distance
        return x1 + dx * scale, y1 + dy * scale

    def load_json(self):
        # Открываем диалог выбора файла
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not filename:
            return  # Если файл не выбран, выходим

        # Загружаем данные из JSON
        with open(filename, "r") as f:
            loaded_data = json.load(f)

        self.clear_canvas()
        # Добавляем вершины и ребра в граф
        for node_data in loaded_data["nodes"]:
            x, y = node_data["x"], node_data["y"]
            node = f"({x}, {y})"
            self.graph.add_node(node, order=self.node_count)
            self.nodes.append(node)
            self.canvas.create_oval(x - self.r, y - self.r, x + self.r, y + self.r, fill="red")
            self.canvas.create_text(x, y, text=str(self.node_count), fill="white")
            self.node_count += 1

        for edge_data in loaded_data["edges"]:
            start_node = f"({edge_data['from']['x']}, {edge_data['from']['y']})"
            end_node = f"({edge_data['to']['x']}, {edge_data['to']['y']})"
            weight = edge_data["weight"]
            
            # Добавляем ребро в таблицу с порядковыми номерами вершин
            start_order = self.graph.nodes[start_node]["order"]
            end_order = self.graph.nodes[end_node]["order"]
            self.table.insert("", "end", values=(start_order, end_order, f"{weight}"))

            self.graph.add_edge(start_node, end_node, weight=weight)
            self.edges.append((start_node, end_node, weight))

            # Рисуем ребро на Canvas
            x1, y1 = edge_data["from"]["x"], edge_data["from"]["y"]
            x2, y2 = edge_data["to"]["x"], edge_data["to"]["y"]

            # Корректируем координаты начала и конца ребра
            start_x, start_y = self.get_edge_point(x1, y1, x2, y2)
            end_x, end_y = self.get_edge_point(x2, y2, x1, y1)

            # Рисуем направленное ребро с весом
            line_id = self.canvas.create_line(start_x, start_y, end_x, end_y, fill="blue", arrow=tk.LAST, width=2, arrowshape=(10, 15, 4))
            mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
            
            text_id = self.canvas.create_text(mid_x, mid_y, text=f"{weight}", fill="black")
            self.edge_labels[(self.start_node, end_node)] = text_id  

            self.edge_ids[(self.start_node, end_node)] = (line_id, text_id)

    def on_double_click(self, event):
        # Получаем выбранную строку и столбец
        region = self.table.identify_region(event.x, event.y)
        if region == "cell":
            column = self.table.identify_column(event.x)
            item = self.table.selection()[0]
            value = self.table.item(item, "values")
            column_index = int(column[1:]) - 1 

            # Создаем Entry для редактирования
            entry = ttk.Entry(self.button_frame)
            entry.insert(0, value[column_index])
            entry.place(x=event.x, y=event.y, width=100)

        def save_edit(event):
            # Сохраняем новое значение
            new_value = entry.get()
            values = list(value)
            values[column_index] = new_value
            self.table.item(item, values=values)

            # Обновляем вес ребра в графе
            start_node = self.nodes[int(values[0]) - 1]
            end_node = self.nodes[int(values[1]) - 1]
            if self.graph.has_edge(start_node, end_node):
                self.graph[start_node][end_node]["weight"] = float(new_value)

                # Обновляем подпись на canvas
                if (start_node, end_node) in self.edge_labels:
                    text_id = self.edge_labels[(start_node, end_node)]
                    self.canvas.itemconfig(text_id, text=f"{float(new_value)}")

            entry.destroy()

        entry.bind("<Return>", save_edit)
        entry.bind("<FocusOut>", save_edit)

    def setup_table(self):
        # Настройка таблицы
        self.table = ttk.Treeview(self.button_frame, columns=("start", "end", "weight"), show="headings", height=20)
        
        # Настройка заголовков с функцией сортировки
        self.table.heading("start", text="Начальная вершина", 
                        command=lambda: self.sort_table("start", False))
        self.table.heading("end", text="Конечная вершина", 
                        command=lambda: self.sort_table("end", False))
        self.table.heading("weight", text="Вес ребра")
        
        self.table.pack(side="top")

        # Привязка двойного клика для редактирования
        self.table.bind("<Double-1>", self.on_double_click)
        self.table.bind("<Delete>", self.delete_edge)

    def sort_table(self, column, reverse):
        """Сортировка таблицы по указанному столбцу"""
        # Получаем все элементы таблицы
        items = [(self.table.set(item, column), item) for item in self.table.get_children('')]
        
        # Сортируем элементы как числа (если столбец содержит числа)
        try:
            items.sort(key=lambda x: int(x[0]), reverse=reverse)
        except ValueError:
            items.sort(key=lambda x: x[0], reverse=reverse)
        
        # Перемещаем элементы в отсортированном порядке
        for index, (val, item) in enumerate(items):
            self.table.move(item, '', index)
        
        # Устанавливаем обратный порядок для следующей сортировки
        self.table.heading(column, 
                        command=lambda: self.sort_table(column, not reverse))
        
    def start_edge(self, event):
        x, y = event.x, event.y
        closest_node = None
        min_distance = float('inf')
        for node in self.nodes:
            nx, ny = map(int, node.strip("()").split(", "))
            distance = ((x - nx) ** 2 + (y - ny) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_node = node
        if closest_node:
            if self.start_node is None:
                self.start_node = closest_node
            else:
                end_node = closest_node
                # Запрашиваем вес ребра у пользователя
                weight = simpledialog.askfloat("Вес ребра", "Введите вес ребра:", parent=self.root)
                #weight = 5
                if weight is not None:  # Если пользователь ввел вес
                    if (self.start_node, end_node) not in self.edges and (end_node, self.start_node) not in self.edges:
                        self.graph.add_edge(self.start_node, end_node, weight=weight)
                        self.edges.append((self.start_node, end_node, weight))

                        # Добавляем ребро в таблицу с порядковыми номерами вершин
                        start_order = self.graph.nodes[self.start_node]["order"]
                        end_order = self.graph.nodes[end_node]["order"]
                        self.table.insert("", "end", values=(start_order, end_order, f"{weight}"))

                        # Получаем координаты центров вершин
                        x1, y1 = map(int, self.start_node.strip("()").split(", "))
                        x2, y2 = map(int, end_node.strip("()").split(", "))

                        # Корректируем координаты начала и конца ребра
                        start_x, start_y = self.get_edge_point(x1, y1, x2, y2)
                        end_x, end_y = self.get_edge_point(x2, y2, x1, y1)

                        # Рисуем направленное ребро с весом
                        line_id = self.canvas.create_line(start_x, start_y, end_x, end_y, fill="blue", arrow=tk.LAST, width=2, arrowshape=(10, 15, 4))
                        mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
                        
                        text_id = self.canvas.create_text(mid_x, mid_y, text=f"{weight}", fill="black")
                        self.edge_labels[(self.start_node, end_node)] = text_id  

                        self.edge_ids[(self.start_node, end_node)] = (line_id, text_id)
                self.start_node = None

    def validate_int(self, new_text):
        if not new_text:  # Разрешаем пустую строку (будет обработано как 0)
            return True
        try:
            int(new_text)
            return True
        except ValueError:
            return False

    def update_entry(self, entry, message):
        entry.delete(0, "end")
        entry.insert(0, message)

    def update_text(self, text, message):
        text.delete("1.0", tk.END)
        text.insert("1.0", message)

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()
