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
            text="Использовать модификацию выбора стартовой вершины",
            variable=self.mode,
            onvalue=1,
            offvalue=0
        )
        self.checkbox.pack(anchor="w", pady=5)

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
        if len(self.nodes) < 3:
            self.update_text(self.cycle_text,"Недостаточно вершин")
            return
        
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
            self.update_text(self.cycle_text, "Гамильтонов цикл не найден")
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

    def find_cycle_without_modification(self):

        min_cycle_length = float('inf')
        min_hamiltonian_cycle = None

        start_node = random.choice(self.nodes)

        current_node = start_node
        cycle = [current_node]
        remaining_nodes = set(self.nodes)
        remaining_nodes.remove(start_node)

        while remaining_nodes:
            # Ищем ближайшего соседа, до которого есть направленное ребро
            closest_node = None
            min_distance = float('inf')

            for node in remaining_nodes:
                if self.graph.has_edge(current_node, node):  # Проверяем наличие ребра
                    distance = self.distance(current_node, node)
                    if distance < min_distance:
                        min_distance = distance
                        closest_node = node

            if closest_node is None:
                break  # Нет подходящего следующего узла

            cycle.append(closest_node)
            remaining_nodes.remove(closest_node)
            current_node = closest_node

        # Проверяем, можно ли замкнуть цикл
        if len(cycle) == len(self.nodes) and self.graph.has_edge(cycle[-1], cycle[0]):
            cycle_length = sum(self.distance(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1))
            cycle_length += self.distance(cycle[-1], cycle[0])

            if cycle_length < min_cycle_length:
                min_cycle_length = cycle_length
                min_hamiltonian_cycle = cycle

        if not min_hamiltonian_cycle:
            return

        return min_hamiltonian_cycle, cycle_length

    def find_cycle_with_modification(self):
        min_cycle_length = float('inf')
        min_hamiltonian_cycle = None

        for start_node in self.nodes:
            current_node = start_node
            cycle = [current_node]
            remaining_nodes = set(self.nodes)
            remaining_nodes.remove(start_node)

            while remaining_nodes:
                # Ищем ближайшего соседа, до которого есть направленное ребро
                closest_node = None
                min_distance = float('inf')

                for node in remaining_nodes:
                    if self.graph.has_edge(current_node, node):
                        distance = self.distance(current_node, node)
                        if distance < min_distance:
                            min_distance = distance
                            closest_node = node

                if closest_node is None:
                    break  # Нет подходящего следующего узла

                cycle.append(closest_node)
                remaining_nodes.remove(closest_node)
                current_node = closest_node

            # Проверяем, можно ли замкнуть цикл
            if len(cycle) == len(self.nodes) and self.graph.has_edge(cycle[-1], cycle[0]):
                cycle_length = sum(self.distance(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1))
                cycle_length += self.distance(cycle[-1], cycle[0])

                if cycle_length < min_cycle_length:
                    min_cycle_length = cycle_length
                    min_hamiltonian_cycle = cycle
                    
        if not min_hamiltonian_cycle:
            return

        return min_hamiltonian_cycle, cycle_length
    
    def find_cycle(self):
        if self.mode == 1:
            return self.find_cycle_with_modification()
        else:
            return self.find_cycle_without_modification()

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
        self.table.heading("start", text="Начальная вершина")
        self.table.heading("end", text="Конечная вершина")
        self.table.heading("weight", text="Вес ребра")
        self.table.pack(side="top")

        # Привязка двойного клика для редактирования
        self.table.bind("<Double-1>", self.on_double_click)
        self.table.bind("<Delete>", self.delete_edge)

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
