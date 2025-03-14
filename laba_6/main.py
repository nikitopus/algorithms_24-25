import tkinter as tk
from tkinter import ttk, simpledialog
import networkx as nx
import math
import random

class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Поиск кратчайшего гамильтонова цикла")

        self.graph = nx.DiGraph()
        self.nodes = []
        self.r = 15
        self.edges = []
        self.edge_labels = {}
        self.edge_ids = {}  # Словарь для хранения ID ребер на canvas
        self.start_node = None

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(side="left")

        self.canvas = tk.Canvas(self.main_frame, width=600, height=400, bg="white")
        self.canvas.grid(row=0, column=0, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.add_node)
        self.canvas.bind("<Button-3>", self.start_edge)

        self.canvas_cycle = tk.Canvas(self.main_frame, width=600, height=400, bg="white")
        self.canvas_cycle.grid(row=1, column=0, padx=5, pady=5)

        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(anchor='ne')

        self.find_cycle_button = ttk.Button(self.button_frame, text="Поиск цикла", command=self.draw_cycle)
        self.find_cycle_button.grid(row=1, column=0)

        self.clear_button = ttk.Button(self.button_frame, text="Очистить", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=1)

        # Переменная для хранения состояния чек-бокса
        self.mode = tk.IntVar(value=0)

        # Создаем чек-бокс
        self.checkbox = ttk.Checkbutton(
            self.button_frame,
            text="Использовать модификацию выбора стартовой вершины",
            variable=self.mode,
            onvalue=1,
            offvalue=0
        )
        self.checkbox.grid(row=2, column=0, columnspan=2, pady=5)

        self.setup_table()

        self.node_count = 1

    def setup_table(self):
        # Настройка таблицы
        self.table = ttk.Treeview(self.button_frame, columns=("start", "end", "weight"), show="headings", height=20)
        self.table.heading("start", text="Начальная вершина")
        self.table.heading("end", text="Конечная вершина")
        self.table.heading("weight", text="Вес ребра")
        self.table.grid(row=0, column=0, columnspan=2)

        # Привязка двойного клика для редактирования
        self.table.bind("<Double-1>", self.on_double_click)
        self.table.bind("<Delete>", self.delete_edge)

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

    def add_node(self, event):
        x, y = event.x, event.y
        node = f"({x}, {y})"
        if node not in self.nodes:
            self.graph.add_node(node, order=self.node_count)
            self.nodes.append(node)
            self.canvas.create_oval(x-self.r, y-self.r, x+self.r, y+self.r, fill="red")
            self.canvas.create_text(x, y, text=str(self.node_count), fill="white")
            self.node_count += 1

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

    def get_edge_point(self, x1, y1, x2, y2):
        """
        Возвращает координаты точки на границе круга (вершины),
        которая лежит на линии, соединяющей центры двух вершин.
        """
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx ** 2 + dy ** 2)
        if distance == 0:
            return x1, y1
        scale = self.r / distance
        return x1 + dx * scale, y1 + dy * scale

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

        if min_hamiltonian_cycle:
            # Преобразуем цикл в строку
            cycle_order = [self.graph.nodes[node]['order'] for node in min_hamiltonian_cycle]
            cycle_str = "-".join(map(str, cycle_order)) + '-' + str(cycle_order[0])

            # Вставляем строку в таблицу
            self.table.insert("", "end", values=("Кратчайший цикл:", cycle_str))            
            self.table.insert("", "end", values=("Итоговый вес:", f"{min_cycle_length}"))
        else:
            print("Гамильтонов цикл не найден")

        return min_hamiltonian_cycle

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

        if min_hamiltonian_cycle:
            # Преобразуем цикл в строку
            cycle_order = [self.graph.nodes[node]['order'] for node in min_hamiltonian_cycle]
            cycle_str = "-".join(map(str, cycle_order)) + '-' + str(cycle_order[0])

            # Вставляем строку в таблицу
            self.table.insert("", "end", values=("Кратчайший цикл:", cycle_str))            
            self.table.insert("", "end", values=("Итоговый вес:", f"{min_cycle_length}"))
        else:
            print("Гамильтонов цикл не найден")

        return min_hamiltonian_cycle
    
    def find_cycle(self):
        if len(self.nodes) < 3:
            print("Недостаточно вершин для построения цикла")
            return
        
        if self.mode == 1:
            return self.find_cycle_with_modification()
        else:
            return self.find_cycle_without_modification()

    def draw_cycle (self):

        min_hamiltonian_cycle = self.find_cycle()

        if not min_hamiltonian_cycle:
            print("Невозможно нарисовать цикл: цикл не найден")
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

    def clear_canvas(self):
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()
        self.canvas.delete("all")
        self.canvas_cycle.delete("all")
        self.table.delete(*self.table.get_children())
        self.node_count = 1
        self.edge_labels.clear()

    def distance(self, node1, node2):
        # Используем вес ребра из графа, если он задан
        if self.graph.has_edge(node1, node2):
            return self.graph[node1][node2]['weight']
        else:
            return float('inf')  # Если ребра нет, возвращаем бесконечность

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()
