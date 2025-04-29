import json
import random

# Параметры графа
num_nodes = 20  # Количество вершин
num_edges = 300
canvas_width = 600  # Ширина Canvas
canvas_height = 400  # Высота Canvas
max_weight = 10  # Максимальный вес ребра

# Генерация случайных вершин
nodes = []
for i in range(num_nodes):
    x = random.randint(50, canvas_width - 50)  # Случайные координаты X
    y = random.randint(50, canvas_height - 50)  # Случайные координаты Y
    nodes.append({"x": x, "y": y})

# Генерация случайных ребер
edges = []
for _ in range(num_edges):  # Количество ребер
    from_node = random.choice(nodes)  # Случайная начальная вершина
    to_node = random.choice(nodes)  # Случайная конечная вершина
    if from_node != to_node:  # Убедимся, что это не петля
        # Проверяем, что такого ребра ещё нет
        if not any(edge["from"] == from_node and edge["to"] == to_node for edge in edges):
            weight = random.randint(1, max_weight)
            weight = float (weight)  # Случайный вес
            edges.append({"from": from_node, "to": to_node, "weight": weight})

# Создаем структуру данных для JSON
graph_data = {
    "nodes": nodes,
    "edges": edges
}

name = "graph_" + str(num_nodes) + "_nodes.json"

# Сохраняем в JSON-файл
with open(name, "w") as f:
    json.dump(graph_data, f, indent=2)

print("Файл " + name + " успешно создан.")