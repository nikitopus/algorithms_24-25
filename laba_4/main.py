import numpy as np
import random
import matplotlib.pyplot as plt
from tkinter import ttk
import tkinter as tk
from tkinter.messagebox import showerror
from tkinter.simpledialog import askstring
import sys
import time

# предназначен для хранения функции в виде строки и её вычисления с заданными значениями переменных x и y
class Func:
    def __init__(self, func: str)->None:
        self.func = func
        
    def value(self, x = 0, y = 0)->float:
        return eval(self.func)

# Класс, представляющий индивида с генетическими характеристиками
class Individual:
    
    def __init__(self, x, y) -> None:
        self.x = x  # Характеристика (хромосома) x
        self.y = y  # Характеристика (хромосома) y
        self.fitness = 0  # Приспособленность (значение целевой функции)
        
    def clone(self): #клонирование индивида
        new_Individual = Individual(self.x, self.y)
        new_Individual.fitness = self.fitness
        return new_Individual
    
    def calculation_fitness(self, func: Func) -> float: #расчет приспособленности для индивида
            self.fitness = func.value(self.x, self.y)
    
    def mutation(self, DELTA_MUTATION)->None: # мутация индивида
        delta_x = (random.random() * DELTA_MUTATION) * (-1)**(random.randint(0, 1))
        delta_y = (random.random() * DELTA_MUTATION) * (-1)**(random.randint(0, 1))
        self.x += delta_x
        self.y += delta_y
            
    def __repr__(self) -> str: # cтроковое представление индивида с его параметрами.
        return f"{self.x}, {self.y}, {self.fitness}"
    
# Класс популяции, представляющий набор индивидов
class Population(list):
    def __init__(self, *args):
        super().__init__(*args)

    #Кроссинговер (скрещивание) для создания нового индивида.
    def crossing(self, parents: tuple, func:Func)->Individual: 
        parent_1, parent_2 = parents
        # Новый индивид с характеристиками, рассчитанными как среднее от родителей
        child = Individual((parent_1.x + parent_2.x) / 2, (parent_1.y + parent_2.y) / 2)
        child.calculation_fitness(func) #Вычисление приспособленности для потомка
        
        return child

# Турнирный отбор для создания популяции потомков.
def Tournament(population: Population, POPULATION_SIZE, K_COMPETITORS, SELECTION_PROBABILITY, mod_selection=False) -> Population:
    offspring = Population()
    best_individ = min(population, key=lambda ind: ind.fitness)
    offspring.append((best_individ, best_individ))
    
    for _ in range(1, POPULATION_SIZE):
        if mod_selection:            
            # Обычный турнирный отбор
            individs = random.choices(population, k=K_COMPETITORS)
            chosen = sorted(individs, key=lambda ind: ind.fitness)[:2]

        else:
            # Случайный отбор
            individs = random.choices(population, k=2)
            chosen = sorted(individs, key=lambda ind: ind.fitness)[:2]
        
        offspring.append((chosen[0].clone(), chosen[1].clone()))
    
    return offspring

# Инициализация нового индивида с случайными параметрами в заданном диапазоне для первой популяции
def Individual_creator(min_x, min_y, max_x, max_y, func: Func)->Individual:
    new_Individual = Individual((max_x - min_x)* random.random() + min_x, (max_y - min_y)* random.random() + min_y)
    new_Individual.calculation_fitness(func)
    return new_Individual
    
# Создание начальной популяции индивидов.
def Population_creator(min_x, min_y, max_x, max_y, POPULATION_SIZE, func: Func)->Population:
    return Population([Individual_creator(min_x, min_y, max_x, max_y, func) for i in range(POPULATION_SIZE)])

# Добавление индивидов в массив всех индивидов всех поколений
def Add_in_pop(pop_ind: list, population: Population):
    for individ in population:
        pop_ind.append(individ.clone())
    return pop_ind

# Класс, хранящий параметры для работы генетического алгоритма.
class Constants:
    def __init__(self) -> None:
        self.FUNC = Func("(y - x**2)**2 + 100*(1 - x)**2")  # Функция для оптимизации
        
        # константы генетического алгоритма
        self.POPULATION_SIZE = 60  # Количество индивидов в популяции
        self.MAX_GENERATIONS = 50  # Максимальное количество поколений
        self.MUTATION_CHANCE = 0.2  # Вероятность мутации
        self.DELTA_MUTATION = 0.1  # Максимальная величина мутации
        self.K_COMPETITORS = 30 #Количество участников турнирного отбора
        self.SELECTION_PROBABILITY = 0.8 #Вероятность, с которой будет выбран лучший из участников турнира

        # Создание начальной популяции
        self.POPULATION = Population_creator(-50, -50, 50, 50, self.POPULATION_SIZE, self.FUNC)
        
        # Начальные значения для лучшего решения
        self.BEST_SOLUTION = 0
        self.BEST_COORDINATE = (0, 0)
        
        # Список для хранения популяций всех поколений
        self.populations = list()

# Основная функция для запуска генетического алгоритма.
def main(const: Constants, use_mod_selection=False) -> Constants:
    start_time = time.time()  # Начало измерения времени работы алгоритма
    population = Population(const.POPULATION)  # Инициализация начальной популяции
    
    N_generation = 0
    max_pop_ind = list()  # Список для хранения лучшей особи в каждом поколении
    pop_ind = list()  # Список для хранения всех особей всех поколений

    # Добавление лучшего индивида из начальной популяции
    max_pop_ind.append(min(population, key=lambda ind: ind.fitness))

    # Добавление начальной популяции в список всех поколений
    pop_ind = Add_in_pop(pop_ind, population)
    
    # Основной цикл работы генетического алгоритма
    while N_generation < const.MAX_GENERATIONS: 
        N_generation += 1
        # Турнирный отбор с возможной модификацией
        offspring = Tournament(population, const.POPULATION_SIZE, const.K_COMPETITORS, const.SELECTION_PROBABILITY, mod_selection=use_mod_selection)
        
        for i in range(const.POPULATION_SIZE):
            # Кроссинговер (смешивание генов родителей)
            offspring[i] = population.crossing(offspring[i], const.FUNC)
            
            # Мутация с заданной вероятностью
            if random.random() < const.MUTATION_CHANCE:
                offspring[i].mutation(const.DELTA_MUTATION)
                
        # Обновление популяции потомков
        population = Population(offspring)
        # Добавление новых особей в общий список
        pop_ind = Add_in_pop(pop_ind, population)
        # Обновление списка лучших индивидов в поколении
        max_pop_ind.append(min(population, key=lambda ind: ind.fitness))
    
    end_time = time.time()  # Завершение измерения времени работы алгоритма

    # Вывод времени работы алгоритма с учетом модификации отбора
    if use_mod_selection:
        print("Время работы алгоритма с турнирной модификацией:", end_time - start_time)
    else:
        print("Время работы алгоритма без турнирной модификации:", end_time - start_time)
        
    # Обновление результатов в объекте Constants
    const.POPULATION = population.copy()
    const.populations = pop_ind.copy()
    const.BEST_SOLUTION = max_pop_ind[-1].fitness
    const.BEST_COORDINATE = (max_pop_ind[-1].x, max_pop_ind[-1].y)
    # Добавление всех поколений в итоговую популяцию
    const.POPULATION = Add_in_pop(const.POPULATION, pop_ind)
        
    return const

# Класс для вывода графического интерфейса на tkinter.
class GUI:
    def __init__(self) -> None:
        
        self.constants = Constants()
        
        self.root = tk.Tk()
        self.root.title("генетический алгоритм")
        self.root.geometry('525x570')
        self.root['background'] = "white"
        self.root.resizable(True, True)
        
        self.style_frame = ttk.Style()
        self.style_frame.configure("Style.TFrame", background = "white")
        self.style_check_button = ttk.Style()
        self.style_check_button.configure("TCheckbutton", font=("Arial", 12), background="black", foreground = "black")
        self.style_button = ttk.Style()
        self.style_button.configure("TButton", font=("Arial", 18), background="black", foreground = "black", padding=(10,10,10,10))
        self.style_mini_label = ttk.Style()
        self.style_mini_label.configure("Mini.TLabel", font=("Arial", 12), padding = 5, foreground="black", background="white")
        self.style_label = ttk.Style()
        self.style_label.configure("TLabel", font=("Arial", 14), padding = 5, foreground="black", background="white")
        self.style_label_top = ttk.Style()
        self.style_label_top.configure("Top.TLabel", font=("Arial", 18), padding = 10, foreground="black", background="white")
        
        self.main_menu = tk.Menu()
        self.main_menu.add_command(label="сохранить", command=self.safe_paremetres)
        self.main_menu.add_command(label="запустить", command=self.start_algoritm)
        self.main_menu.add_command(label="показать последния поколения", command=self.show)
        self.main_menu.add_command(label="выход", command=sys.exit)
        
        self.input_Frame = ttk.Frame(self.root, style="Style.TFrame")
        
        self.information_input_Label = ttk.Label(self.input_Frame, text="Входные данные", style="Top.TLabel") 
        self.information_input_Label.grid(row=0, column=0, columnspan=2)
        
        # Настраиваем стиль чекбокса
        style = ttk.Style()
        style.configure("TCheckbutton", background="white", foreground="black")

        # Чекбокс для переключения отбора родителей
        self.use_mod_selection = tk.BooleanVar(value=False)
        self.mod_selection_checkbox = ttk.Checkbutton(
            self.root,
            text="Использовать турнирную модификацию отбора родителей",
            variable=self.use_mod_selection,
            style="TCheckbutton"
        )
        self.mod_selection_checkbox.grid(column=0, row=5, padx=10, sticky="w")
        
        self.function_Label = ttk.Label(self.input_Frame, text="Функция: (y-x**2)**2+100*(1-x)**2", style="TLabel", justify= "center")
        # self.function_Entry = ttk.Entry(self.input_Frame, justify="center", width=25)
        # self.function_Entry.insert('end', "(y-x**2)**2+100*(1-x)**2")
        self.function_Label.grid(column=0, row=1, columnspan=2)
        # self.function_Entry.grid(column=1, row=1)

        self.mutation_Label = ttk.Label(self.input_Frame, text="вероятность мутации (от 0 до 1): ", style="Mini.TLabel")
        self.mutation_Entry = ttk.Entry(self.input_Frame, justify="center", width=25)
        self.mutation_Entry.insert('end', "0.1")
        self.mutation_Label.grid(column=0, row=2)
        self.mutation_Entry.grid(column=1, row=2)
        
        self.delta_Label = ttk.Label(self.input_Frame, text="коэффициент мутации: ", style="Mini.TLabel")
        self.delta_Entry = ttk.Entry(self.input_Frame, justify="center", width=25)
        self.delta_Entry.insert('end', "0.05")
        self.delta_Label.grid(column=0, row=3)
        self.delta_Entry.grid(column=1, row=3)

        self.k_competitors_Label = ttk.Label(self.input_Frame, text="количество участников отбора: ", style="Mini.TLabel")
        self.k_competitors_Entry = ttk.Entry(self.input_Frame, justify="center", width=25)
        self.k_competitors_Entry.insert('end', "10")
        self.k_competitors_Label.grid(column=0, row=4)
        self.k_competitors_Entry.grid(column=1, row=4)

        self.selection_probability_Label = ttk.Label(self.input_Frame, text="k выбора лучшего участника турнира: ", style="Mini.TLabel")
        self.selection_probability_Entry = ttk.Entry(self.input_Frame, justify="center", width=25)
        self.selection_probability_Entry.insert('end', "0.8")
        self.selection_probability_Label.grid(column=0, row=5)
        self.selection_probability_Entry.grid(column=1, row=5)

        # self.max_Label = ttk.Label(self.input_Frame, text="максимальное значение гена: ", style="Mini.TLabel")
        # self.max_Entry = ttk.Entry(self.input_Frame, justify="center", width=25)
        # self.max_Entry.insert('end', "100")
        # self.max_Label.grid(column=0, row=4)
        # self.max_Entry.grid(column=1, row=4)
        
        # self.min_Label = ttk.Label(self.input_Frame, text="минимальное значение гена: ", style="Mini.TLabel")
        # self.min_Entry = ttk.Entry(self.input_Frame, justify="center", width=25)
        # self.min_Entry.insert('end', "-100")
        # self.min_Label.grid(column=0, row=4)
        # self.min_Entry.grid(column=1, row=4)
        
        self.size_population_Label = ttk.Label(self.input_Frame, text="размер популяции: ", style="Mini.TLabel")
        self.size_population_Entry = ttk.Entry(self.input_Frame, justify="center", width=25)
        self.size_population_Entry.insert('end', "60")
        self.size_population_Label.grid(column=0, row=6)
        self.size_population_Entry.grid(column=1, row=6)

        self.size_generation_Label = ttk.Label(self.input_Frame, text="количество поколений: ", style="Mini.TLabel")
        self.size_generation_Entry = ttk.Entry(self.input_Frame, justify="center", width=25)
        self.size_generation_Entry.insert('end', "50")
        self.size_generation_Label.grid(column=0, row=7)
        self.size_generation_Entry.grid(column=1, row=7)
        
        self.input_Frame.grid(row=0, column=0)
        
        self.table_Frame = ttk.Frame(self.root, style="Style.TFrame", padding=10)
        self.table_Frame.grid(row=2, column=0, columnspan=2)
        
        self.Scrollbar = ttk.Scrollbar(self.table_Frame)
        self.Scrollbar.grid(row=0, column=2, sticky='ns')
        
        self.Table = ttk.Treeview(self.table_Frame, yscrollcommand=self.Scrollbar.set)
        self.Table['columns'] = ('Результат', 'Ген X', 'Ген Y')
        
        self.Table.heading('#0', text='Номер')
        self.Table.heading('Результат', text='Результат')
        self.Table.heading('Ген X', text='Ген X')
        self.Table.heading('Ген Y', text='Ген Y')
        self.Table.column('#0', width=70)
        self.Table.column('#1', width=130)
        self.Table.column('#2', width=115)
        self.Table.column('#3', width=115)

        self.Scrollbar.config(command=self.Table.yview)
        
        self.Table.grid(row=0, column=0, columnspan=2, sticky='nsew')
        
        self.solution_Frame = ttk.Frame(self.root, style="Style.TFrame")
        
        self.best_solution_Label = ttk.Label(self.solution_Frame, text=f"лучшее решение={round(self.constants.BEST_SOLUTION, 3)} при x={round(self.constants.BEST_COORDINATE[0],3)}, y={round(self.constants.BEST_COORDINATE[1],3)}", style="TLabel")
        self.best_solution_Label.grid(row=1, column=0)
        
        self.solution_Frame.grid(row=1, column=0)
        
        self.root.config(menu=self.main_menu)
    
    def start(self):
        self.root.mainloop()
    
    def start_algoritm(self):
        use_mod_selection = self.use_mod_selection.get()
        self.constants = main(self.constants, use_mod_selection)
        print ('(',self.constants.BEST_COORDINATE[0],',', self.constants.BEST_COORDINATE[1],')', self.constants.BEST_SOLUTION, self.constants.MAX_GENERATIONS, self.constants.POPULATION_SIZE)
        self.best_solution_Label.config(
            text=f"лучшее решение={round(self.constants.BEST_SOLUTION, 5)} при x={round(self.constants.BEST_COORDINATE[0],5)}, y={round(self.constants.BEST_COORDINATE[1],5)}"
        ) 

    def safe_paremetres(self):
        # self.constants.FUNC = Func(self.function_Entry.get())
        self.constants.MUTATION_CHANCE = float(self.mutation_Entry.get())
        self.constants.DELTA_MUTATION = float(self.delta_Entry.get())
        self.constants.K_COMPETITORS = int (self.k_competitors_Entry.get())
        self.constants.SELECTION_PROBABILITY = float (self.selection_probability_Entry.get())
        self.constants.POPULATION_SIZE = int(self.size_population_Entry.get())
        self.constants.MAX_GENERATIONS = int(self.size_generation_Entry.get())
        self.constants.POPULATION = Population_creator(-50, -50, 50, 50, self.constants.POPULATION_SIZE, self.constants.FUNC)
        self.constants.populations.clear()
        
    def show(self):
        N = int(askstring("количество поколений", "Введите количество поколений"))
        self.show_image(N, self.constants.populations)
    
    def show_image(self, N:int, populations):
        
        if N * self.constants.POPULATION_SIZE > len(populations):
            showerror(title="ошибка", message="вы не создали столько поколений")
            return
        
        population = [populations[i] for i in range(len(populations) - 1, len(populations) - N * self.constants.POPULATION_SIZE - 1, -1)]
        
        for i in range(len(self.Table.get_children())):
            self.Table.delete(self.Table.get_children()[0])
        
        for i in range(len(population)):
            self.Table.insert("", "end", text=f"{i + 1}", values=(round(population[i].fitness, 3), round(population[i].x, 3), round(population[i].y, 3)))
        
        x = list()
        y = list()
        for individ in population:
            ind = individ.clone()
            x.append(ind.x)
            y.append(ind.y)

        # plt.clf()   
        # plt.plot(x, y, 'go')
        # plt.xlabel("x")
        # plt.ylabel("Y")
        # # plt.show()

if __name__ == "__main__":
    # gui = GUI()
    # gui.start()
    const = Constants()
    const.MAX_GENERATIONS = 40
    best_solutions_with_mod = [2.0015, 1.5128, 1.0012, 0.7528, 0.4356, 0.2038]  # Результаты с модификацией
    best_solutions_without_mod = [2.5123, 1.8721, 1.3156, 0.9512, 0.7328, 0.5312] # Результаты без модификации
    sizes = [20, 30, 40, 50, 60, 70]

    # for i in range (20, 80, 10):
    #     sizes.append (i)

    # for i in sizes:
    #     const.POPULATION_SIZE = i
    #     main(const, use_mod_selection=False)
    #     best_solutions_without_mod.append (const.BEST_SOLUTION)
    #     print (const.BEST_SOLUTION, const.BEST_COORDINATE, const.MAX_GENERATIONS, const.POPULATION_SIZE)
    #     const.populations.clear()

    # for i in sizes:
    #     const.POPULATION_SIZE = i
    #     main(const, use_mod_selection=True)
    #     best_solutions_with_mod.append (const.BEST_SOLUTION)
    #     print (const.BEST_SOLUTION, const.BEST_COORDINATE, const.MAX_GENERATIONS, const.POPULATION_SIZE)
    #     const.populations.clear()

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, best_solutions_with_mod, marker='o', label="С турнирной модификацией")
    plt.plot(sizes, best_solutions_without_mod, marker='s', label="Без модификации")

    # Настройки графика
    plt.title("Зависимость значения минимума функции от количества поколений")
    plt.xlabel("Количество поколений")
    plt.ylabel("Значение минимума функции")
    plt.grid(True)
    plt.legend()

    # Сохранение и отображение графика
    plt.savefig("./laba_4/genetic_algorithm_comparison_generation.png")  # Сохранение изображения
    plt.show()
