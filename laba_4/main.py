import numpy as np
import random
import matplotlib.pyplot as plt
from tkinter import ttk
import tkinter as tk
from tkinter.messagebox import showerror, showinfo
from tkinter.simpledialog import askstring
import sys
import time

#сохранение и использование функции
class Func:
    def __init__(self, func: str)->None:
        self.func = func
        
    def value(self, x = 0, y = 0)->float:
        return eval(self.func)

#класс, который хранит параметры об индивиде
class Individual:
    
    def __init__(self, x, y) -> None:
        self.x = x #хросомы
        self.y = y
        self.fitness = 0 #значение целевой функции
        
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
            
    def __repr__(self) -> str:
        return f"{self.x}, {self.y}, {self.fitness}"
    
class Population(list): # класса популяции, наследуется от списка
    def __init__(self, *args):
        super().__init__(*args)
        
    def crossing(self, parents: tuple, func:Func)->Individual: #кроссинговер
        
        parent_1, parent_2 = parents
        child = Individual((parent_1.x + parent_2.x) / 2, (parent_1.y + parent_2.y) / 2) # создается индивид нового поколения со среднеарифметическими параметрами родителей.
        child.calculation_fitness(func)
        
        return child
    
def Tournament(population: Population, POPULATION_SIZE, mod_selection=False) -> Population:
    offspring = Population()
    best_individ = min(population, key=lambda ind: ind.fitness)
    offspring.append((best_individ, best_individ))
    
    for _ in range(1, POPULATION_SIZE):
        if mod_selection:
            # Модифицированный отбор родителей с порогом среднего значения
            avg_fitness = sum(ind.fitness for ind in population) / len(population)
            individs = [ind for ind in population if ind.fitness <= avg_fitness]
            if len(individs) < 2:
                individs = population  # Возвращаемся к полной популяции, если выборка пуста
            chosen = random.sample(individs, 2)
        else:
            # Обычный турнирный отбор
            individs = random.choices(population, k=3)
            chosen = sorted(individs, key=lambda ind: ind.fitness)[:2]
        
        offspring.append((chosen[0].clone(), chosen[1].clone()))
    
    return offspring

def Individual_creator(min_x, min_y, max_x, max_y, func: Func)->Individual: #инициализация индивида для первой популяции
    new_Individual = Individual((max_x - min_x)* random.random() + min_x, (max_y - min_y)* random.random() + min_y)
    new_Individual.calculation_fitness(func)
    return new_Individual
    
def Population_creator(min_x, min_y, max_x, max_y, POPULATION_SIZE, func: Func)->Population: # инициализация начальной популяции
    return Population([Individual_creator(min_x, min_y, max_x, max_y, func) for i in range(POPULATION_SIZE)])

def add_in_pop(pop_ind: list, population: Population): #добавление индивидов в массив всех индивидов всех поколений
    for individ in population:
        pop_ind.append(individ.clone())
    return pop_ind

class Constants: # параметы работы алгоритма
    def __init__(self) -> None:
        self.FUNC = Func("(y - x**2)**2 + 100*(1 - x)**2") 
        # константы генетического алгоритма
        self.POPULATION_SIZE = 100   # количество индивидуумов в популяции
        self.MAX_GENERATIONS = 100
        self.MUTATION_CHANCE = 0.1
        self.DELTA_MUTATION = 0.05
        self.POPULATION = Population_creator(-100, -100, 100, 100, self.POPULATION_SIZE, self.FUNC)
        self.BEST_SOLUTION = 0
        self.BEST_COORDINATE = (0, 0)
        self.populations = list()

def main(const: Constants, use_mod_selection=False) -> Constants:
    start_time = time.time()
    population = Population(const.POPULATION)
    
    N_generation = 0
    max_pop_ind = list()  # Инициализация списков хранения лучшей особи в поколении и всех особей во всех поколениях
    pop_ind = list()
    max_pop_ind.append(min(population, key=lambda ind: ind.fitness))
    pop_ind = add_in_pop(pop_ind, population)
    
    while N_generation < const.MAX_GENERATIONS: 
        N_generation += 1
        offspring = Tournament(population, const.POPULATION_SIZE, mod_selection=use_mod_selection)
        
        for i in range(const.POPULATION_SIZE):
            offspring[i] = population.crossing(offspring[i], const.FUNC)
            
            if random.random() < const.MUTATION_CHANCE:
                offspring[i].mutation(const.DELTA_MUTATION)
                
        population = Population(offspring)
        pop_ind = add_in_pop(pop_ind, population)
        max_pop_ind.append(min(population, key=lambda ind: ind.fitness))
    
    end_time = time.time()
    if use_mod_selection:
        print ("Время работы алгоритма с турнирной модификацией:", end_time - start_time)
    else:
        print ("Время работы алгоритма без турнирной модификации:", end_time - start_time)

    const.POPULATION = population.copy()
    const.populations = pop_ind.copy()
    const.BEST_SOLUTION = round(max_pop_ind[-1].fitness, 3)
    const.BEST_COORDINATE = (round(max_pop_ind[-1].x, 3), round(max_pop_ind[-1].y, 3))
    const.POPULATION = add_in_pop(const.POPULATION, pop_ind)
        
    return const

class GUI:
    def __init__(self) -> None:
        
        self.constants = Constants()
        
        self.root = tk.Tk()
        self.root.title("генетический алгоритм")
        self.root.geometry('480x540')
        self.root['background'] = "white"
        self.root.resizable(False, False)
        
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
        
        self.function_Label = ttk.Label(self.input_Frame, text="функция: ", style="Mini.TLabel")
        self.function_Entry = ttk.Entry(self.input_Frame, justify="center", width=25)
        self.function_Entry.insert('end', "(y-x**2)**2+100*(1-x)**2")
        self.function_Label.grid(column=0, row=1)
        self.function_Entry.grid(column=1, row=1)

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

        self.max_Label = ttk.Label(self.input_Frame, text="максимальное значение гена: ", style="Mini.TLabel")
        self.max_Entry = ttk.Entry(self.input_Frame, justify="center", width=25)
        self.max_Entry.insert('end', "100")
        self.max_Label.grid(column=0, row=4)
        self.max_Entry.grid(column=1, row=4)
        
        self.min_Label = ttk.Label(self.input_Frame, text="минимальное значение гена: ", style="Mini.TLabel")
        self.min_Entry = ttk.Entry(self.input_Frame, justify="center", width=25)
        self.min_Entry.insert('end', "-100")
        self.min_Label.grid(column=0, row=4)
        self.min_Entry.grid(column=1, row=4)
        
        self.size_population_Label = ttk.Label(self.input_Frame, text="размер популяции: ", style="Mini.TLabel")
        self.size_population_Entry = ttk.Entry(self.input_Frame, justify="center", width=25)
        self.size_population_Entry.insert('end', "100")
        self.size_population_Label.grid(column=0, row=5)
        self.size_population_Entry.grid(column=1, row=5)

        self.size_generation_Label = ttk.Label(self.input_Frame, text="количество поколений: ", style="Mini.TLabel")
        self.size_generation_Entry = ttk.Entry(self.input_Frame, justify="center", width=25)
        self.size_generation_Entry.insert('end', "100")
        self.size_generation_Label.grid(column=0, row=6)
        self.size_generation_Entry.grid(column=1, row=6)
        
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
        self.Table.column('#0', width=50)
        self.Table.column('#1', width=110)
        self.Table.column('#2', width=105)
        self.Table.column('#3', width=105)

        self.Scrollbar.config(command=self.Table.yview)
        
        self.Table.grid(row=0, column=0, columnspan=2, sticky='nsew')
        
        self.solution_Frame = ttk.Frame(self.root, style="Style.TFrame")
        
        self.best_solution_Label = ttk.Label(self.solution_Frame, text=f"лучшее решение={self.constants.BEST_SOLUTION} при x={self.constants.BEST_COORDINATE[0]}, y={self.constants.BEST_COORDINATE[1]}", style="TLabel")
        self.best_solution_Label.grid(row=1, column=0)
        
        self.solution_Frame.grid(row=1, column=0)
        
        self.root.config(menu=self.main_menu)
    
    def start(self):
        self.root.mainloop()
    
    def start_algoritm(self):
        use_mod_selection = self.use_mod_selection.get()
        self.constants = main(self.constants, use_mod_selection)
        self.best_solution_Label.config(
            text=f"лучшее решение={self.constants.BEST_SOLUTION} при x={self.constants.BEST_COORDINATE[0]}, y={self.constants.BEST_COORDINATE[1]}"
        ) 

    def safe_paremetres(self):
        self.constants.FUNC = Func(self.function_Entry.get())
        self.constants.MUTATION_CHANCE = float(self.mutation_Entry.get())
        self.constants.DELTA_MUTATION = float(self.delta_Entry.get())
        self.constants.POPULATION_SIZE = int(self.size_population_Entry.get())
        self.constants.MAX_GENERATIONS = int(self.size_generation_Entry.get())
        self.constants.POPULATION = Population_creator(float(self.min_Entry.get()), float(self.min_Entry.get()), float(self.max_Entry.get()), float(self.max_Entry.get()), self.constants.POPULATION_SIZE, self.constants.FUNC)
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
    print(tk.__file__)
    gui = GUI()
    gui.start()
    # const = Constants()
    # main(const)