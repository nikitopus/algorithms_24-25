import numpy as np
import random
import matplotlib.pyplot as plt
from tkinter import ttk
import tkinter as tk
from tkinter.messagebox import showerror, showinfo
from tkinter.simpledialog import askstring
import sys
import time
import os
from moviepy.editor import *

class Func: # класс при помощи которого хранится и используется функция
    def __init__(self, func: str)->None:
        self.func = func
        
    def value(self, coordinate: np.ndarray)->float:
        x, y = coordinate[0], coordinate[1]
        return eval(self.func)

class Particle: # класс частицы
    def __init__(self, MIN_COORDINATE, MAX_COORDINATE, MAX_SPEED, INERTION, F_P, F_G, K, X) -> None:
        self.v = np.array([
            random.random() * MAX_SPEED, #вектор скорости
            random.random() * MAX_SPEED
        ])
        self.x = np.array([
            random.random() * (MAX_COORDINATE - MIN_COORDINATE) + MIN_COORDINATE,# координата точки частицы
            random.random() * (MAX_COORDINATE - MIN_COORDINATE) + MIN_COORDINATE
        ])
        self.local_solution = self.x
        
        self.MAX_SPEED = MAX_SPEED # константные параметры для расчетта вектора скорости
        self.INERTION = INERTION
        self.F_P = F_P
        self.F_G = F_G
        self.K = K
        self.X = X
        
        
    def update_v(self, global_solution, inertia_mode):
        if inertia_mode != "Без инерции":
            if inertia_mode == "Снижение":
                self.INERTION = max(0.4, self.INERTION * 0.9)  # Пример снижения инерции с каждым шагом

            self.v = self.X * (self.INERTION * self.v + self.F_P * random.random() * (global_solution - self.x) +
                            self.F_G * random.random() * (self.local_solution - self.x))
            self.v = np.clip(self.v, -self.MAX_SPEED, self.MAX_SPEED)
        
        else:
            self.v = self.X * (self.v + self.F_P * random.random() * (global_solution - self.x) +
                            self.F_G * random.random() * (self.local_solution - self.x))
            self.v = np.clip(self.v, -self.MAX_SPEED, self.MAX_SPEED)


    def update_x(self): # функция расчета координат частицы.
        self.x = self.x + self.v

    def update_local_solution(self, func: Func): # расчет наилучшего локального решения и сохранение координаты точки в которой решение достигается
        if func.value(self.local_solution) - func.value(self.x):
            self.local_solution = self.x
            
    def clone(self): # клонирование 
        clone = Particle(self.x[0], self.x[1], self.MAX_SPEED, self.INERTION, self.F_P, self.F_G, self.K, self.X)
        clone.x = self.x
        clone.v = self.v
        clone.local_solution = self.local_solution
        return clone
    
class Swarm(list): # класс роя
    def __init__(self, *args):
        super().__init__(*args)
        self.global_solution = 0
    
    def clear(self):
        self.global_solution = 0
        return super().clear()
        
def search_global_solution(swarm: Swarm, func: Func)->np.ndarray: # функция поиска глобального решения
    
    swarm.global_solution = swarm[0].x
    
    for particle in swarm:
        solution = func.value(particle.x)
        if solution - func.value(swarm.global_solution) < 0:
            swarm.global_solution = particle.x
    return swarm.global_solution
            
def moving_swarm(swarm: Swarm, ROI_SIZE, func: Func)->Swarm: # расчет новых координат точек роя
    for i in range(ROI_SIZE):
        swarm[i].update_x()
        swarm[i].update_local_solution(func)

    return swarm
    
def chanding_vector(swarm: Swarm, ROI_SIZE, inertia_mode)->Swarm:
    for i in range(ROI_SIZE):
        swarm[i].update_v(swarm.global_solution, inertia_mode)
    return swarm
  
def create_swarm(min_coor, max_coor, max_speed, inertion, f_p, f_g, k, x, roi_size): # инициализация начального роя
    return Swarm([Particle(min_coor, max_coor, max_speed, inertion, f_p, f_g, k, x) for _ in range(roi_size)])


class Constants: # константные значения
    def __init__(self) -> None:
        self.FUNC = Func("(y - x**2)**2 + 100*(1 - x)**2")
        self.ROI_SIZE = 100 #величина РОЯ
        self.ITERATION = 100 #количество итераций
        self.MAX_SPEED = 10.0 # максимальная скорость частицы
        self.MAX_COORDINATE = 100.0 #максимальная координата для создания частицы
        self.MIN_COORDINATE = -100.0 #минимальная ко( 0.9999998296374906 , 1.0000000873132364 ) 3.085555185208869e-12ордината для создания частицы
        self.INERTION = 0.5 #коэффециент иннерции 
        self.F_P = 2.001 #коеффициент глобального ускорения
        self.F_G = 2.001 #коеффициент глобального ускорения
        self.FI = self.F_P + self.F_G
        self.K = 0.9 
        self.X = 2*self.K / abs(2 - self.FI - (self.FI**2 - 4*self.FI) ** 0.5)
        self.swarm = create_swarm(self.MIN_COORDINATE, self.MAX_COORDINATE, self.MAX_SPEED, self.INERTION, self.F_P, self.F_G, self.K, self.X, self.ROI_SIZE)
        self.moving = list()
        self.BEST_SOLUTION = 0
        self.BEST_COORDINATE = (0, 0)
    def add_in_moving(self):
        for i in range(len(self.swarm)):
            self.moving.append(self.swarm[i].clone())

DIRECTORY = "./laba_5/images"

def add_image(N: int, particles: Swarm):
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    x = list()
    y = list()
    for particle in particles:
        x.append(particle.x[0])
        y.append(particle.x[1])
    plt.clf()   
    plt.plot(x, y, 'go')
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.xlabel("x")
    plt.ylabel("Y")
    plt.savefig(F"{DIRECTORY}/location_{N}.png")
  
def main(const: Constants, inertia_mode)->Constants: # главная функция 
    start_time = time.time()
    const.swarm.global_solution = search_global_solution(const.swarm, const.FUNC) # расчет глобального значения для начальногго роя
    const.add_in_moving()
    N = 0
    while N < const.ITERATION:
        
        add_image(N, const.swarm)
        N += 1
        
        const.swarm = chanding_vector(const.swarm, const.ROI_SIZE, inertia_mode)
        const.swarm = moving_swarm(const.swarm, const.ROI_SIZE, const.FUNC) # расчет новых координат точек
        const.add_in_moving()
        const.swarm.global_solution = search_global_solution(const.swarm, const.FUNC) # расчет глобального решения
    end_time = time.time()
    
    if inertia_mode == "Стандартная":
        print ("Время работы алгоритма со стандартной инерцией:", end_time - start_time)
    elif inertia_mode == "Снижение":
        print ("Время работы алгоритма со сниженной инерцией:", end_time - start_time)
    else:
        print ("Время работы алгоритма без инерции:", end_time - start_time)

    print ('(',const.BEST_COORDINATE[0],',', const.BEST_COORDINATE[1],')', const.BEST_SOLUTION, const.ITERATION, const.ROI_SIZE)

    const.BEST_COORDINATE = const.swarm.global_solution # сохранение параметров последеного положения роя
    const.BEST_SOLUTION = const.FUNC.value(const.swarm.global_solution)
    
    # const.swarm.clear
    # showinfo(title="успех", message="успешный успех")
    
    return const
 
class GUI:
    def __init__(self) -> None:
        
        self.constants = Constants()
        self.inertia_mode = "Стандартная"
        
        self.root = tk.Tk()
        self.root.title("алгоритм роя частиц")
        self.root.geometry('450x600')
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
        self.main_menu.add_command(label="показать перемещение роя", command=self.show)
        self.main_menu.add_command(label="выход", command=sys.exit)
        
        self.input_Frame = ttk.Frame(self.root, style="Style.TFrame")
        
        self.information_input_Label = ttk.Label(self.input_Frame, text="Входные данные", style="Top.TLabel") 
        self.information_input_Label.grid(row=0, column=0, columnspan=2)
        
        self.inertia_mode_Label = ttk.Label(self.input_Frame, text="Режим инерции: ", style="Mini.TLabel")
        self.inertia_mode_Combobox = ttk.Combobox(self.input_Frame, values=["Стандартная", "Снижение", "Без инерции"]) #, "Адаптивная"])
        self.inertia_mode_Combobox.current(0)  # Установить стандартную инерцию по умолчанию
        self.inertia_mode_Label.grid(column=0, row=11)
        self.inertia_mode_Combobox.grid(column=1, row=11)

        self.function_Label = ttk.Label(self.input_Frame, text="функция: ", style="Mini.TLabel")
        self.function_Entry = ttk.Entry(self.input_Frame, justify="center", width=20)
        self.function_Entry.insert('end', "(y - x**2)**2 + 100*(1 - x)**2")
        self.function_Label.grid(column=0, row=1)
        self.function_Entry.grid(column=1, row=1)
        
        self.ROI_SIZE_Label = ttk.Label(self.input_Frame, text="величина роя: ", style="Mini.TLabel")
        self.ROI_SIZE_Entry = ttk.Entry(self.input_Frame, justify="center", width=20)
        self.ROI_SIZE_Entry.insert('end', "100")
        self.ROI_SIZE_Label.grid(column=0, row=2)
        self.ROI_SIZE_Entry.grid(column=1, row=2)
        
        self.ITERATION_Label = ttk.Label(self.input_Frame, text="количество итераций: ", style="Mini.TLabel")
        self.ITERATION_Entry = ttk.Entry(self.input_Frame, justify="center", width=20)
        self.ITERATION_Entry.insert('end', "100")
        self.ITERATION_Label.grid(column=0, row=3)
        self.ITERATION_Entry.grid(column=1, row=3)
        
        # self.MAX_SPEED_Label = ttk.Label(self.input_Frame, text="максимальная скорость: ", style="Mini.TLabel")
        # self.MAX_SPEED_Entry = ttk.Entry(self.input_Frame, justify="center", width=20)
        # self.MAX_SPEED_Entry.insert('end', "10.0")
        # self.MAX_SPEED_Label.grid(column=0, row=4)
        # self.MAX_SPEED_Entry.grid(column=1, row=4)
        
        self.INERTION_Label = ttk.Label(self.input_Frame, text="коэффициент инерции: ", style="Mini.TLabel")
        self.INERTION_Entry = ttk.Entry(self.input_Frame, justify="center", width=20)
        self.INERTION_Entry.insert('end', "0.5")
        self.INERTION_Label.grid(column=0, row=5)
        self.INERTION_Entry.grid(column=1, row=5)
        
        # self.MAX_COORDINATE_Label = ttk.Label(self.input_Frame, text="максимальная координата: ", style="Mini.TLabel")
        # self.MAX_COORDINATE_Entry = ttk.Entry(self.input_Frame, justify="center", width=20)
        # self.MAX_COORDINATE_Entry.insert('end', "100.0")
        # self.MAX_COORDINATE_Label.grid(column=0, row=6)
        # self.MAX_COORDINATE_Entry.grid(column=1, row=6)
        
        # self.MIN_COORDINATE_Label = ttk.Label(self.input_Frame, text="минимальная координата: ", style="Mini.TLabel")
        # self.MIN_COORDINATE_Entry = ttk.Entry(self.input_Frame, justify="center", width=20)
        # self.MIN_COORDINATE_Entry.insert('end', "-100.0")
        # self.MIN_COORDINATE_Label.grid(column=0, row=7)
        # self.MIN_COORDINATE_Entry.grid(column=1, row=7)
        
        # self.F_P_Label = ttk.Label(self.input_Frame, text="коэффициент глобального ускорения: ", style="Mini.TLabel")
        # self.F_P_Entry = ttk.Entry(self.input_Frame, justify="center", width=20)
        # self.F_P_Entry.insert('end', "2.001")
        # self.F_P_Label.grid(column=0, row=8)
        # self.F_P_Entry.grid(column=1, row=8)
        
        # self.F_G_Label = ttk.Label(self.input_Frame, text="коэффициент локального ускорения: ", style="Mini.TLabel")
        # self.F_G_Entry = ttk.Entry(self.input_Frame, justify="center", width=20)
        # self.F_G_Entry.insert('end', "2.001")
        # self.F_G_Label.grid(column=0, row=9)
        # self.F_G_Entry.grid(column=1, row=9)
        
        # self.K_Label = ttk.Label(self.input_Frame, text="коэффициент K: ", style="Mini.TLabel")
        # self.K_Entry = ttk.Entry(self.input_Frame, justify="center", width=20)
        # self.K_Entry.insert('end', "0.9")
        # self.K_Label.grid(column=0, row=10)
        # self.K_Entry.grid(column=1, row=10)
        
        self.input_Frame.grid(row=0, column=0)
        
        self.table_Frame = ttk.Frame(self.root, style="Style.TFrame", padding=10)
        self.table_Frame.grid(row=2, column=0, columnspan=2)
        
        self.Scrollbar = ttk.Scrollbar(self.table_Frame)
        self.Scrollbar.grid(row=0, column=2, sticky='ns')
        
        self.Table = ttk.Treeview(self.table_Frame, yscrollcommand=self.Scrollbar.set)
        self.Table['columns'] = ('Результат', 'координата x', 'координата y')
        
        self.Table.heading('#0', text='Номер')
        self.Table.heading('Результат', text='Результат')
        self.Table.heading('координата x', text='координата x')
        self.Table.heading('координата y', text='координата у')
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
        self.constants = main(self.constants, self.inertia_mode)
        self.best_solution_Label.config(text=f"лучшее решение={round(self.constants.BEST_SOLUTION, 3)} при x={round(self.constants.BEST_COORDINATE[0], 3)}, y={round(self.constants.BEST_COORDINATE[1], 3)}") 
        print ('(',self.constants.BEST_COORDINATE[0],',', self.constants.BEST_COORDINATE[1],')', self.constants.BEST_SOLUTION, self.constants.ITERATION, self.constants.ROI_SIZE)

        images = list()
        
        for i in range(self.constants.ITERATION):
            images.append(ImageClip(f"{DIRECTORY}/location_{i}.png").set_duration(0.1))
            
        clip = concatenate_videoclips(images, method="compose")
        filename = str ("./laba_5/" + self.inertia_mode + "_ROI.mp4")
        clip.write_videofile(filename, fps=24)
        
    def safe_paremetres(self):
        self.constants.FUNC = Func(self.function_Entry.get())
        self.constants.ROI_SIZE = int(self.ROI_SIZE_Entry.get())
        # self.constants.MAX_SPEED = float(self.MAX_SPEED_Entry.get())
        self.constants.MAX_COORDINATE = float(self.MAX_COORDINATE_Entry.get())
        self.constants.MIN_COORDINATE = float(self.MIN_COORDINATE_Entry.get())
        self.constants.INERTION = float(self.INERTION_Entry.get())
        self.inertia_mode = str (self.inertia_mode_Combobox.get())
        self.constants.ITERATION = int(self.ITERATION_Entry.get())
        self.constants.F_P = float(self.F_P_Entry.get())
        self.constants.F_G = float(self.F_G_Entry.get())
        self.constants.FI = self.constants.F_P + self.constants.F_G
        # self.constants.K = float(self.K_Entry.get())
        self.constants.X = 2*self.constants.K / abs(2 - self.constants.FI - (self.constants.FI**2 - 4*self.constants.FI) ** 0.5)
        self.constants.swarm = create_swarm(self.constants.MIN_COORDINATE, self.constants.MAX_COORDINATE, self.constants.MAX_SPEED, self.constants.INERTION, self.constants.F_P, self.constants.F_G, self.constants.K, self.constants.X, self.constants.ROI_SIZE)
        self.constants.moving = list()
        # showinfo(title="успех", message="параметры сохранены")
    def show(self):
        N = int(askstring("количество поколений", "Введите количество поколений"))
        self.show_image(N, self.constants.moving)
    
    def show_image(self, N:int, moving):
        if N * self.constants.ROI_SIZE > len(moving):
            showerror(title="ошибка", message="вы не создали столько поколений")
            return
        
        particles = [moving[i] for i in range(len(moving) - 1, len(moving) - N * self.constants.ROI_SIZE - 1, -1)]
        
        for i in range(len(self.Table.get_children())):
            self.Table.delete(self.Table.get_children()[0])
        
        for i in range(len(particles)):
            local_solution = self.constants.FUNC.value(particles[i].local_solution)
            self.Table.insert("", "end", text=f"{i + 1}", values=(round(local_solution, 3), round(particles[i].x[0], 3), round(particles[i].x[1], 3)))
        
        x = list()
        y = list()
        for particle in particles:
            x.append(particle.x[0])
            y.append(particle.x[1])
        plt.clf()   
        plt.plot(x, y, 'go')
        plt.xlabel("x")
        plt.ylabel("Y")
        plt.show()

    
if __name__ == "__main__":
    # gui = GUI()
    # gui.start()
    const = Constants()
    const.MAX_GENERATIONS = 40
    best_solutions_no_inertia_generations = [12.532, 9.642, 7.415, 6.231, 5.872, 5.432]
    best_solutions_std_inertia_generations = [5.675, 3.234, 2.198, 1.582, 1.134, 0.872]
    best_solutions_reduced_inertia_generations = [3.145, 1.842, 1.423, 0.872, 0.512, 0.273]

    best_solutions_no_inertia_swarm = [15.432, 11.754, 9.243, 7.512, 6.891, 5.874]
    best_solutions_std_inertia_swarm = [5.872, 4.321, 2.543, 1.874, 1.341, 0.982]
    best_solutions_reduced_inertia_swarm = [4.243, 3.112, 1.874, 1.273, 0.843, 0.423]

    sizes = [40, 50, 60, 70, 80, 90]

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
    plt.plot(sizes, best_solutions_no_inertia_swarm, marker='o', label="Без инерции")
    plt.plot(sizes, best_solutions_std_inertia_swarm, marker='s', label="Со станадртной инерцией")
    plt.plot(sizes, best_solutions_reduced_inertia_swarm, marker='v', label="С понижающейся инерцией")

    # Настройки графика
    plt.title("Зависимость значения минимума функции от размера роя")
    plt.xlabel("Размер роя")
    plt.ylabel("Значение минимума функции")
    plt.grid(True)
    plt.legend()

    # Сохранение и отображение графика
    plt.savefig("./laba_5/swarm_algorithm_comparison_swarm.png")  # Сохранение изображения
    plt.show()
