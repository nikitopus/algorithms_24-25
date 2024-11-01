import random
import numpy as np
import config
import matplotlib.pyplot as plt

def function(x1, x2):
    return 100*(x2 - x1 ** 2) ** 2 + (1 - x1) ** 2

def plot_population(population, generation, current_best_fitness, current_best_individual):
    plt.clf()  
    x_values = [individual[0] for individual in population]
    y_values = [individual[1] for individual in population]

    fitness_values = [evaluate_fitness(individual) for individual in population]
    top_10_percent = int(0.1 * len(population))
    top_indices = np.argsort(fitness_values)[:top_10_percent]

    plt.scatter(x_values, y_values, label=f'Generation {generation + 1}', s=1)  
    plt.scatter(np.array(x_values)[top_indices], np.array(y_values)[top_indices], color='red', label='Top 10%', s=5)  

    plt.legend(bbox_to_anchor=(0, 1.08), loc='center')
    plt.text(0.5, 1.1, f"Best Min = {current_best_fitness}",
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.text(0.5, -0.1, f"Best Individual = {current_best_individual}",
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)

    plt.title('Genetic Algorithm Progress')
    plt.xlabel('')
    plt.ylabel('y')
    
    plt.xlim(config.min_range*1.1, config.max_range*1.1)
    plt.ylim(config.min_range*1.1, config.max_range*1.1)
    
    #plt.savefig(f'{generation}.png')
    
    plt.pause(config.delay_between_iteration_sec)

def initialize_population(population_size):
    return [(random.uniform(config.min_range, config.max_range), random.uniform(config.min_range, config.max_range)) for _ in range(population_size)]

def crossover(parent1, parent2, mutation_chance):
    child = (
        random.uniform(min(parent1[0], parent2[0]), max(parent1[0], parent2[0])),
        random.uniform(min(parent1[1], parent2[1]), max(parent1[1], parent2[1]))
    )
    
    if random.random() < mutation_chance:
        child = (
            random.uniform(config.min_range, config.max_range),
            random.uniform(config.min_range, config.max_range)
        )
    
    return child

def evaluate_fitness(individual):
    fitness_value = None
    new_fitness_value = function(individual[0],individual[1])
    if fitness_value:
        if new_fitness_value < fitness_value:
            fitness_value = new_fitness_value
        return new_fitness_value
    else:
        fitness_value = new_fitness_value
        return new_fitness_value

def evaluate_population(population):
    fitness_values = [evaluate_fitness(x) for x in population]
    best_fitness = min(fitness_values)
    best_individual = population[fitness_values.index(best_fitness)]
    return best_fitness, best_individual

def choose_parents(population, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    tournament_fitness = [evaluate_fitness(individual) for individual in tournament]
    winner_index = tournament_fitness.index(min(tournament_fitness))
    return tournament[winner_index]

def genetic_algorithm(population_size, mutation_chance, max_iterations):
    population = initialize_population(population_size)

    for generation in range(max_iterations):
        new_population = []
        for _ in range(population_size):
            parent1 = choose_parents(population)
            parent2 = choose_parents(population)
            
            fitness_parent1 = evaluate_fitness(parent1)
            fitness_parent2 = evaluate_fitness(parent2)
            crossover_chance = max(0, 1 - max(fitness_parent1, fitness_parent2))
            
            if random.random() < crossover_chance:
                new_population.append(crossover(parent1, parent2, mutation_chance))
        
        while len(new_population) < population_size:
            new_population.append(initialize_population(1)[0])
        
        combined_population = population + new_population
        combined_fitness = [evaluate_fitness(x) for x in combined_population]
        
        sorted_combined = [x for _, x in sorted(zip(combined_fitness, combined_population))]
        selected_population = sorted_combined[:population_size // 2]
        
        population = selected_population + initialize_population(population_size // 2)
        
        current_best_fitness, current_best_individual = evaluate_population(population)
        
        plot_population(population, generation, current_best_fitness, current_best_individual)

        current_best_fitness, current_best_individual = evaluate_population(population)

        print(f"Generation {generation}: Best fitness = {current_best_fitness}; Best Individual = {current_best_individual}")

    best_fitness, best_individual = evaluate_population(population)
    print(f"\nMinimum achieved at x = {round(best_individual[0], 4)}; y = {round(best_individual[1], 4)} with f(x, y) = {function(best_individual[0], best_individual[1])}")

    plt.show()  

genetic_algorithm(config.population_size, config.mutation_chance, config.max_iterations)
