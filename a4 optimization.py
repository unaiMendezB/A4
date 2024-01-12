import random
import tsplib95
import matplotlib.pyplot as plt
import os


def fitness(route, graph):
    return -sum(graph[i][j]['weight'] for i, j in zip(route, route[1:] + route[:1]))


# Selection
# Tournament selection
def tournament_selection(population, graph):
    tournament_size = 3
    return max(random.sample(population, tournament_size), key=lambda x: fitness(x, graph))


# Roulette Wheel Selection
def roulette_wheel_selection(population, graph):
    fitness_values = [fitness(individual, graph) for individual in population]
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]
    return random.choices(population, weights=probabilities, k=1)[0]


# Crossovers
# Order crossover (OX1)
def order_crossover(parent, notUsed):
    size = len(parent)
    i, j = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[i:j] = parent[i:j]
    available = set(parent) - set(child)
    for k in list(range(j, size)) + list(range(0, j)):
        if child[k] == -1:
            child[k] = available.pop()
    return child


# Uniform Crossover
def uniform_crossover(parent1, parent2):
    child = []
    for gene1, gene2 in zip(parent1, parent2):
        child.append(gene1 if random.random() < 0.5 else gene2)
    return child


# Mutations
# Swap mutation
def swap_mutate(route):
    i, j = random.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]


# Scramble Mutation
def scramble_mutate(route):
    i, j = sorted(random.sample(range(len(route)), 2))
    segment = route[i:j]
    random.shuffle(segment)
    route[i:j] = segment


# Main function for the calculation of the TSP
def a4_optimization(filepath, num_generations, population_size, mutation_rate, selection_method='tournament',
                    crossover_method='order', mutation_method='swap'):
    problem = tsplib95.load(filepath)
    graph = problem.get_graph()
    num_cities = len(graph.nodes)

    population = []
    for _ in range(population_size):
        individual = list(range(1, num_cities + 1))
        random.shuffle(individual)
        population.append(individual)

    best_fitness_values = []

    # Select appropriate functions based on user input
    selection_function = tournament_selection if selection_method == 'tournament' else roulette_wheel_selection
    crossover_function = order_crossover if crossover_method == 'order' else uniform_crossover
    mutation_function = swap_mutate if mutation_method == 'swap' else scramble_mutate

    for generation in range(num_generations):
        new_population = []
        for _ in range(population_size):
            parent1 = selection_function(population, graph)
            parent2 = selection_function(population, graph)
            child = crossover_function(parent1, parent2)
            if random.random() < mutation_rate:
                mutation_function(child)
            new_population.append(child)

        population = new_population
        best_fitness_values.append(-fitness(max(population, key=lambda x: fitness(x, graph)), graph))

    best_route = max(population, key=lambda x: fitness(x, graph))
    print(f'Best route for file {filepath}:', best_route)

    minimal_tour_length = -fitness(best_route, graph)
    print(f'Minimal tour length for file {filepath}:', minimal_tour_length)

    # Plotting with filename and method information
    plt.plot(best_fitness_values)
    plt.title(f'Evolution of the minimum total traveling distance -> File: {filepath}'
              f'\nSelection: {selection_method}, Crossover: {crossover_method}, Mutation: {mutation_method}')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.show()


# Starting the calls

# Dataset: Dantzig42
a4_optimization('dantzig42.tsp', 5, 30, 0.2,
                selection_method='tournament', crossover_method='order', mutation_method='swap')
a4_optimization('dantzig42.tsp', 5, 20, 0.2,
                selection_method='tournament', crossover_method='order', mutation_method='scramble')
a4_optimization('dantzig42.tsp', 30, 100, 0.2,
                selection_method='tournament', crossover_method='uniform', mutation_method='swap')
a4_optimization('dantzig42.tsp', 30, 100, 0.2,
                selection_method='tournament', crossover_method='uniform', mutation_method='scramble')

a4_optimization('dantzig42.tsp', 5, 25, 0.2,
                selection_method='roulette', crossover_method='order', mutation_method='swap')
a4_optimization('dantzig42.tsp', 5, 25, 0.2,
                selection_method='roulette', crossover_method='order', mutation_method='scramble')
a4_optimization('dantzig42.tsp', 35, 100, 0.2,
                selection_method='roulette', crossover_method='uniform', mutation_method='swap')
a4_optimization('dantzig42.tsp', 35, 100, 0.2,
                selection_method='roulette', crossover_method='uniform', mutation_method='scramble')
