import random

import tsplib95


def fitness(route):
    return -sum(graph[i][j]['weight'] for i, j in zip(route, route[1:] + route[:1]))


def selection_parent(population):
    # Tournament selection
    tournament_size = 3
    return max(random.sample(population, tournament_size), key=fitness)


def crossover(parent):
    # Order crossover (OX1)
    size = len(parent)
    i, j = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[i:j] = parent[i:j]
    available = set(parent) - set(child)
    for k in list(range(j, size)) + list(range(0, j)):
        if child[k] == -1:
            child[k] = available.pop()
    return child


def mutate(route):
    # Swap mutation
    i, j = random.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]


# Load data from .tsp file
problem = tsplib95.load('dantzig42.tsp')
graph = problem.get_graph()
num_cities = len(graph.nodes)

# Parameters
num_generations = 100
population_size = 50
crossover_rate = 0.8
mutation_rate = 0.2

# Create the initial population
population = []
for _ in range(population_size):
    individual = list(range(1, num_cities + 1))
    random.shuffle(individual)
    population.append(individual)

# Main Generic_algorithm loop
for generation in range(num_generations):
    new_population = []
    for _ in range(population_size):
        parent = selection_parent(population)
        child = crossover(parent)
        if random.random() < mutation_rate:
            mutate(child)
        new_population.append(child)
    population = new_population

# Print best route
best_route = max(population, key=fitness)
print('Best route:', best_route)

# Calculate and print minimal tour length
minimal_tour_length = -fitness(best_route)
print('Minimal tour length:', minimal_tour_length)