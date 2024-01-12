import tsplib95
import random


def fitness(route):
    return -sum(graph[i][j]['weight'] for i, j in zip(route, route[1:] + route[:1]))


def selection_parent(population):
    # Tournament selection
    tournament_size = 3
    return max(random.sample(population, tournament_size), key=fitness)


# Load data from .tsp file
problem = tsplib95.load('fri26.tsp')
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
        parent1 = selection_parent(population)
        parent2 = selection_parent(population)
