import tsplib95
import random

problem = tsplib95.load('fri26.tsp')

# Get the list of nodes (cities)
nodes = list(problem.get_nodes())

# Calculate the distance matrix
distance_matrix = {}
for node1 in nodes:
    distance_matrix[node1] = {}
    for node2 in nodes:
        distance_matrix[node1][node2] = problem.get_weight(node1, node2)

num_cities = len(distance_matrix)

population_size = 50

population = []
for _ in range(population_size):
    individual = list(range(1, num_cities + 1))
    random.shuffle(individual)
    population.append(individual)
