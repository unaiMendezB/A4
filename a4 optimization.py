import random
import tsplib95
import matplotlib.pyplot as plt


def fitness(route):
    return -sum(graph[i][j]['weight'] for i, j in zip(route, route[1:] + route[:1]))


# Selection

# Tournament selection
def tournament_selection(population):
    tournament_size = 3
    return max(random.sample(population, tournament_size), key=fitness)


# Roulette Wheel Selection
def roulette_wheel_selection(population):
    fitness_values = [fitness(individual) for individual in population]
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]
    return random.choices(population, weights=probabilities, k=1)[0]


# Crossovers

# Order crossover (OX1)
def order_crossover(parent):
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


# Load data from .tsp file
problem = tsplib95.load('dantzig42.tsp')
graph = problem.get_graph()
num_cities = len(graph.nodes)

# Parameters
num_generations = 10
population_size = 50
crossover_rate = 0.8
mutation_rate = 0.2

# Create the initial population
population = []
for _ in range(population_size):
    individual = list(range(1, num_cities + 1))
    random.shuffle(individual)
    population.append(individual)

best_fitness_values = []
# Main Generic_algorithm loop
for generation in range(num_generations):
    new_population = []
    for _ in range(population_size):
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)
        child = order_crossover(parent1)
        if random.random() < mutation_rate:
            swap_mutate(child)
        new_population.append(child)
    population = new_population
    best_fitness_values.append(-fitness(max(population, key=fitness)))

# Print best route
best_route = max(population, key=fitness)
print('Best route:', best_route)

# Calculate and print minimal tour length
minimal_tour_length = -fitness(best_route)
print('Minimal tour length:', minimal_tour_length)

# Figure with the evolution of the minimum total traveling distance
plt.plot(best_fitness_values)
plt.title('Evolution of the minimum total traveling distance')
plt.xlabel('Generation')
plt.ylabel('Distance')
plt.show()
