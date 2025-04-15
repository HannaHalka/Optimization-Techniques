import numpy as np


def generate_population():
    return np.random.randint(0, 2, size=(population_size, chromosome_length))


def fitness(population, weights, benefits, capacity):
    total_weights = population @ weights
    total_benefits = population @ benefits
    fitness_scores = np.where(total_weights <= capacity, total_benefits, 0)
    return fitness_scores


def selection(population, fitness_scores):
    sorted_indices = np.argsort(fitness_scores)
    best = population[sorted_indices[-6:]]
    worst_indices = sorted_indices[:6]
    return best, worst_indices


def crossover_and_mutation(parents):
    children = []
    for i in range(0, len(parents), 2):
        p1, p2 = parents[i], parents[(i + 1) % len(parents)]
        mid = chromosome_length // 2
        child1 = np.concatenate((p1[:mid], p2[mid:]))
        child2 = np.concatenate((p2[:mid], p1[mid:]))

        if np.random.rand() < mutation_prob:
            child1 = 1 - child1
        if np.random.rand() < mutation_prob:
            child2 = 1 - child2
        children.extend([child1, child2])
    return np.array(children)


def replace(population, worst_indices, children):
    new_population = population.copy()
    for idx, child in zip(worst_indices, children):
        new_population[idx] = child
    return new_population


def evolution(weights, benefits, capacity):
    population = generate_population()
    prev_best = -1
    stagnation_counter = 0

    for generation in range(num_generations):
        fitness_scores = fitness(population, weights, benefits, capacity)
        best_score = np.max(fitness_scores)

        if abs(best_score - prev_best) < 1e-6:
            stagnation_counter += 1
        else:
            stagnation_counter = 0
        if stagnation_counter >= 20:
            break
        prev_best = best_score

        best, worst_indices = selection(population, fitness_scores)
        children = crossover_and_mutation(best)
        population = replace(population, worst_indices, children)

    final_fitness = fitness(population, weights, benefits, capacity)
    best_index = np.argmax(final_fitness)
    best_chromosome = population[best_index]
    best_value = final_fitness[best_index]

    return best_chromosome, best_value


population_size = 50
chromosome_length = 20
num_generations = 1000
mutation_prob = 0.06

weights = np.array([4, 3, 6, 2, 1, 8, 5, 2,
                    7, 6, 3, 4, 2, 1, 9, 3,
                    2, 1, 4, 6])

benefits = np.array([4, 5, 6, 7, 0, 0, 0, 0,
             7, 6, 5, 4, 0, 0, 0, 0,
             3, 9, 2, 8]).T

capacity = 35

best_chromosome, best_value = evolution(weights, benefits, capacity)

print(f"Best fitness: {best_value}")
print(f"Chromosome: {best_chromosome}")
