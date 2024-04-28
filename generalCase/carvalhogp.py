import random
import numpy as np
import copy
import time

#########SYSTEM#########
def terminal_or_ephemeral(terminals):
    total_options = len(terminals) + 1
    ephemeral_chance = 1 / total_options
    if random.random() < ephemeral_chance:
        return random.uniform(-1, 1)
    else:
        return random.choice(terminals)


def generate_tree(terminals, arity, ops, max_global_depth, min_depth, current_depth=0):
    if current_depth >= max_global_depth or (current_depth >= min_depth and random.random() < 0.5):
        return Node(terminal_or_ephemeral(terminals))
    else:
        op = random.choice(list(ops.keys()))
        ar = arity[op]
        children = [generate_tree(terminals, arity, ops, max_global_depth, current_depth + 1, min_depth) for _ in range(ar)]
        return Node(op, children)


def one_point_crossover(parent1, parent2, max_global_depth, max_crossover_growth):
    offspring1 = []
    offspring2 = []
    for gene1, gene2 in zip(parent1, parent2):
        crossover_point = random.randint(1, min(gene1.depth(), gene2.depth()))
        new_gene1, new_gene2 = gene1.crossover(gene2, crossover_point, max_global_depth, max_crossover_growth)
        offspring1.append(new_gene1)
        offspring2.append(new_gene2)
    return offspring1, offspring2


def mutate(individual, terminals, arity, ops, max_global_depth, max_mutation_growth):
    mutated_individual = []
    for gene in individual:
        mutated_gene = gene.mutate(terminals, arity, ops, max_global_depth, max_mutation_growth)
        mutated_individual.append(mutated_gene)
    return mutated_individual


def initialize_population(pop_size, num_genes, terminals, arity, ops, min_depth, max_depth):
    return [[generate_tree(terminals, arity, ops, max_depth, min_depth) for _ in range(num_genes)] for _ in range(pop_size)]


def tournament_selection(population, ops, data_points, fitnessFunc, fitCheck, worstScore, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    best_fitness = worstScore
    best_individual = None
    for individual in tournament:
        fitness, _ = fitnessFunc(individual, ops, data_points)
        if fitCheck(fitness, best_fitness):
            best_fitness = fitness
            best_individual = individual
    return best_individual


def evolve_population(population, data_points, terminals, arity, ops, max_depth, mutation_rate, elitism_size, crossover_rate, fitness_func, max_crossover_growth, max_mutation_growth, fitCheck, worstScore, fitnessType):
    new_population = []

    population_with_models = [(ind, fitness_func(ind, ops, data_points)[1]) for ind in population]

    if fitnessType == "Minimize":
        sorted_population_with_models = sorted(population_with_models, key=lambda x: fitness_func(x[0], ops, data_points)[0])
    else:
        sorted_population_with_models = sorted(population_with_models, key=lambda x: fitness_func(x[0], ops, data_points)[0], reverse=True)
    elites_with_models = sorted_population_with_models[:elitism_size]
    elites = [copy.deepcopy(elite[0]) for elite in
              elites_with_models]
    new_population.extend(elites)

    while len(new_population) < len(population):
        if random.random() < crossover_rate:
            parent1 = tournament_selection(population, ops, data_points, fitness_func, fitCheck, worstScore)
            parent2 = tournament_selection(population, ops, data_points, fitness_func, fitCheck, worstScore)
            offspring1, offspring2 = one_point_crossover(parent1, parent2, max_depth, max_crossover_growth)
            new_population.extend([offspring1, offspring2][:len(population) - len(new_population)])
        else:
            individual = tournament_selection(population, ops, data_points, fitness_func, fitCheck, worstScore)
            new_population.append(individual)

    for i in range(len(new_population)):
        if random.random() < mutation_rate and i >= elitism_size:
            new_population[i] = mutate(new_population[i], terminals, arity, ops, max_depth, max_mutation_growth=max_mutation_growth)

    new_population_with_models = [(ind, fitness_func(ind, ops, data_points)[1]) for ind in new_population]
    return [ind[0] for ind in new_population_with_models]


class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

    def depth(self):
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def mutate(self, terminals, arity, ops, max_global_depth, max_mutation_growth):
        if self.depth() >= max_global_depth - max_mutation_growth:
            return self  # Prevent depth exceeding max_global_depth after mutation
        new_subtree = generate_tree(terminals, arity, ops, max_global_depth, min_depth=0, current_depth=self.depth())
        self.value = new_subtree.value
        self.children = new_subtree.children
        return self

    def crossover(self, other, point, max_global_depth, max_crossover_growth, current_depth=1):
        if current_depth >= max_global_depth - max_crossover_growth:
            return self, other  # Prevent depth exceeding max_global_depth after crossover
        if point == current_depth:
            return other.copy(), self.copy()
        else:
            for i in range(min(len(self.children), len(other.children))):
                self.children[i], other.children[i] = self.children[i].crossover(other.children[i], point,
                                                                                 max_global_depth,
                                                                                 max_crossover_growth,
                                                                                 current_depth + 1)
        return self, other

    def copy(self):
        return Node(self.value, [child.copy() for child in self.children])

    def evaluate(self, mapping, ops):
        if self.value in ops:
            results = [child.evaluate(mapping, ops) for child in self.children]
            return ops[self.value](*results)
        elif self.value in mapping:
            return mapping[self.value]
        else:
            return self.value

    def __str__(self):
        if self.children:
            return f"{self.value}({', '.join(str(child) for child in self.children)})"
        return str(self.value)
def lt(a, b):
    return a < b
def gt(a, b):
    return a > b
def main(seed, pop_size, num_genes, terminals, arity, ops,
         fitnessFunc, minInitDepth, maxInitDepth,
         max_global_depth, mutation_rate, max_mutation_growth, elitism_size,
         crossover_rate, max_crossover_growth, num_generations, data, fitnessType):
    if fitnessType != "Minimize" and fitnessType != "Maximize":
        raise ValueError("fitnessType must be either 'Minimize' or 'Maximize'")
    if fitnessType == "Minimize":
        fitCheck = lt
        worstScore = float('inf')
    else:
        fitCheck = gt
        worstScore = float('-inf')
    random.seed(seed)
    np.random.seed(seed)
    best_fitness_global = worstScore
    best_individual_global = None
    best_model_global = None
    best_generation = 0
    best_index = -1
    start_time = time.time()
    genFitness = []
    genAvgs = []
    genMins = []
    genMaxs = []
    genMeds = []
    population = initialize_population(pop_size, num_genes, terminals, arity, ops, minInitDepth, maxInitDepth)
    for individual in population:
        idv = individual
        fitness, model = fitnessFunc(individual, ops, data)  # Ensure this should be multi_gene_fitness_torch if using GPU
        genFitness.append(fitness)
        if fitCheck(fitness, best_fitness_global):
            best_fitness_global = fitness
            best_individual_global = copy.deepcopy(idv)
            best_model_global = model
            best_index = population.index(individual)
    genAvgs.append(np.mean(genFitness))
    genMins.append(np.min(genFitness))
    genMaxs.append(np.max(genFitness))
    genMeds.append(np.median(genFitness))
    print(f"Generation 0: Best Fitness = {best_fitness_global}")
    for gen in range(num_generations):
        genFitness = []
        population = evolve_population(population, data, terminals, arity, ops, max_global_depth, mutation_rate, elitism_size,
                                       crossover_rate, fitnessFunc, max_crossover_growth, max_mutation_growth, fitCheck, worstScore, fitnessType)
        index = 0
        for individual in population:
            idv = individual
            fitness, model = fitnessFunc(individual, ops, data)  # Ensure this should be multi_gene_fitness_torch if using GPU
            genFitness.append(fitness)
            if fitCheck(fitness, best_fitness_global):
                best_fitness_global = fitness
                best_individual_global = copy.deepcopy(idv)
                best_model_global = model
                best_generation = gen + 1
                best_index = index
            index += 1
        genAvgs.append(np.mean(genFitness))
        genMins.append(np.min(genFitness))
        genMaxs.append(np.max(genFitness))
        genMeds.append(np.median(genFitness))
        print(f"Generation {gen + 1}: Best Fitness = {best_fitness_global}")

    end_time = time.time()
    time_elapsed = end_time - start_time

    # Format the elapsed time into a more readable format if desired
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

    print(
        f"Best individual found in {formatted_time} - Generation {best_generation} with Fitness = {best_fitness_global}, index = {best_index}")
    # print(predict_output(best_model_global, best_individual_global, new_data_points))
    return [genAvgs, genMins, genMaxs, genMeds], best_individual_global, best_model_global