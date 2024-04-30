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

# function to generate tree based on specified params
def generate_tree(terminals, arity, ops, max_global_depth, min_depth, current_depth=0):
    if current_depth >= max_global_depth or (current_depth >= min_depth and random.random() < 0.5):
        return Node(terminal_or_ephemeral(terminals))
    else:
        op = random.choice(list(ops.keys()))
        ar = arity[op]
        children = [generate_tree(terminals, arity, ops, max_global_depth, current_depth + 1, min_depth) for _ in range(ar)]
        return Node(op, children)


# function wrapper for one point crossover
def one_point_crossover(parent1, parent2, max_global_depth, max_crossover_growth, terminals):
    gene1 = random.choice(parent1)
    gene2 = random.choice(parent2)
    gene1Index = parent1.index(gene1)
    gene2Index = parent2.index(gene2)
    crossover_point = random.randint(1, min(gene1.depth(), gene2.depth()))
    new_gene1, new_gene2 = gene1.crossover(gene2, crossover_point, max_global_depth, max_crossover_growth)
    offspring1 = parent1[:gene1Index] + [new_gene1] + parent1[gene1Index + 1:]
    offspring2 = parent2[:gene2Index] + [new_gene2] + parent2[gene2Index + 1:]
    # returns 2 offspring post crossover
    return offspring1, offspring2


# function wrapper for mutation, each gene gets mutated
def mutate(individual, terminals, arity, ops, max_global_depth, max_mutation_growth):
    mutated_individual = []
    for gene in individual:
        mutated_gene = gene.mutate(terminals, arity, ops, max_global_depth, max_mutation_growth)
        mutated_individual.append(mutated_gene)
    return mutated_individual

# function wrapper for mutation where only a gene and a random node gets mutated.
def mutate2(individual, terminals, max_global_depth, max_mutation_growth):
    gene = random.choice(individual)
    geneIndex = individual.index(gene)
    mutated_gene = gene.mutate(terminals, max_global_depth, max_mutation_growth)
    mutated_individual = individual[:geneIndex] + [mutated_gene] + individual[geneIndex + 1:]
    return mutated_individual

# function to initialize population
def initialize_population(pop_size, num_genes, terminals, arity, ops, min_depth, max_depth):
    return [[generate_tree(terminals, arity, ops, max_depth, min_depth) for _ in range(random.randint(1, num_genes))] for _ in range(pop_size)]

# function to preform tournament selection
def tournament_selection2(population_with_fit_and_models, fitCheck, worstScore, tournament_size=3):
    tournament = random.sample(population_with_fit_and_models, tournament_size)
    best_fitness = worstScore
    best_individual = None
    for individual in tournament:
        fitness = individual[1]
        if fitCheck(fitness, best_fitness):
            best_fitness = fitness
            best_individual = individual
    return best_individual[0]

# abstracted function to evolve population based on params
def evolve_population(population, terminals, arity, ops, max_depth, mutation_rate, elitism_size, crossover_rate, fitness_func, max_crossover_growth, max_mutation_growth, fitCheck, worstScore, fitnessType):
    new_population = []
    population_with_fit_and_models = population

    if fitnessType == "Minimize":
        sorted_population_with_fit_and_models = sorted(population_with_fit_and_models, key=lambda x: x[1])
    else:
        sorted_population_with_fit_and_models = sorted(population_with_fit_and_models, key=lambda x: x[1], reverse=True)
    elites_with_fit_and_models = sorted_population_with_fit_and_models[:elitism_size]
    elites = [copy.deepcopy(elite[0]) for elite in
              elites_with_fit_and_models]
    new_population.extend(elites)

    # preforms crossover
    while len(new_population) < len(population):
        if random.random() < crossover_rate:
            parent1 = tournament_selection2(population_with_fit_and_models, fitCheck, worstScore)
            parent2= tournament_selection2(population_with_fit_and_models, fitCheck, worstScore)
            offspring1, offspring2 = one_point_crossover(parent1, parent2, max_depth, max_crossover_growth, terminals)
            new_population.extend([offspring1, offspring2][:len(population) - len(new_population)])
        else:
            individual = tournament_selection2(population_with_fit_and_models, fitCheck, worstScore)
            new_population.append(individual)

    for i in range(len(new_population)):
        # preform mutation
        if random.random() < mutation_rate and i >= elitism_size:
            new_population[i] = mutate(new_population[i], terminals, arity, ops, max_depth, max_mutation_growth=max_mutation_growth)



    return new_population


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

# function to run the gp that can be imported for ease of use
def runGP(seed, pop_size, num_genes, terminals, arity, ops,
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
    population_with_fit_and_models = []
    for individual in population:
        fitness, model = fitnessFunc(individual, ops, data)  # Ensure this should be multi_gene_fitness_torch if using GPU
        population_with_fit_and_models.append([individual, fitness, model])
        genFitness.append(fitness)
        if fitCheck(fitness, best_fitness_global):
            best_fitness_global = fitness
            best_individual_global = copy.deepcopy(individual)
            best_model_global = model
            best_index = population.index(individual)
    genAvgs.append(np.mean(genFitness))
    genMins.append(np.min(genFitness))
    genMaxs.append(np.max(genFitness))
    genMeds.append(np.median(genFitness))
    print(f"Generation 0: Best Fitness = {best_fitness_global}")
    for gen in range(num_generations):
        genFitness = []
        population = evolve_population(population_with_fit_and_models, terminals, arity, ops, max_global_depth, mutation_rate, elitism_size,
                                       crossover_rate, fitnessFunc, max_crossover_growth, max_mutation_growth, fitCheck, worstScore, fitnessType)
        index = 0
        population_with_fit_and_models = []
        for individual in population:
            idv = individual
            fitness, model = fitnessFunc(individual, ops, data)  # Ensure this should be multi_gene_fitness_torch if using GPU
            population_with_fit_and_models.append([idv, fitness, model])
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
    return [genAvgs, genMins, genMaxs, genMeds], best_individual_global, best_model_global, best_fitness_global