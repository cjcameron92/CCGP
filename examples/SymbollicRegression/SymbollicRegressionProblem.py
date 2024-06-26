import concurrent.futures
import math
import random
import operator
import numpy as np
from scipy.io import arff
import copy
from sklearn.linear_model import LinearRegression
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
pop_size = 300
num_genes = 4
terminals = ['x']
minInitDepth = 2
maxInitDepth = 5
max_global_depth = 8
max_crossover_growth = 3
max_mutation_growth = 3
mutation_rate = 0.2
crossover_rate = 0.9
num_generations = 50
elitism_size = 1


#new_data_points = [{'x': -0.2}, {'x': 0.5}, {'x': 0.7}]

# function to add nums
def add(*args):
    return sum(args)

# function to divide using protected division (if denominator is 0 return 1)
def protected_division(x, y):
    if y == 0:
        return 1
    return x / y

# define operators
ops = {
        'add': add,
        'sub': operator.sub,
        'mul': operator.mul,
        'div': protected_division,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
}
# define arity
arity = {
        'add': 2,
        'sub': 2,
        'mul': 2,
        'div': 2,
        'sin': 1,
        'cos': 1,
        'tan': 1,
}

# function to determine terminal or ephemeral (-1, 1)
def terminal_or_ephemeral(terminals):
    total_options = len(terminals) + 1
    ephemeral_chance = 1 / total_options
    if random.random() < ephemeral_chance:
        return random.uniform(-1, 1)
    else:
        return random.choice(terminals)
# Function to generate tree based on params
def generate_tree(terminals, max_global_depth, current_depth=0, min_depth=minInitDepth):
    # base case (leaf node)
    if current_depth >= max_global_depth or (current_depth >= min_depth and random.random() < 0.5):
        return Node(terminal_or_ephemeral(terminals))
    else:
        # add more
        op = random.choice(list(ops.keys()))
        ar = arity[op]
        children = [generate_tree(terminals, max_global_depth, current_depth + 1, min_depth) for _ in range(ar)]
        return Node(op, children)
# Node class
class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

    # returns depth as int
    def depth(self):
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    # function to preform the mutation by generating a subtree
    def mutate(self, terminals, max_global_depth, max_mutation_growth):
        if self.depth() >= max_global_depth - max_mutation_growth:
            return self  # Prevent depth exceeding max_global_depth after mutation
        new_subtree = generate_tree(terminals, max_global_depth, min_depth=0, current_depth=self.depth())
        self.value = new_subtree.value
        self.children = new_subtree.children
        return self

    # function to preform crossover, swapping 2 nodes at fixed positions
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

    # function to provide a copy (fix aliasing)
    def copy(self):
        return Node(self.value, [child.copy() for child in self.children])

    # function to return a value
    def evaluate(self, mapping):
        if self.value in ops:
            results = [child.evaluate(mapping) for child in self.children]
            return ops[self.value](*results)
        elif self.value in mapping:
            return mapping[self.value]
        else:
            return self.value

    def __str__(self):
        if self.children:
            return f"{self.value}({', '.join(str(child) for child in self.children)})"
        return str(self.value)
# Main method  / define seed
def main(seed=7246325):
    random.seed(seed)
    np.random.seed(seed)

    # Wrapper for crossover function
    def one_point_crossover(parent1, parent2, max_global_depth, max_crossover_growth):
        gene1 = random.choice(parent1)
        gene1Index = parent1.index(gene1)
        gene2 = random.choice(parent2)
        gene2Index = parent2.index(gene2)
        # select indexes from parents and preform crossover
        crossover_point = random.randint(1, min(gene1.depth(), gene2.depth()))
        new_gene1, new_gene2 = gene1.crossover(gene2, crossover_point, max_global_depth, max_crossover_growth)
        offspring1 = parent1[:gene1Index] + [new_gene1] + parent1[gene1Index + 1:]
        offspring2 = parent2[:gene2Index] + [new_gene2] + parent2[gene2Index + 1:]
        return offspring1, offspring2

    # Wrapper for mutation function
    def mutate(individual, terminals, max_global_depth, max_mutation_growth):
        mutated_individual = []
        for gene in individual:
            mutated_gene = gene.mutate(terminals, max_global_depth, max_mutation_growth)
            mutated_individual.append(mutated_gene)
        return mutated_individual

    # fitness function for SymbolicRegression
    def multi_gene_fitness(individual, data_points):
        # define
        X = np.array([[gene.evaluate(data) for gene in individual] for data in data_points])
        y = np.array([data['actual'] for data in data_points])

        # create and fit model
        model = LinearRegression()
        model.fit(X, y)

        # create predictions
        predictions = model.predict(X)

        # create mse
        mse = np.mean((predictions - y) ** 2)

        return mse, model

    # create an array for initial individual for fixed population size.
    def initialize_population(pop_size, num_genes, terminals, max_depth):
        return [[generate_tree(terminals, max_depth) for _ in range(random.randint(1, num_genes))] for _ in range(pop_size)]

    # tournament selection
    def tournament_selection2(population_with_fit_and_models, tournament_size=3):
        tournament = random.sample(population_with_fit_and_models, tournament_size)
        best_fitness = float('inf')
        best_individual = None
        for individual in tournament:
            fitness = individual[1]
            if fitness < best_fitness:
                best_fitness = fitness
                best_individual = individual
        # return best fit
        return best_individual[0]

    # abstracted function to handle evolution of a population
    def evolve_population(population, terminals, max_depth, mutation_rate, elitism_size, crossover_rate):
        new_population = []

        population_with_fit_and_models = population

        # sort
        sorted_population_with_fit_and_models = sorted(population_with_fit_and_models, key=lambda x: x[1])

        # handle elites
        elites_with_fit_and_models = sorted_population_with_fit_and_models[:elitism_size]
        elites = [copy.deepcopy(elite[0]) for elite in
                  elites_with_fit_and_models]
        new_population.extend(elites)

        # preform crossover and mutation
        while len(new_population) < len(population):
            if random.random() < crossover_rate:
                parent1 = tournament_selection2(population_with_fit_and_models)
                parent2 = tournament_selection2(population_with_fit_and_models)
                offspring1, offspring2 = one_point_crossover(parent1, parent2, max_depth, max_crossover_growth)
                new_population.extend([offspring1, offspring2][:len(population) - len(new_population)])
            else:
                individual = tournament_selection2(population_with_fit_and_models)
                new_population.append(individual)

        for i in range(len(new_population)):
            if random.random() < mutation_rate and i >= elitism_size:
                new_population[i] = mutate(new_population[i], terminals, max_depth,
                                           max_mutation_growth=max_mutation_growth)

        return new_population

    def readData(filepath):
        with open(filepath, 'r') as f:
            dataset, meta = arff.loadarff(f)
        data = np.array(dataset)
        return np.array(
            [[element.decode() if isinstance(element, bytes) else element for element in row] for row in data])

    to_compute = readData('points.arff')[:, 0].tolist()

    def target_function(x):
        return protected_division((math.sin(x) * math.cos(x)), (math.tan(x) + 1)) + (
                    x ** 3 - 2 * x ** 2 + 5 * x - 3) * math.sin(x) * 0.5

    data_points = [{'x': x, 'actual': target_function(x)} for x in to_compute]

    best_fitness_global = float('inf')
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
    # init population
    population = initialize_population(pop_size, num_genes, terminals, maxInitDepth)
    population_with_fit_and_models = []

    for individual in population:
        idv = individual
        fitness, model = multi_gene_fitness(individual,
                                            data_points)  # Ensure this should be multi_gene_fitness_torch if using GPU
        population_with_fit_and_models.append([individual, fitness, model])
        genFitness.append(fitness)
        if fitness < best_fitness_global:
            best_fitness_global = fitness
            best_individual_global = copy.deepcopy(idv)
            best_model_global = model
            best_index = population.index(individual)
    genAvgs.append(np.mean(genFitness))
    genMins.append(np.min(genFitness))
    genMaxs.append(np.max(genFitness))
    genMeds.append(np.median(genFitness))

    print(f"Generation 0: Best Fitness = {best_fitness_global}")
    # loop through all max generations
    for gen in range(num_generations):
        genFitness = []
        # evolve population
        population = evolve_population(population_with_fit_and_models, terminals, max_global_depth, mutation_rate,
                                       elitism_size,
                                       crossover_rate)
        index = 0
        population_with_fit_and_models = []
        for individual in population:
            idv = individual
            fitness, model = multi_gene_fitness(individual,
                                                data_points)  # Ensure this should be multi_gene_fitness_torch if using GPU
            population_with_fit_and_models.append([individual, fitness, model])
            genFitness.append(fitness)
            if fitness < best_fitness_global:
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


if __name__ == '__main__':
    random.seed(7246325)
    seeds = [random.randint(0, 100000000) for _ in range(10)]

    # Run on multithreads!
    with concurrent.futures.ProcessPoolExecutor() as executor:
        all_results = list(executor.map(main, seeds))
    bestIndex = -1
    bestResult = float('inf')
    for result in all_results:
        print(f"Stats (genAvg, genMin, genMax, genMed): {result[0]}")
        print(f"y = {result[2].intercept_}")
        for gene, coef in zip(result[1], result[2].coef_):
            if coef < 0:
                print(f" - {-coef}", end="")
            else:
                print(f" + {coef}", end="")
            print(f" * {gene}")
        if result[3] < bestResult:
            bestResult = result[3]
            bestIndex = all_results.index(result)
    # Plotting the average results
    grandAvg = []
    grandMin = []
    grandMax = []
    grandMed = []
    for gen in range(num_generations+1):
        grandAvg.append(np.mean([result[0][0][gen] for result in all_results]))
        grandMin.append(np.mean([result[0][1][gen] for result in all_results]))
        grandMax.append(np.mean([result[0][2][gen] for result in all_results]))
        grandMed.append(np.mean([result[0][3][gen] for result in all_results]))
    plt.plot(range(0, num_generations + 1), grandAvg, label='Average')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Average Fitness over Generations')
    plt.legend()
    plt.savefig('plots/average_fitness_plot.png')
    plt.close()

    # Plotting the minimum results
    plt.plot(range(0, num_generations + 1), grandMin, label='Minimum')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Minimum Fitness over Generations')
    plt.legend()
    plt.savefig('plots/minimum_fitness_plot.png')
    plt.close()

    # Plotting the median results
    plt.plot(range(0, num_generations + 1), grandMed, label='Median')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Median Fitness over Generations')
    plt.legend()
    plt.savefig('plots/median_fitness_plot.png')
    plt.close()

    print(f"Best Individual Fitness: {all_results[bestIndex][3]}, From run: {bestIndex+1}")
    print(f"Average best fitness: {np.mean([result[3] for result in all_results])}")