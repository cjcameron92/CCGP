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
pop_size = 400
num_genes = 5
terminals = ['Area','Perimeter','Major_Axis_Length','Minor_Axis_Length','Eccentricity','Convex_Area','Extent']
minInitDepth = 2
maxInitDepth = 3
max_global_depth = 4
max_crossover_growth = 2
max_mutation_growth = 2
mutation_rate = 0.3
crossover_rate = 0.9
num_generations = 50
elitism_size = 1

def add(*args):
    return sum(args)

def protected_division(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def protectedLog(x):
    if x <= 0:
        return 0
    else:
        return math.log(x)

ops = {
    'add': add,
    'sub': operator.sub,
    'mul': operator.mul,
    'div': protected_division,
    'sin': math.sin,
    'cos': math.cos,
    'log': protectedLog,
}
arity = {
    'add': 2,
    'sub': 2,
    'mul': 2,
    'div': 2,
    'sin': 1,
    'cos': 1,
    'log': 1,
}

def terminal_or_ephemeral(terminals):
    total_options = len(terminals) + 1
    ephemeral_chance = 1 / total_options
    if random.random() < ephemeral_chance:
        return random.uniform(-1, 1)
    else:
        return random.choice(terminals)
def generate_tree(terminals, max_global_depth, current_depth=0, min_depth=minInitDepth):
    if current_depth >= max_global_depth or (current_depth >= min_depth and random.random() < 0.5):
        return Node(terminal_or_ephemeral(terminals))
    else:
        op = random.choice(list(ops.keys()))
        ar = arity[op]
        children = [generate_tree(terminals, max_global_depth, current_depth + 1, min_depth) for _ in range(ar)]
        return Node(op, children)
class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

    def depth(self):
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def mutate(self, terminals, max_global_depth, max_mutation_growth):
        if self.depth() >= max_global_depth - max_mutation_growth:
            return self  # Prevent depth exceeding max_global_depth after mutation
        new_subtree = generate_tree(terminals, max_global_depth, min_depth=0, current_depth=self.depth())
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
def main(seed=7246325):
    random.seed(seed)
    np.random.seed(seed)

    def one_point_crossover(parent1, parent2, max_global_depth, max_crossover_growth):
        offspring1 = []
        offspring2 = []
        for gene1, gene2 in zip(parent1, parent2):
            crossover_point = random.randint(1, min(gene1.depth(), gene2.depth()))
            new_gene1, new_gene2 = gene1.crossover(gene2, crossover_point, max_global_depth, max_crossover_growth)
            offspring1.append(new_gene1)
            offspring2.append(new_gene2)
        return offspring1, offspring2

    def mutate(individual, terminals, max_global_depth, max_mutation_growth):
        mutated_individual = []
        for gene in individual:
            mutated_gene = gene.mutate(terminals, max_global_depth, max_mutation_growth)
            mutated_individual.append(mutated_gene)
        return mutated_individual

    def multi_gene_fitness(individual, data_points):
        misses = 0
        X = []
        Y = []
        for data in data_points:
            X.append([gene.evaluate(data) for gene in individual])
            Y.append(data['Class'])
        X = np.array(X)
        Y = np.array(Y)
        model = LinearRegression()
        model.fit(X, Y)
        for i in range(len(X)):
            predictedY = model.predict([X[i]])
            if predictedY < 0.5:
                if Y[i] == 1:
                    misses += 1
            else:
                if Y[i] == 0:
                    misses += 1
        return misses, model

    def initialize_population(pop_size, num_genes, terminals, max_depth):
        return [[generate_tree(terminals, max_depth) for _ in range(num_genes)] for _ in range(pop_size)]

    def tournament_selection(population, data_points, tournament_size=3):
        tournament = random.sample(population, tournament_size)
        best_fitness = float('inf')
        best_individual = None
        for individual in tournament:
            fitness, _ = multi_gene_fitness(individual, data_points)
            if fitness < best_fitness:
                best_fitness = fitness
                best_individual = individual
        return best_individual

    def tournament_selection2(population_with_fit_and_models, tournament_size=3):
        tournament = random.sample(population_with_fit_and_models, tournament_size)
        best_fitness = float('inf')
        best_individual = None
        for individual in tournament:
            fitness = individual[1]
            if fitness < best_fitness:
                best_fitness = fitness
                best_individual = individual
        return best_individual[0]

    def evolve_population(population, terminals, max_depth, mutation_rate, elitism_size, crossover_rate):
        new_population = []
        population_with_fit_and_models = population

        sorted_population_with_fit_and_models = sorted(population_with_fit_and_models, key=lambda x: x[1])

        elites_with_fit_and_models = sorted_population_with_fit_and_models[:elitism_size]
        elites = [copy.deepcopy(elite[0]) for elite in
                  elites_with_fit_and_models]
        new_population.extend(elites)

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
            data, meta = arff.loadarff(f)
        # Convert the 'data' attribute to a NumPy array and decode byte strings
        return np.array([[element.decode() if isinstance(element, bytes) else element for element in row] for row in np.array(data)])

    fullData = readData("Rice_Cammeo_Osmancik.arff").tolist()
    random.shuffle(fullData)
    #iterate through column 7 and change the class to 0 or 1
    for row in fullData:
        if row[7] == 'Cammeo':
            row[7] = 0
        else:
            row[7] = 1
    full_data_points = [{'Area': float(row[0]), 'Perimeter': float(row[1]), 'Major_Axis_Length': float(row[2]), 'Minor_Axis_Length': float(row[3]), 'Eccentricity': float(row[4]), 'Convex_Area': float(row[5]), 'Extent': float(row[6]), 'Class': float(row[7])} for row in fullData]
    #first450 of full is training
    training_data_points = full_data_points[:450]
    testing_data_points = full_data_points[450:]


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
    population = initialize_population(pop_size, num_genes, terminals, maxInitDepth)
    population_with_fit_and_models = []
    for individual in population:
        fitness, model = multi_gene_fitness(individual,
                                            training_data_points)  # Ensure this should be multi_gene_fitness_torch if using GPU
        population_with_fit_and_models.append([individual, fitness, model])
        genFitness.append(fitness)
        if fitness < best_fitness_global:
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
        population = evolve_population(population_with_fit_and_models, terminals, max_global_depth, mutation_rate, elitism_size,
                                       crossover_rate)
        index = 0
        population_with_fit_and_models = []
        for individual in population:
            idv = individual
            fitness, model = multi_gene_fitness(individual,
                                                training_data_points)  # Ensure this should be multi_gene_fitness_torch if using GPU
            population_with_fit_and_models.append([idv, fitness, model])
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
    return [genAvgs, genMins, genMaxs, genMeds], best_individual_global, best_model_global, testing_data_points, best_fitness_global, training_data_points


if __name__ == '__main__':
    random.seed(7246325)
    seeds = [random.randint(0, 100000000) for _ in range(10)]
    # Create a ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use map to execute main() function for each seed and maintain order
        all_results = list(executor.map(main, seeds))
    # Process or utilize the collected results as needed
    #get index of best individual
    best_fitness_index = [result[4] for result in all_results].index(min([result[4] for result in all_results]))
    accuracies = []
    for i in range(len(all_results)):
        print(f"Results for run {i+1}:")
        print(f"Stats (genAvg, genMin, genMax, genMed): {all_results[i][0]}")
        print(f"prediction = {all_results[i][2].intercept_}")
        for gene, coef in zip(all_results[i][1], all_results[i][2].coef_):
            print(f" - {-coef} * {gene}") if coef < 0 else print(f" + {coef} * {gene}")
        print(f"\nRun {i+1} performance on testing data:")
        misses = 0
        trueCammeos = 0
        trueOsmanciks = 0
        falseCammeos = 0
        falseOsmanciks = 0
        for data in all_results[i][3]:#cammeo = 0 osmancik = 1
            prediction = all_results[i][2].predict([[gene.evaluate(data) for gene in all_results[i][1]]])
            if prediction < 0.5:
                if data['Class'] == 1:
                    misses += 1
                    falseCammeos += 1
                else:
                    trueCammeos += 1
            else:
                if data['Class'] == 0:
                    misses += 1
                    falseOsmanciks += 1
                else:
                    trueOsmanciks += 1

        print(f"Misses: {misses}, Accuracy: {(1 - (misses / len(all_results[i][3])))*100}%")
        print(f"True Cammeos: {trueCammeos}, False Cammeos: {falseCammeos} True Osmanciks: {trueOsmanciks}, False Osmanciks: {falseOsmanciks}")
        accuracies.append((1 - (misses / len(all_results[i][3])))*100)
        print(f"Run {i+1} performance on training data:")
        misses = 0
        for data in all_results[i][5]:
            prediction = all_results[i][2].predict([[gene.evaluate(data) for gene in all_results[i][1]]])
            if prediction < 0.5:
                if data['Class'] == 1:
                    misses += 1
            else:
                if data['Class'] == 0:
                    misses += 1
        print(f"Misses: {misses}, Accuracy: {(1 - (misses / len(all_results[i][5])))*100}%")
    print(f"Average accuracy: {np.mean(accuracies)}%")
    print(f"Highest accuracy: {np.max(accuracies)}%, on run {np.argmax(accuracies) + 1}")
    grandAvg = []
    grandMin = []
    grandMax = []
    grandMed = []
    for gen in range(num_generations+1):
        grandAvg.append(np.mean([result[0][0][gen] for result in all_results]))
        grandMin.append(np.mean([result[0][1][gen] for result in all_results]))
        grandMax.append(np.mean([result[0][2][gen] for result in all_results]))
        grandMed.append(np.mean([result[0][3][gen] for result in all_results]))
    print("Best individual:")
    print(f"prediction = {all_results[np.argmax(accuracies)][2].intercept_}")
    for gene, coef in zip(all_results[np.argmax(accuracies)][1], all_results[np.argmax(accuracies)][2].coef_):
        print(f" - {-coef} * {gene}") if coef < 0 else print(f" + {coef} * {gene}")
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
