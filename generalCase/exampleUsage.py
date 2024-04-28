#Necessary imports
import carvalhogp
from sklearn.linear_model import LinearRegression
import numpy as np
import random
import math
import operator
import concurrent.futures
import matplotlib.pyplot as plt
from scipy.io import arff
#User Defined Functions
def add(*args):
    return sum(args)
def protected_division(x, y):
    if y == 0:
        return 1
    return x / y
def call_main(args):
    return carvalhogp.runGP(*args)
def multi_gene_fitness(individual, ops, data_points):
    X = np.array([[gene.evaluate(data, ops) for gene in individual] for data in data_points])
    y = np.array([data['actual'] for data in data_points])

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    mse = np.mean((predictions - y) ** 2)

    return mse, model

if __name__ == '__main__':
    #User Defined Variables
    random.seed(7246325)
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
    fitnessType = "Minimize"

    #User Defined Language
    ops = {
        'add': add,
        'sub': operator.sub,
        'mul': operator.mul,
        'div': protected_division,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
    }
    #User Defined Language Arity
    arity = {
        'add': 2,
        'sub': 2,
        'mul': 2,
        'div': 2,
        'sin': 1,
        'cos': 1,
        'tan': 1,
    }

    #User Defined Data
    def target_function(x):
        return protected_division((math.sin(x) * math.cos(x)), (math.tan(x) + 1)) + (
                x ** 3 - 2 * x ** 2 + 5 * x - 3) * math.sin(x) * 0.5
    def readData(filepath):
        with open(filepath, 'r') as f:
            dataset, meta = arff.loadarff(f)
        data = np.array(dataset)
        return np.array(
            [[element.decode() if isinstance(element, bytes) else element for element in row] for row in data])
    to_compute = readData('points.arff')[:, 0].tolist()
    data_points = [{'x': x, 'actual': target_function(x)} for x in to_compute]


    #Example Usage
    seeds = [random.randint(0, 100000000) for _ in range(10)]
    params = []
    for i in range(10):
        paramTuple = (pop_size, num_genes, terminals, arity, ops, multi_gene_fitness,
                      minInitDepth, maxInitDepth, max_global_depth, mutation_rate, max_mutation_growth,
                      elitism_size, crossover_rate,  max_crossover_growth, num_generations, data_points, fitnessType)
        params.append(paramTuple)
    # Create a ProcessPoolExecutor to run main() concurrently
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use map to execute main() function for each seed and maintain order
        all_results = list(executor.map(call_main, [(seed,) + paramTuple for seed in seeds]))

    # Process or utilize the collected results as needed
    for result in all_results:
        print(f"Stats (genAvg, genMin, genMax, genMed): {result[0]}")
        print(f"y = {result[2].intercept_}")
        for gene, coef in zip(result[1], result[2].coef_):
            if coef < 0:
                print(f" - {-coef}", end="")
            else:
                print(f" + {coef}", end="")
            print(f" * {gene}")
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