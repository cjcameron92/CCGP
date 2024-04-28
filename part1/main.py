#Assignment: 1
#Date: 2/16/2024
#Names: Cole Corbett, Cameron Carvalho
#Student ID: 7246325, 7240450
#Emails: cc21gg@brocku.ca, cc21lz@brocku.ca

#necessary imports
import numpy as np
import toInfix
import operator
import math
import random
from scipy.io import arff
import matplotlib
import matplotlib.pyplot as plt
from functools import partial
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


matplotlib.use('TkAgg') #use the TkAgg backend for matplotlib

'''
@function protectedDiv
@param left - the left operand
@param right - the right operand
@return - the result of the division, or 1 if a ZeroDivisionError occurs
'''
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


'''
@function readData
@param filepath - the path to the file to read
@return - the data from the file as a NumPy array
'''
def readData(filepath):
    with open(filepath, 'r') as f:
        data, meta = arff.loadarff(f)
    # Convert the 'data' attribute to a NumPy array and decode byte strings
    return np.array([[element.decode() if isinstance(element, bytes) else element for element in row] for row in np.array(data)])

pset = gp.PrimitiveSet("MAIN", 1) #create a primitive set with one input
pset.addPrimitive(operator.add, 2) #add the add operator
pset.addPrimitive(operator.sub, 2) #add the subtract operator
pset.addPrimitive(operator.mul, 2) #add the multiply operator
pset.addPrimitive(protectedDiv, 2) #add the protected division operator
pset.addPrimitive(operator.neg, 1) #add the negation operator
#pset.addPrimitive(max, 2) #add the max operator (UNUSED)
#pset.addPrimitive(min, 2) #add the min operator (UNUSED)
#pset.addPrimitive(math.cos, 1) #add the cosine operator (UNUSED)
#pset.addPrimitive(math.sin, 1) #add the sine operator (UNUSED)
pset.addEphemeralConstant("rand101", partial(random.uniform, -1, 1)) #add a random constant between -1 and 1
pset.renameArguments(ARG0='x') #rename the input to x

creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #create a fitness class for minimization
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) #create an individual class with a fitness attribute

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=5) #register the expression generation function
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr) #register the individual generation function
toolbox.register("population", tools.initRepeat, list, toolbox.individual) #register the population generation function
toolbox.register("compile", gp.compile, pset=pset) #register the compilation function

toEvaluate = readData('points.arff')[:, 0].tolist() #read the points to compute from the file

'''
@function evalSymbReg
@param individual - the individual to evaluate
@param points - the points to evaluate the individual on
@return - the average mean squared error between the individual and the real function
'''
def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression and the real function: x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - targetFunction(x))**2 for x in points)
    summed = math.fsum(sqerrors) #Sum the MSEs for each point
    return summed / len(points), #return the average MSE
def targetFunction(x):
    return protectedDiv((math.sin(x) * math.cos(x)), (math.tan(x) + 1)) + (
            x ** 3 - 2 * x ** 2 + 5 * x - 3) * math.sin(x) * 0.5
'''
@function getHits
@param individual - the individual to evaluate
@param points - the points to evaluate the individual on
@return - the number of points the individual hits
'''
def getHits(individual, points):
    hits=0 #initialize the number of hits
    func = toolbox.compile(expr=individual) #compile the individual
    realFunc = lambda x: x**4 + x**3 + x**2 + x #create the real function
    for x in points: #iterate over the points
        if abs(func(x) - realFunc(x)) < 0.0001: #if the individual is within 0.0001 of the real function
            hits+=1 #increment the number of hits
    return str(hits) #return the number of hits


toolbox.register("evaluate", evalSymbReg, points=[_ for _ in toEvaluate]) #register the evaluation function
toolbox.register("select", tools.selTournament, tournsize=3) #register the selection function with tournament size 3
toolbox.register("mate", gp.cxOnePoint) #register the crossover function
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2) #register the expression mutation function
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) #register the mutation function

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)) #set the maximum height for crossover
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)) #set the maximum height for mutation

def main(seed=7246325, mpb=0.2, cpb=0.9, n_gens=50, n_pop=300, elitism=True): #main function to run the GP
    random.seed(seed) #seed the random number generator
    pop = toolbox.population(n=n_pop) #create the initial population
    hof = tools.HallOfFame(1) #create a hall of fame to store the best individuals (also used for elitism)
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0]) #create a statistics object for the fitness
    stats_size = tools.Statistics(len) #create a statistics object for the size of the individuals
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size) #create a multi statistics object
    mstats.register("avg", np.mean) #register the average function for the statistics
    mstats.register("med", np.median) #register the median function for the statistics
    mstats.register("std", np.std) #register the standard deviation function for the statistics
    mstats.register("min", np.min) #register the minimum function for the statistics
    mstats.register("max", np.max) #register the maximum function for the statistics
    pop, log = algorithms.eaSimple(pop, toolbox, cpb, mpb, n_gens, stats=mstats, halloffame=hof, verbose=True, elitism=elitism) #run the GP
    fit_log = log.chapters.get("fitness") #get the fitness log
    return hof[0].fitness.values[0], fit_log, hof[0] #return the best individual's fitness, the fitness log, and the best individual


'''
@function harness
@param numSeeds - the number of seeds to run the experiments with
@param plotFunc1 - the first function to plot
@param plotFunc2 - the second function to plot
@param plotFunc3 - the third function to plot

This function runs the GP over a number of seeds and plots the average, median, and minimum MSE over the generations for each experiment
and prints some information about the runs
'''
def harness(numSeeds=10, plotFunc1="avg", plotFunc2="med", plotFunc3="min"):
    print("Executing GP Runs Please Wait...") #print a message to the user
    random.seed(7246325) #seed the random number generator
    seeds = [random.randint(0, 10000000) for _ in range(numSeeds)] #generate the seeds
    averages = [] + [None] * len(seeds) #experiment 1 avg
    medians = [] + [None] * len(seeds) #experiment 1 median
    minimums = [] + [None] * len(seeds) #experiment 1 min
    acc1 = [] + [None] * len(seeds) #accuracy experiment 1
    expressions1 = [] + [None] * len(seeds) #expressions for experiment 1
    for i in range(len(seeds)): #run the experiments
        _, fit_log, expression = main(seed=seeds[i], cpb=0.9, mpb=0.2, elitism=True) #run experiment 1
        averages[i] = fit_log.select(plotFunc1) #store the avg data
        medians[i] = fit_log.select(plotFunc2) #store the median data
        minimums[i] = fit_log.select(plotFunc3)
        acc1[i] = _ #store the accuracy
        expressions1[i] = expression #store the expression
    averageAvgMSE = [] + [None] * len(averages[0]) #initialize the average data for experiment 1
    averageMedMSE = [] + [None] * len(averages[0]) #initialize the median data for experiment 1
    averageMinMSE = [] + [None] * len(averages[0]) #initialize the min data for experiment 1
    for i in range(len(averages[0])):
        averageAvgMSE[i] = np.mean([averages[j][i] for j in range(len(averages))])
        averageMedMSE[i] = np.mean([medians[j][i] for j in range(len(medians))])
        averageMinMSE[i] = np.mean([minimums[j][i] for j in range(len(minimums))])
    avg_acc1 = np.mean(acc1) #average the accuracy for experiment 1
    med_acc1 = np.median(acc1) #median the accuracy for experiment 1
    min_acc1 = np.min(acc1) #min the accuracy for experiment 1
    print("Average MSE for parameter set 1 (cpb=0.9, mpb=0.2, elitism=True): " + str(avg_acc1))
    print("Median MSE for parameter set 1 (cpb=0.9, mpb=0.2, elitism=True): " + str(med_acc1))
    print("Minimum MSE for parameter set 1 (cpb=0.9, mpb=0.2, elitism=True): " + str(min_acc1) + "\n")
    # print the best accuracy, the number of hits, the seed, the expression, and the experiment
    plt.plot(averageAvgMSE, label="Average") #plot the data for experiment 1
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Average Fitness over Generations')
    plt.legend()
    plt.savefig('plots/average_fitness_plot.png')
    plt.close()
    plt.plot(averageMinMSE, label="Minimum") #plot the data for experiment 1
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Minimum Fitness over Generations')
    plt.legend()
    plt.savefig('plots/minimum_fitness_plot.png')
    plt.close()
    plt.plot(averageMedMSE, label="Median") #plot the data for experiment 1
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Median Fitness over Generations')
    plt.legend()
    plt.savefig('plots/median_fitness_plot.png')
    plt.close()
if __name__ == "__main__":
    harness()

