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
def if_then_else(input, output1, output2):
    return output1 if input else output2

'''
@function protectedLog
@param x - the value to take the log of
@return - the log of x if x > 0, otherwise 0
'''
def protectedLog(x):
    if x <= 0:
        return 0
    else:
        return math.log(x)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) #create a fitness class that minimizes the fitness value
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) #create an individual class that is a primitive tree with a fitness attribute

toolbox = base.Toolbox() #create a toolbox for the genetic programming algorithm

'''
@function main
@param data_filepath - the path to the file containing the data
@param pop_size - the size of the population
@param n_gen - the number of generations to run
@param seed - the seed for the random number generator
@param cpb - the probability of crossover
@param mpb - the probability of mutation
@param eph - the type of ephemeral constant to use
@param elitism - whether or not to use elitism
@return - the accuracy of the best individual, the fitness log, the best individual, the testing data, the testing answers, and the training accuracy'''
def main(data_filepath='Rice_Cammeo_Osmancik.arff', pop_size=100, n_gen=40, seed=7246325, cpb=0.85, mpb=0.25, eph="int", elitism=True):
    pset = gp.PrimitiveSet("MAIN", 7) #create a primitive set with 7 input features
    pset.addPrimitive(operator.add, 2) #add the add operator
    pset.addPrimitive(operator.sub, 2) #add the subtract operator
    pset.addPrimitive(operator.mul, 2) #add the multiply operator
    pset.addPrimitive(protectedDiv, 2) #add the protected division operator
    # pset.addPrimitive(operator.neg, 1) #add the negation operator (UNUSED)
    # pset.addPrimitive(max, 2) #add the max operator (UNUSED)
    # pset.addPrimitive(min, 2) #add the min operator (UNUSED)
    # pset.addPrimitive(if_then_else, 3) #add the if-then-else operator (UNUSED)
    pset.addPrimitive(math.cos, 1) #add the cosine operator
    pset.addPrimitive(math.sin, 1) #add the sine operator
    pset.addPrimitive(protectedLog, 1) #add the protected log operator
    pset.addEphemeralConstant("rand101", partial(random.uniform, -1, 1)) #add an ephemeral constant that is a random float between -1 and 1
    #rename the arguments
    pset.renameArguments(ARG0='Area')
    pset.renameArguments(ARG1='Perimeter')
    pset.renameArguments(ARG2='Major_Axis_Length')
    pset.renameArguments(ARG3='Minor_Axis_Length')
    pset.renameArguments(ARG4='Eccentricity')
    pset.renameArguments(ARG5='Convex_Area')
    pset.renameArguments(ARG6='Extent')
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=5) #register the expr function to generate a random tree
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr) #register the individual function to create an individual with the expr function
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) #register the population function to create a population of individuals
    toolbox.register("compile", gp.compile, pset=pset) #register the compile function to compile a tree into a callable function


    '''
    @function evalRice
    @param individual - the individual to evaluate
    @param points - the points to evaluate the individual on
    @return - the fitness of the individual
    '''
    def readData(filepath):
        with open(filepath, 'r') as f: #open the file
            data, meta = arff.loadarff(f) #load the data from the file
        return np.array([[element.decode() if isinstance(element, bytes) else element for element in row] for row in np.array(data)]) #return the data as a NumPy array

    random.seed(seed) #seed the random number generator

    # Read data and convert to list
    fullData = readData(data_filepath).tolist() #read the data from the file and convert it to a list

    # Shuffle the data
    random.shuffle(fullData) #shuffle the data

    # Extract input and validation data
    trainingData = np.array(fullData)[0:450, :7].tolist()  # Assuming columns 0 to 6 are input features
    trainingAnswers = np.array(fullData)[0:450, 7].tolist()  # Assuming column 7 is the target variable
    testingData = np.array(fullData)[450:, :7].tolist()  # Assuming columns 0 to 6 are input features
    testingAnswers = np.array(fullData)[450:, 7].tolist()  # Assuming column 7 is the target variable


    '''
    @function evalRice
    @param individual - the individual to evaluate
    @param points - the points to evaluate the individual on
    @return - the fitness of the individual
    '''
    def evalRice(individual, points):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual) #compile the individual into a callable function
        score = 0 #initialize the score to 0
        for i in range(len(points)): #for each point
            for el in points[i]: #for each element in the point
                points[i][points[i].index(el)] = float(el) #convert the element to a float
            classification = func(*points[i]) #classify the point
            if classification <=0.5: # <= 0.5 should be Osmancik
                if trainingAnswers[i] == "Cammeo": #if the actual value is Cammeo but the classification is Osmancik
                    score += 1 #increment the score
            else: # > 0.5 should be Cammeo
                if trainingAnswers[i] == "Osmancik": #if the actual value is Osmancik but the classification is Cammeo
                    score += 1 #increment the score
        return score, #return the score as the fitness of the individual


    toolbox.register("evaluate", evalRice, points=[_ for _ in trainingData]) #register the evaluate function to evaluate the individual on the training data
    toolbox.register("select", tools.selTournament, tournsize=3) #register the select function to select individuals using tournament selection
    toolbox.register("mate", gp.cxOnePoint) #register the mate function to perform one-point crossover
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2) #register the expr_mut function to generate a random tree for mutation
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset) #register the mutate function to perform uniform mutation
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)) #limit the height of the trees
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)) #limit the height of the trees

    '''
    @function runEA
    @return - the final population, the log, and the hall of fame
    '''
    def runEA():
        pop = toolbox.population(pop_size) #create the initial population
        hof = tools.HallOfFame(1) #create the hall of fame to store the best individuals also used for elitism

        stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0]) #create a statistics object for the fitness
        stats_size = tools.Statistics(len) #create a statistics object for the size of the individuals
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size) #create a multi-statistics object to store the statistics
        mstats.register("avg", np.mean) #register the average function for the statistics
        mstats.register("med", np.median) #register the median function for the statistics
        mstats.register("std", np.std) #register the standard deviation function for the statistics
        mstats.register("min", np.min) #register the minimum function for the statistics
        mstats.register("max", np.max) #register the maximum function for the statistics

        pop, log = algorithms.eaSimple(pop, toolbox, cpb, mpb, n_gen, stats=mstats,
                                       halloffame=hof, verbose=False) #run the genetic programming algorithm
        return pop, log, hof #return the final population, the log, and the hall of fame
    pop, log, hof = runEA() #run the genetic programming algorithm
    func = toolbox.compile(expr=hof[0]) #compile the best individual into a callable function
    counter = 0 #initialize the counter to 0
    score = 0 #initialize the score to 0
    for i in range(len(testingData)): #for each point in the testing data
        counter += 1 #increment the counter
        for el in testingData[i]: #for each element in the point
            testingData[i][testingData[i].index(el)] = float(el) #convert the element to a float
        classification = func(*testingData[i]) #classify the point
        if classification <= 0.5:  # <= 0.5 should be Osmancik
            if testingAnswers[i] == "Cammeo": #if the actual value is Cammeo but the classification is Osmancik
                score += 1 #increment the score
        else: # > 0.5 should be Cammeo
            if testingAnswers[i] == "Osmancik": #if the actual value is Osmancik but the classification is Cammeo
                score += 1 #increment the score
    trainingCounter = 0 #initialize the training counter to 0
    trainingScore = 0 #initialize the training score to 0
    for i in range(len(trainingData)): #for each point in the training data
        trainingCounter += 1 #increment the training counter
        for el in trainingData[i]: #for each element in the point
            trainingData[i][trainingData[i].index(el)] = float(el) #convert the element to a float
        classification = func(*trainingData[i]) #classify the point
        if classification <= 0.5: # <= 0.5 should be Osmancik
            if trainingAnswers[i] == "Cammeo": #if the actual value is Cammeo but the classification is Osmancik
                trainingScore += 1 #increment the training score
        else: # > 0.5 should be Cammeo
            if trainingAnswers[i] == "Osmancik": #if the actual value is Osmancik but the classification is Cammeo
                trainingScore += 1 #increment the training score
    fit_log = log.chapters.get("fitness") #get the fitness log
    # return the accuracy of the best individual, the fitness log, the best individual, the testing data, the testing answers, and the training accuracy
    return 100*((counter-score)/counter), fit_log, hof[0], testingData, testingAnswers, 100*((trainingCounter-trainingScore)/trainingCounter)

'''
@function confusionMatrix
@param individual - the individual to evaluate
@param testing - the testing data
@param answers - the answers for the testing data
@return - the confusion matrix values for the individual with respect to the testing data
'''
def confusionMatrix(individual, testing, answers):
    func = toolbox.compile(expr=individual)
    trueOsmancik = 0
    trueCammeo = 0
    falseCammeo = 0
    falseOsmancik = 0
    for i in range(len(testing)):
        for el in testing[i]:
            testing[i][testing[i].index(el)] = float(el)
        classification = func(*testing[i])
        if classification <= 0.5:  # <= 0.5 should be Osmancik
            if answers[i] == "Cammeo": # if the actual value is Cammeo but the classification is Osmancik
                falseCammeo += 1 #increment the falseCammeo counter
            else: # if the actual value is Osmancik and the classification is Osmancik
                trueOsmancik += 1  #increment the trueOsmancik counter
        else: # > 0.5 should be Cammeo
            if answers[i] == "Osmancik": # if the actual value is Osmancik but the classification is Cammeo
                falseOsmancik += 1 # increment the falseOsmancik counter
            else: # if the actual value is Cammeo and the classification is Cammeo
                trueCammeo += 1 # increment the trueCammeo counter
    return trueOsmancik, trueCammeo, falseOsmancik, falseCammeo

'''
@function harness
@param numSeeds - the number of seeds to use
@param plotFunc - the fitness function to plot
@param plotFunc2 - the fitness function to plot

This function runs the GP over a number of seeds and plots the average, median, and minimum MSE over the generations for each experiment
and prints some information about the runs
'''
def harness(numSeeds=10, plotFunc="min", plotFunc2="avg", plotfunc3="med"):
    print("Executing GP Runs Please Wait...")
    random.seed(7246325) #seed the random number generator
    seeds = [random.randint(0, 10000000) for _ in range(numSeeds)] #generate a list of random seeds
    datasets2 = [[],[]] #initialize the datasets for the second experiment
    exp21 = [] + [None] * len(seeds) #initialize the fitness logs for the second experiment
    exp22 = [] + [None] * len(seeds) #initialize the fitness logs for the second experiment
    exp23 = [] + [None] * len(seeds) #initialize the fitness logs for the second experiment
    acc2 = [] + [None] * len(seeds) #initialize the accuracy logs for the second experiment
    acc2Train = [] + [None] * len(seeds) #initialize the accuracy logs for the second experiment
    expressions2 = [] + [None] * len(seeds) #initialize the expressions for the second experiment
    for i in range(len(seeds)): #for each seed
        accuracy, fit_log, expression, data, answers, trainScore = main(seed=seeds[i], eph="float") #run the second experiment
        acc2[i]=accuracy #store the accuracy
        acc2Train[i]=trainScore #store the training accuracy
        exp21[i]=fit_log.select(plotFunc) #store the minimum fitness log
        exp22[i]=fit_log.select(plotFunc2) #store the average fitness log
        exp23[i]=fit_log.select(plotfunc3) #store the median fitness log
        expressions2[i]=expression #store the best individual
        datasets2[0].append(data) #store the testing data
        datasets2[1].append(answers) #store the testing answers
    avg_exp2 = [np.mean([exp21[i][j] for i in range(len(exp21))]) for j in range(len(exp21[0]))] #calculate the average fitness log for the second experiment
    min_exp2 = [np.mean([exp22[i][j] for i in range(len(exp22))]) for j in range(len(exp22[0]))] #calculate the minimum fitness log for the second experiment
    med_exp2 = [np.mean([exp23[i][j] for i in range(len(exp23))]) for j in range(len(exp23[0]))] #calculate the median fitness log for the second experiment
    avg_acc2 = np.mean(acc2) #calculate the average accuracy for the second experiment
    avg_acc2Train = np.mean(acc2Train) #calculate the average training accuracy for the second experiment
    max_acc2 = max(acc2) #calculate the best accuracy for the second experiment
    max_acc2Train = max(acc2Train) #calculate the best training accuracy for the second experiment
    print("Average Accuracy for parameter set 2 on training %" + str(avg_acc2Train)) #print the average training accuracy for the second experiment
    print("Best Accuracy for parameter set 2 on training %" + str(max_acc2Train)) #print the best training accuracy for the second experiment
    print("Average Accuracy for parameter set 2 on testing %" + str(avg_acc2)) #print the average accuracy for the second experiment
    print("Best Accuracy for parameter set 2 on testing %" + str(max_acc2)) #print the best accuracy for the second experiment
    print("Confusion Matrix for best expression on testing of parameter set 2: (True Osmancik, True Cammeo, False Osmancik, False Cammeo)", confusionMatrix(expressions2[acc2.index(max_acc2)], datasets2[0][acc2.index(max_acc2)], datasets2[1][acc2.index(max_acc2)]), "expression: " + str(toInfix.functional_to_infix(toInfix.parse_functional_expression(str(expressions2[acc2.index(max_acc2)])))) + "\n\n")
    #print the confusion matrix for the best individual of the second experiment
    bestAcc = max(acc2)
    print("Best Accuracy: %" + str(bestAcc) + " on seed " + str(seeds[acc2.index(bestAcc)]) + " and expression: " + str(toInfix.functional_to_infix(toInfix.parse_functional_expression(str(expressions2[acc2.index(bestAcc)])))) + " from experiment 2 (Run: " + str(acc2.index(bestAcc)) + ")")
        # print the best accuracy, the seed, and the best individual and run number of the first experiment
    plt.plot(avg_exp2, label="Average") #plot the average fitness for the second experiment
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Average Fitness over Generations')
    plt.legend()
    plt.savefig('plots/average_fitness_plot.png')
    plt.close()
    plt.plot(min_exp2, label="Minimum") #plot the minimum fitness for the second experiment
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Minimum Fitness over Generations')
    plt.legend()
    plt.savefig('plots/minimum_fitness_plot.png')
    plt.close()
    plt.plot(med_exp2, label="Median") #plot the median fitness for the second experiment
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Median Fitness over Generations')
    plt.legend()
    plt.savefig('plots/median_fitness_plot.png')
    plt.close()


harness() #run the harness function