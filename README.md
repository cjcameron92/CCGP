#  MGGP User Guide

## Setup and Installation
Create a python 3.9 or higher enviorment and install the following python packages onto the enviorment.
```python
pip install ccgp
```

## Define Problem Scope
Determine what problem you're trying to solve and create a fitness function. 
Below is an example for the evaluation of each gene in an individual on multiple datapoints, along with a basic regression model. We recommend sklearn LinearRegression to determine weights
for each genes solution. 
The fitness function must critically include three variables in its signature, and they should be expected in this order.
The fitness function must also return the fitness and model in this order.
```python
def multi_gene_fitness(individual, operators, data_points):
    X = np.array([[gene.evaluate(point, operators) for gene in individual] for point in data_points])
    y = np.array([data['actual'] for point in data_points])

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    mse = np.mean((predictions - y) ** 2)

    return mse, model
```

## Configuration 
Next step is to configure your settings for your terminals, operators, and parameters.
Provided below is an example configuration for the basic regression model. 
```python
import operator
import random
import math


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
def add(*args):
    return sum(args)
def protected_division(x, y):
    if y == 0:
        return 1
    return x / y
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
```

## Execute and Run the GP
Then create an object with all the configurations from above, you can copy the method to execute by using call_main and passing the object with the seed (if required)
```python
 paramTuple = (pop_size, num_genes, terminals, arity, ops, multi_gene_fitness,
                      minInitDepth, maxInitDepth, max_global_depth, mutation_rate, max_mutation_growth,
                      elitism_size, crossover_rate,  max_crossover_growth, num_generations, data_points, fitnessType)

# Create a ProcessPoolExecutor to run main() concurrently
with concurrent.futures.ProcessPoolExecutor() as executor:
  # Use map to execute main() function for each seed and maintain order
  all_results = list(executor.map(call_main, [(seed,) + paramTuple for seed in seeds]))

# Process or utilize the collected results as needed
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

// do something with your "bestResult"
``` 
