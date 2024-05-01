# CCGP
[![Paper Build Status](https://img.shields.io/github/actions/workflow/status/PaperMC/Paper/build.yml?branch=master)](https://github.com/PaperMC/Paper/actions)
[![Discord](https://img.shields.io/discord/289587909051416579.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/papermc)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/papermc?label=GitHub%20Sponsors)](https://github.com/sponsors/cjcameron92)
----
Python-based multi-gene genetic program designed to manipulate and analyze complex data.
## Table of Contents
- [Installation](https://link-url-here.org)
- [Docs](https://github.com/cjcameron92/CCGP/blob/main/docs/)
- [Examples](https://github.com/cjcameron92/CCGP/tree/main/examples)
- [FAQ](https://github.com/cjcameron92/CCGP/blob/main/docs/faq.md)

### Quickstart
Install python library
```pip
pip install ccgp
```
Setup Enviorment
```py
from ccgp import runGP

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

ops = {
    'add': add,
    'sub': operator.sub,
    'mul': operator.mul,
    'div': protected_division,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
}

arity = {
    'add': 2,
    'sub': 2,
    'mul': 2,
    'div': 2,
    'sin': 1,
    'cos': 1,
    'tan': 1,
}
seed = 1
paramTuple = (seed, pop_size, num_genes, terminals, arity, ops, multi_gene_fitness,
    minInitDepth, maxInitDepth, max_global_depth, mutation_rate, max_mutation_growth,
    elitism_size, crossover_rate,  max_crossover_growth, num_generations, data_points, fitnessType)

def call_main(args):
    return runGP(*args)        

call_main(paramTuple)
```
---
#### Contributors 
- [Cameron Carvalho](https://github.com/cjcameron92)
- [Cole Corbett](https://github.com/ccorbett0116)
