def fitness(individual):
    validation = [0.0, -0.16290000000000004, -0.26239999999999997, -0.31289999999999996, -0.32639999999999997, -0.3125, -0.2784, -0.2289, -0.1664, -0.09090000000000001, 0.0, 0.1111, 0.24960000000000002, 0.4251, 0.6496000000000001, 0.9375, 1.3056, 1.7730999999999997, 2.3616, 3.0951, 4.0]
    points = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    squarredError = 0
    function = compileTree(individual)
    for i in range (len(points)):
        squarredError += (validation[i] - function(points[i]))**2
    return squarredError / len(points)

realFunction = lambda x: x**4 + x**3 + x**2 + x
for i in range(-10, 12):
    print(realFunction(i/10))