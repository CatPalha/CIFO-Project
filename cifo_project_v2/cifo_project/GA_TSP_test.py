from cifo.algorithm.genetic_algorithm import GeneticAlgorithm
from cifo.custom_problem.travel_salesman_problem import TravelSalesmanProblem

from cifo.algorithm.ga_operators import (
    initialize_randomly, initialize_using_hc, initialize_using_sa,
    RankSelection, RouletteWheelSelection, TournamentSelection, 
    singlepoint_crossover, pmx_crossover, cycle_crossover,
    n_point_crossover, uniform_crossover, order_1_crossover,
    single_point_mutation, swap_mutation, insert_mutation,
    inversion_mutation, scramble_mutation, uniform_mutation,
    standard_replacement, elitism_replacement, initialize_using_ts
)

from cifo.custom_problem.travel_salesman_problem import tsp_bitflip_get_neighbors

data = [
    [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
    [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
    [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
    [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
    [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
    [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
    [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
    [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
    [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
    [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
    [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
    [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
    [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
]

TSP_decision_variables_example = {
    "Distances" : data, #<< Number, Mandatory
    "Cities"    : [i for i in range(0, len(data))], #<< Number, Mandatory
}

TSP_encoding_rule = {
    "Size"         : len(data), # It must be defined by the size of DV (Number of products)
    "Is ordered"   : True,
    "Can repeat"   : False,
    "Data"         : [i for i in range(0, len(data))],
    "Data Type"    : "Choices"
}

tsp = TravelSalesmanProblem(
    decision_variables = TSP_decision_variables_example,
    encoding_rule = TSP_encoding_rule
    )

params = {
    "Population-Size"           : 20,
    "Number-of-Generations"     : 250,
    
    "Crossover-Probability"     : 0.8,
    "Mutation-Probability"      : 0.5,
    
    "Initialization-Approach"   : initialize_randomly,
    "Selection-Approach"        : RouletteWheelSelection(),
    "Tournament-Size"           : 5,
    "Crossover-Approach"        : pmx_crossover,
    "Mutation-Aproach"          : insert_mutation,
    "Replacement-Approach"      : elitism_replacement
}

hc_init_params = {
    "Maximum-Iterations" : 20,
    "Stop-Conditions" : "Alternative-01",
    "Neighborhood-Size": -1,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

sa_init_params = {
    "Maximum-Internal-Iterations" : 10,
    "Maximum-Iterations" : 10,
    #"Initial-C" : 200,
    #"Minimum-C" : 0.01,
    "Update-Method" : "Linear",
    "Update-Rate" : 0.5,
    "Initialize-Method-C" : "Classical",
    "Initialize-Method-Minimum-C" : "Classical",
    "Neighborhood-Size" : 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

ts_init_params = {
    "Maximum-Iterations" : 20,
    "Stop-Conditions" : "Alternative-01",
    "Neighborhood-Size": -1,
    "Memory-Size":-1,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

"""
solution = GeneticAlgorithm(
    problem_instance = tsp
)
"""

solution = GeneticAlgorithm(
    problem_instance = tsp,
    params = params,
    init_params = sa_init_params
)

import matplotlib.pyplot as plt

fit = []
for i in range(1,20):
    print(solution.search())
    fit.append(solution.search().fitness)

avg = sum(fit)/len(fit)

plt.plot(range(1,20),fit)
plt.axhline(y=avg, color='r', linestyle='-', label=str(avg))
plt.legend()
plt.show()

# Optimum starting on 0: 7999 