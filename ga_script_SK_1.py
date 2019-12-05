from cifo.algorithm.genetic_algorithm import GeneticAlgorithm
from cifo.algorithm.hill_climbing import HillClimbing
from cifo.custom_problem.knapsack_problem import (
    KnapsackProblem, knapsack_decision_variables_example, knapsack_constraints_example, 
    knapsack_bitflip_get_neighbors
)
from cifo.problem.objective import ProblemObjective
from cifo.algorithm.ga_operators import (
    initialize_randomly,
    RouletteWheelSelection, RankSelection, TournamentSelection, 
    singlepoint_crossover,
    single_point_mutation,
    elitism_replacement, standard_replacement 
)    
from cifo.util.terminal import Terminal, FontColor
from random import randint

dv = {
    "Values"    : [360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147, 
    78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28, 87, 73, 78, 15, 
    26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276, 312], 

    "Weights"   : [7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
    42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
    3, 86, 66, 31, 65, 0, 79, 20, 65, 52, 13]
}

knapsack_problem_instance = KnapsackProblem( 
    decision_variables = dv,
    constraints = { "Max-Weight" : 400 })


ga1 = GeneticAlgorithm( 
    problem_instance = knapsack_problem_instance,
    params = {
        # params
        "Population-Size"           : 10,
        "Number-of-Generations"     : 10,
        "Crossover-Probability"     : 0.8,
        "Mutation-Probability"      : 0.5,
        # operators / approaches
        "Initialization-Approach"   : initialize_randomly,
        "Selection-Approach"        : RouletteWheelSelection,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : singlepoint_crossover,
        "Mutation-Aproach"          : single_point_mutation,
        "Replacement-Approach"      : elitism_replacement
    } )

ga1.search()


