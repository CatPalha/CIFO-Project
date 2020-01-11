from random import uniform, randint, choice, sample
from copy import deepcopy

from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import EncodingDataType
from cifo.problem.population import Population
from cifo.problem.solution import LinearSolution

from cifo.algorithm.hill_climbing import HillClimbing
from cifo.algorithm.simulated_annealing import SimulatedAnnealing
from cifo.algorithm.tabu_search import TabuSearch

from cifo.custom_problem.travel_salesman_problem import tsp_bitflip_get_neighbors
from cifo.custom_problem.knapsack_problem import knapsack_bitflip_get_neighbors
from cifo.custom_problem.portfolio_investment_problem import pip_bitflip_get_neighbors

import numpy as np

###################################################################################################
# INITIALIZATION APPROACHES
###################################################################################################

# (!) REMARK:
# Initialization signature: <method_name>( problem, population_size ):

# -------------------------------------------------------------------------------------------------
# Random Initialization 
# -------------------------------------------------------------------------------------------------
def initialize_randomly( problem, population_size, params={} ):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm
    
    Required:
    
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    
    @ population_size - to define the size of the population to be returned. 
    """
    solution_list = []

    i = 0
    # generate a population of admissible solutions (individuals)
    for _ in range(0, population_size):
        s = problem.build_solution()
        
        # check if the solution is admissible
        while not problem.is_admissible( s ):
            s = problem.build_solution()
        
        s.id = [0, i]
        i += 1
        problem.evaluate_solution ( s )
        
        solution_list.append( s )

    population = Population( 
        problem = problem , 
        maximum_size = population_size, 
        solution_list = solution_list )
    
    return population

# -------------------------------------------------------------------------------------------------
# Initialization using Hill Climbing
# -------------------------------------------------------------------------------------------------
#TODO: OPTIONAL, implement a initialization based on Hill Climbing
# Remark: remember, you will need a neighborhood functin for each problem
def initialize_using_hc( problem, population_size, params={} ):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm
    
    Required:
    
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    
    @ population_size - to define the size of the population to be returned. 

    @ params - the parameters to create the Hill Climbing object (default: empty dictionary)
    """
    solution_list = []

    neighborhood_function = params["Neighborhood-Function"]

    # initialize the Hill Climbing class
    hc = HillClimbing(
        problem_instance = problem,
        neighborhood_function = neighborhood_function,
        params = params
        )

    # generate a population of admissible solutions (individuals)
    for _ in range(0, population_size):
        # search for an optimal solution using Hill Climbing
        s = hc.search()
        
        # there's no need to check if the solution is admissible
        # because hill climbing already generates admissible solutions
        
        # get the fitness of the solution
        problem.evaluate_solution ( s )

        # add the solution to the population
        solution_list.append( s )

    # create the population object
    population = Population( 
        problem = problem , 
        maximum_size = population_size, 
        solution_list = solution_list
        )
    
    return population

# -------------------------------------------------------------------------------------------------
# Initialization using Simulated Annealing
# -------------------------------------------------------------------------------------------------
#TODO: OPTIONAL, implement a initialization based on Hill Climbing
# Remark: remember, you will need a neighborhood functin for each problem
def initialize_using_sa( problem, population_size, params={} ):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm
    
    Required:
    
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    
    @ population_size - to define the size of the population to be returned. 

    @ params - the parameters to create the Simulated Annealing object (default: empty dictionary)
    """
    
    solution_list = []

    neighborhood_function = params["Neighborhood-Function"]

    # generate a population of admissible solutions (individuals)
    for _ in range(0, population_size):
        # initialize the Simulated Annealing class
        # notice that we need to create a new Simulated Annealing object for every iteration
        # because the initial value of C has to be always the same
        sa = SimulatedAnnealing(
            problem_instance = problem,
            neighborhood_function = neighborhood_function,
            params = params
        )
        
        # search for the optimal solution using Simulated Annealing
        s = sa.search()
        
        # there's no need to check if the solution is admissible
        # because Simulated Annealing already generates admissible solutions
        
        # get the fitness of the solution
        problem.evaluate_solution ( s )
        
        # add the solution to the population
        solution_list.append( s )

    # create the Population object
    population = Population( 
        problem = problem , 
        maximum_size = population_size, 
        solution_list = solution_list
        )
    
    return population

def initialize_using_ts( problem, population_size, params={} ):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm
    
    Required:
    
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    
    @ population_size - to define the size of the population to be returned. 

    @ params - the parameters to create the Tabu Search object (default: empty dictionary)
    """
    solution_list = []

    neighborhood_function = params["Neighborhood-Function"]

    # initialize the Tabu Search class
    ts = TabuSearch(
        problem_instance = problem,
        neighborhood_function = neighborhood_function,
        params = params
        )

    # generate a population of admissible solutions (individuals)
    for _ in range(0, population_size):
        # find the optimal solution using Tabu Search
        s = ts.search()
        
        # there's no need to check if the solution is admissible
        # because Tabu Search already generates admissible solutions
        
        # get the fitness of the solution
        problem.evaluate_solution ( s )
        
        # add the solution to the population
        solution_list.append( s )

    # create the Population object
    population = Population( 
        problem = problem , 
        maximum_size = population_size, 
        solution_list = solution_list
        )
    
    return population

###################################################################################################
# SELECTION APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# class RouletteWheelSelection
# -------------------------------------------------------------------------------------------------
# TODO: implement Roulette Wheel for Minimization
class RouletteWheelSelection:
    """
    Main idea: better individuals get higher chance
    The chances are proportional to the fitness
    Implementation: roulette wheel technique
    Assign to each individual a part of the roulette wheel
    Spin the wheel n times to select n individuals

    REMARK: This implementation does not consider minimization problem
    """

    def select(self, population, objective, params):
        """
        select two different parents using roulette wheel
        """
        # objective is a new argument in _select_index
        index1 = self._select_index(population = population, objective = objective)
        index2 = index1
        
        while index2 == index1:
            index2 = self._select_index( population = population, objective =objective )

        return population.get( index1 ), population.get( index2 )

    # we added objective as an argument
    def _select_index(self, population, objective ):
        # We changed this whole function by creating fitness_list

        # save the fitness of the solutions in the population so we don't affect
        # the actual fitness of the solutions
        fitness_list = [solution.fitness for solution in population.solutions]

        if objective == 'Minimization':
            # save the largest fitness
            fit_max = max(fitness_list)

            # subtract a solution's fitness from the largest fitness to get
            # a temporary finess to use with the roulette wheel
            fitness_list = [fit_max - solution.fitness for solution in population.solutions]

        # check if any solution has negative fitness
        if any(fitness < 0 for fitness in fitness_list):
            # save the smallest fitness
            fit_min = min(fitness_list)

            # subtract the smallest fitness from all of the solutions' fitness
            fitness_list = [fitness - fit_min for fitness in fitness_list]
        
        # check if any solution has fitness 0
        if 0 in fitness_list:
            # add 1 to all the solution's fitness
            fitness_list = [fitness + 1 for fitness in fitness_list]
        
        # Get the Total Fitness (all solutions in the population) to calculate the chances proportional to fitness
        total_fitness = 0
        for fitness in fitness_list:
            total_fitness += fitness

        # spin the wheel
        wheel_position = uniform( 0, 1 )

        # calculate the position which wheel should stop
        stop_position = 0
        index = 0
        for fitness in fitness_list:
            stop_position += (fitness / total_fitness)
            if stop_position > wheel_position :
                break
            index += 1    

        return index    
        
# -------------------------------------------------------------------------------------------------
# class RankSelection
# -------------------------------------------------------------------------------------------------
class RankSelection:
    """
    Rank Selection sorts the population first according to fitness value and ranks them. Then every chromosome is allocated selection probability with respect to its rank. Individuals are selected as per their selection probability. Rank selection is an exploration technique of selection.
    """
    def select(self, population, objective, params):
        # Step 1: Sort / Rank
        population = self._sort( population, objective )

        # Step 2: Create a rank list [0, 1, 1, 2, 2, 2, ...]
        rank_list = []

        for index in range(0, len(population.solutions)):
            for _ in range(0, index + 1):
                rank_list.append( index )

        # print(f" >> rank_list: {rank_list}")       

        # Step 3: Select solution index
        index1 = randint(0, len( rank_list )-1)
        index2 = index1
        
        while index2 == index1:
            index2 = randint(0, len( rank_list )-1)

        return population.get( rank_list [index1] ), population.get( rank_list[index2] )

    
    def _sort( self, population, objective ):

        if objective == ProblemObjective.Maximization:
            for i in range (0, len( population.solutions )):
                for j in range (i, len (population.solutions )):
                    if population.solutions[ i ].fitness > population.solutions[ j ].fitness:
                        swap = population.solutions[ j ]
                        population.solutions[ j ] = population.solutions[ i ]
                        population.solutions[ i ] = swap
                        
        else:    
            for i in range (0, len( population.solutions )):
                for j in range (i, len (population.solutions )):
                    if population.solutions[ i ].fitness < population.solutions[ j ].fitness:
                        swap = population.solutions[ j ]
                        population.solutions[ j ] = population.solutions[ i ]
                        population.solutions[ i ] = swap

        return population

# -------------------------------------------------------------------------------------------------
# class TournamentSelection
# -------------------------------------------------------------------------------------------------
class TournamentSelection:  
    """
    """
    def select(self, population, objective, params):
        tournament_size = 2
        if "Tournament-Size" in params:
            tournament_size = params[ "Tournament-Size" ]

        index1 = self._select_index( objective, population, tournament_size )    
        index2 = index1
        
        while index2 == index1:
            index2 = self._select_index( objective, population, tournament_size )

        return population.solutions[ index1 ], population.solutions[ index2 ]


    def _select_index(self, objective, population, tournament_size ): 
        
        index_temp      = -1
        index_selected  = randint(0, population.size - 1)

        if objective == ProblemObjective.Maximization: 
            for _ in range( 0, tournament_size ):
                index_temp = randint(0, population.size - 1 )

                if population.solutions[ index_temp ].fitness > population.solutions[ index_selected ].fitness:
                    index_selected = index_temp
        elif objective == ProblemObjective.Minimization:
            for _ in range( 0, tournament_size ):
                index_temp = randint(0, population.size - 1 )

                if population.solutions[ index_temp ].fitness < population.solutions[ index_selected ].fitness:
                    index_selected = index_temp            

        return index_selected         

###################################################################################################
# CROSSOVER APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Singlepoint crossover
# -------------------------------------------------------------------------------------------------
def singlepoint_crossover( problem, solution1, solution2):
    singlepoint = randint(0, len(solution1.representation)-1)
    #print(f" >> singlepoint: {singlepoint}")

    offspring1 = deepcopy(solution1) #solution1.clone()
    offspring2 = deepcopy(solution2) #.clone()

    for i in range(singlepoint, len(solution2.representation)):
        offspring1.representation[i] = solution2.representation[i]
        offspring2.representation[i] = solution1.representation[i]

    return offspring1, offspring2    

# -------------------------------------------------------------------------------------------------
# Partially Mapped Crossover
# -------------------------------------------------------------------------------------------------
# TODO: implement Partially Mapped Crossover
def pmx_crossover( problem, solution1, solution2):
    # get the representation (list) of both solutions
    solution_1 = solution1.representation
    solution_2 = solution2.representation

    # choose two different random points in the solutions
    point1 = randint( 0, len( solution_1 )-1 )
    point2 = point1

    while point1 == point2:
        point2 = randint( 0, len( solution_1 )-1 )

    # make sure that the fist crossover point is the smallest and vice versa
    firstCrossPoint = min(point1,point2)
    secondCrossPoint = max(point1,point2)
    
    # select the middle section of each parent
    parent1MiddleCross = solution_1[firstCrossPoint:secondCrossPoint]
    parent2MiddleCross = solution_2[firstCrossPoint:secondCrossPoint]

    # create children that are equal to the respective parent minus the middle section
    child1 = solution_1[:firstCrossPoint] + parent2MiddleCross + solution_1[secondCrossPoint:]
    child2 = solution_2[:firstCrossPoint] + parent1MiddleCross + solution_2[secondCrossPoint:]

    # save the relations in the middle section
    relations = []

    for i in range(len(parent1MiddleCross)):
        relations.append([parent2MiddleCross[i], parent1MiddleCross[i]])

    # check if any of the children have repeated elements in them
    counts1 = [child1.count(i) for i in child1]
    counts2 = [child2.count(i) for i in child2]

    # while there are repeated elements in child 1
    while len([x for x in counts1 if x > 1]) > 0:
        # check the part before the middle section
        for i in child1[:firstCrossPoint]:
            for j in parent2MiddleCross:
                if i == j:
                    # replace the repeated element OUTSIDE of the middle section using the relations we defined
                    index_j = parent2MiddleCross.index(j)
                    relation = relations[index_j]
                    index_i = child1.index(i)
                    child1[index_i] = relation[1]
        
        # check the part after the middle section
        for i in child1[secondCrossPoint:]:
            for j in parent2MiddleCross:
                if i == j:
                    # replace the repeated element OUTSIDE of the middle section using the relations we defined
                    index_j = parent2MiddleCross.index(j)
                    relation = relations[index_j]
                    index_i = child1.index(i,secondCrossPoint)
                    child1[index_i] = relation[1]

        # check again if there are repeated elements in child 1
        counts1 = [child1.count(i) for i in child1]

    # while there are repeated elements in child 1
    while len([x for x in counts2 if x > 1]) > 0:
        # check the part before the middle section
        for i in child2[:firstCrossPoint]:
            for j in parent1MiddleCross:
                if i == j:
                    # replace the repeated element OUTSIDE of the middle section using the relations we defined
                    index_j = parent1MiddleCross.index(j)
                    relation = relations[index_j]
                    index_i = child2.index(i)
                    child2[index_i] = relation[0]
        
        # check the part after the middle section
        for i in child2[secondCrossPoint:]:
            for j in parent1MiddleCross:
                if i == j:
                    # replace the repeated element OUTSIDE of the middle section using the relations we defined
                    index_j = parent1MiddleCross.index(j)
                    relation = relations[index_j]
                    index_i = child2.index(i,secondCrossPoint)
                    child2[index_i] = relation[0]

        # check again if there are repeated elements in child 1
        counts2 = [child2.count(i) for i in child2]

    # create LinearSolution objects
    child_1 = LinearSolution(child1, solution1.encoding_rule)
    child_2 = LinearSolution(child2, solution2.encoding_rule)

    return child_1, child_2

# -------------------------------------------------------------------------------------------------
# Cycle Crossover
# -------------------------------------------------------------------------------------------------
# TODO: implement Cycle Crossover
def cycle_crossover(problem, solution1, solution2):
    # get the representation (list) of both solutions
    solution_1 = solution1.representation
    solution_2 = solution2.representation

    cycles = []    
    considered = []
    
    # find the cycles
    while len(considered) < len(solution_1):
        i = 0

        while i in considered:
            i += 1
        
        cycle =  []
        full_cycle = False

        while full_cycle == False:
            cycle.append(i)
            considered.append(i)
            i = solution_1.index(solution_2[i])

            if i in considered:
                full_cycle = True

        cycles.append(cycle)

    # create empty children    
    child1 = [None] * len(solution_1)
    child2 = [None] * len(solution_1)
    
    # getting the children
    for i, cycle in enumerate(cycles):
        # note that here cycle 1 is the cycle with index 0
        if i % 2 == 0:
            for j in cycle:
                child1[j] = solution_1[j]
                child2[j] = solution_2[j]
        else:
            for j in cycle:
                child1[j] = solution_2[j]
                child2[j] = solution_1[j]

    # save the children as LinearSolution objects
    child_1 = LinearSolution(child1, solution1.encoding_rule)
    child_2 = LinearSolution(child2, solution2.encoding_rule)

    return child_1, child_2

# -------------------------------------------------------------------------------------------------
# N-point crossover
# -------------------------------------------------------------------------------------------------
def n_point_crossover( problem, solution1, solution2):
    # choose n different random points
    n_points_choice = []

    for i in range(0, len(solution1.representation)):
        n_points_choice.append(choice([0,1]))

    n_points = []

    for i in range(0, len(solution1.representation)):
        if n_points_choice[i] == 1:
            n_points.append(i)

    # create two children
    offspring1 = deepcopy(solution1) #solution1.clone()
    offspring2 = deepcopy(solution2) #.clone()

    # alternate whether we're crossing the parents or not
    for j in n_points:
        ind = n_points.index(j)
        if ind % 2 == 0:
            # check if we're in the end section
            if ind < len(n_points)-1:
                j2 = n_points[ind+1]
            else:
                j2 = len(solution1.representation)

            # cross the parents in the selected section
            for i in range(j, j2):
                offspring1.representation[i] = solution2.representation[i]
                offspring2.representation[i] = solution1.representation[i]

    return offspring1, offspring2

# -------------------------------------------------------------------------------------------------
# Order 1 crossover
# -------------------------------------------------------------------------------------------------
def order_1_crossover(problem, solution1, solution2):
    # get the representation (list) of both solutions
    solution_1 = solution1.representation
    solution_2 = solution2.representation

    # choose two different random points in the solutions
    point1 = randint( 0, len( solution_1 )-1 )
    point2 = point1

    while point1 == point2:
        point2 = randint( 0, len( solution_1 )-1 )

    # make sure that the first crossover point is the smallest and vice versa
    firstCrossPoint = min(point1,point2)
    secondCrossPoint = max(point1,point2)
    
    # save the middle section of each parent
    parent1MiddleCross = solution_1[firstCrossPoint:secondCrossPoint]
    parent2MiddleCross = solution_2[firstCrossPoint:secondCrossPoint]

    order_1 = []
    
    # we start saving the sequence after the middle section
    for i in solution_2[secondCrossPoint:]:
        # make sure we don't have repeated elements
        if i not in parent1MiddleCross:
            order_1.append(i)

    # then we go from the beginning up to where we started
    for i in solution_2[:secondCrossPoint]:
        # make sure we don't have repeated elements
        if i not in parent1MiddleCross:
            order_1.append(i)

    order_2 = []
    
    # we start saving the sequence after the middle section
    for i in solution_1[secondCrossPoint:]:
        # make sure we don't have repeated elements
        if i not in parent2MiddleCross:
            order_2.append(i)

    # then we go from the beginning up to where we started
    for i in solution_1[:secondCrossPoint]:
        # make sure we don't have repeated elements
        if i not in parent2MiddleCross:
            order_2.append(i)

    # create two empty children
    child1 = [None] * len(solution_1)
    child2 = [None] * len(solution_1)

    # first save the middle section
    for i in range(firstCrossPoint, secondCrossPoint):
        child1[i] = solution_1[i]
        child2[i] = solution_2[i]

    # then we start filling in from the end of the middle section
    j = 0
    for i in range(secondCrossPoint, len(solution_1)):
        child1[i] = order_1[j]
        child2[i] = order_2[j]

        j += 1
    
    # and then we fill in from the beginning up to the beginning of the middle section
    for i in range(0, firstCrossPoint):
        child1[i] = order_1[j]
        child2[i] = order_2[j]

        j += 1

    # save the children as LinearSolution objects
    child_1 = LinearSolution(child1, solution1.encoding_rule)
    child_2 = LinearSolution(child2, solution1.encoding_rule)

    return child_1,child_2

# -------------------------------------------------------------------------------------------------
# Uniform crossover
# -------------------------------------------------------------------------------------------------
def uniform_crossover( problem, solution1, solution2):
    # select n different random points
    n_points = []

    for i in range(0, len(solution1.representation)):
        n_points.append(choice([0,1]))

    # create the children
    offspring1 = deepcopy(solution1) #solution1.clone()
    offspring2 = deepcopy(solution2) #.clone()

    # if a point was selected cross the parents in that point
    for j in range(0, len(n_points)):
        if n_points[j] == 1:
            offspring1.representation[j] = solution2.representation[j]
            offspring2.representation[j] = solution1.representation[j]

    return offspring1, offspring2

###################################################################################################
# MUTATION APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Singlepoint mutation
# -----------------------------------------------------------------------------------------------
def single_point_mutation( problem, solution):
    singlepoint = randint( 0, len( solution.representation )-1 )
    #print(f" >> singlepoint: {singlepoint}")

    encoding    = problem.encoding

    if encoding.encoding_type == EncodingDataType.choices :
        try:
            temp = deepcopy( encoding.encoding_data )

            temp.pop( solution.representation[ singlepoint ] )

            gene = temp[0]
            if len(temp) > 1 : gene = choice( temp )  

            solution.representation[ singlepoint ] = gene

            return solution
        except:
            print('(!) Error: singlepoint mutation encoding.data issues)' )     

    # return solution           

# -------------------------------------------------------------------------------------------------
# Swap mutation
# -----------------------------------------------------------------------------------------------
#TODO: Implement Swap mutation
def swap_mutation( problem, solution):
    # choose two different random points in the solution
    point1 = randint( 0, len( solution.representation )-1 )
    point2 = point1

    while point1 == point2:
        point2 = randint( 0, len( solution.representation )-1 )
    #print(f" >> singlepoint: {singlepoint}")
     
    # save the points we chose
    sol_point1 = solution.representation[point1]
    sol_point2 = solution.representation[point2]

    # swap them
    solution.representation[point1] = sol_point2
    solution.representation[point2] = sol_point1

    return solution

# -------------------------------------------------------------------------------------------------
# Insert mutation
# -----------------------------------------------------------------------------------------------
def insert_mutation( problem, solution):
    # choose two different random points in the solution
    # point 1 is the index where point 2 will be inserted
    point1 = randint( 0, len( solution.representation )-1 )
    point2 = point1

    while point1 == point2:
        point2 = randint( 0, len( solution.representation )-1 )
    #print(f" >> singlepoint: {singlepoint}")

    # we split the solution in point 1 to insert point 2
    if point1 < point2:
        sol_1 = solution.representation[:point1]
        sol_2 = solution.representation[point1:]
    else:
        sol_1 = solution.representation[:point1+1]
        sol_2 = solution.representation[point1+1:]
    
    point = solution.representation[point2]
    
    # we remove point 2 to avoid repetition
    if point in sol_1:
        sol_1.remove( point )
    else:
        sol_2.remove( point )

    # we build the new solution
    solution.representation = sol_1 + [point] + sol_2

    return solution

# -------------------------------------------------------------------------------------------------
# Inversion mutation
# -----------------------------------------------------------------------------------------------
def inversion_mutation(problem, solution):
    # choose 2 random points in the solution
    point1 = randint( 0, len( solution.representation )-1 )
    point2 = point1

    while point1 == point2:
        point2 = randint( 0, len( solution.representation )-1 )
    #print(f" >> singlepoint: {singlepoint}")

    # guarantee that point 1 is the smallest and vice versa
    point_1 = min(point1,point2)
    point_2 = max(point1,point2)

    # define the middle section
    middle = solution.representation[point_1:point_2+1]

    # split the solution outside of the middle section
    sol_1 = solution.representation[:point_1]
    sol_2 = solution.representation[point_2+1:]

    # reverse the middle section
    middle.reverse()

    # build the new solution
    solution.representation = sol_1 + middle + sol_2

    return solution

# -------------------------------------------------------------------------------------------------
# Scramble mutation
# -----------------------------------------------------------------------------------------------
def scramble_mutation(problem, solution):
    # select 2 different random points in the solution
    point1 = randint( 0, len( solution.representation )-1 )
    point2 = point1

    while point1 == point2:
        point2 = randint( 0, len( solution.representation )-1 )
    #print(f" >> singlepoint: {singlepoint}")

    # guarantee that point 1 is the smallest and vice versa
    point_1 = min(point1,point2)
    point_2 = max(point1,point2)

    # save the middle section
    middle = solution.representation[point_1:point_2+1]

    # split the solution outside the middle section
    sol_1 = solution.representation[:point_1]
    sol_2 = solution.representation[point_2+1:]

    # scramble the middle section
    middle = sample(middle, len(middle))

    # build the new solution
    solution.representation = sol_1 + middle + sol_2

    return solution

# -------------------------------------------------------------------------------------------------
# Uniform mutation
# -----------------------------------------------------------------------------------------------
def uniform_mutation( problem, solution):
    # choose n random different points
    n_points = []

    for i in range(0, len(solution.representation)):
        n_points.append(choice([0,1]))
    
    encoding = problem.encoding

    # for each point we chose to change, switch it by another possible point in the encoding
    if encoding.encoding_type == EncodingDataType.choices :
        try:
            for i in range(0, len(n_points)):
                temp = deepcopy( encoding.encoding_data )
                n = n_points[i]

                if n == 1:
                    temp.pop( solution.representation[ i ] )

                    gene = temp[0]
                    if len(temp) > 1 : gene = choice( temp )  

                    solution.representation[ i ] = gene

            return solution
        except:
            print('(!) Error: singlepoint mutation encoding.data issues)' )

###################################################################################################
# REPLACEMENT APPROACHES
###################################################################################################
# -------------------------------------------------------------------------------------------------
# Standard replacement
# -----------------------------------------------------------------------------------------------
def standard_replacement(problem, current_population, new_population ):
    return deepcopy(new_population)

# -------------------------------------------------------------------------------------------------
# Elitism replacement
# -----------------------------------------------------------------------------------------------
def elitism_replacement(problem, current_population, new_population ):
    

    if problem.objective == ProblemObjective.Minimization :
        if current_population.fittest.fitness < new_population.fittest.fitness :
           new_population.solutions[0] = current_population.solutions[-1]
    
    elif problem.objective == ProblemObjective.Maximization : 
        if current_population.fittest.fitness > new_population.fittest.fitness :
           new_population.solutions[0] = current_population.solutions[-1]

    return deepcopy(new_population)