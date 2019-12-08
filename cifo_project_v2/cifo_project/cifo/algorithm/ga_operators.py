from random import uniform, randint, choices
from copy import deepcopy

from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import EncodingDataType
from cifo.problem.population import Population

import numpy as np

###################################################################################################
# INITIALIZATION APPROACHES
###################################################################################################

# (!) REMARK:
# Initialization signature: <method_name>( problem, population_size ):

# -------------------------------------------------------------------------------------------------
# Random Initialization 
# -------------------------------------------------------------------------------------------------
def initialize_randomly( problem, population_size ):
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
def initialize_using_hc( problem, population_size ):
    pass

# -------------------------------------------------------------------------------------------------
# Initialization using Simulated Annealing
# -------------------------------------------------------------------------------------------------
#TODO: OPTIONAL, implement a initialization based on Hill Climbing
# Remark: remember, you will need a neighborhood functin for each problem
def initialize_using_sa( problem, population_size ):
    pass

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
        index1 = self._select_index(population = population)
        index2 = index1
        
        while index2 == index1:
            index2 = self._select_index( population = population )

        return population.get( index1 ), population.get( index2 )

    #we added objective as an argument
    def _select_index(self, population ):
        
        #this is the part we added, definition of minimization.
        if objective == 'Minimization':
            fit_max = population.fittest

            for solution in population.solutions:
                solution.fitness = fit_max - solution.fitness
        
        # Get the Total Fitness (all solutions in the population) to calculate the chances proportional to fitness
        total_fitness = 0
        for solution in population.solutions:
            total_fitness += solution.fitness

        # spin the wheel
        wheel_position = uniform( 0, 1 )

        # calculate the position which wheel should stop
        stop_position = 0
        index = 0
        for solution in population.solutions :
            stop_position += (solution.fitness / total_fitness)
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

        for index in range(0, len(population)):
            for _ in range(0, index + 1):
                rank_list.append( index )

        print(f" >> rank_list: {rank_list}")       

        # Step 3: Select solution index
        index1 = randint(0, len( rank_list )-1)
        index2 = index1
        
        while index2 == index1:
            index2 = randint(0, len( rank_list )-1)

        return population.get( rank_list [index1] ), population.get( rank_list[index2] )

    
    def _sort( self, population, objective ):

        if objective == ProblemObjective.Maximization:
            for i in range (0, len( population )):
                for j in range (i, len (population )):
                    if population.solutions[ i ].fitness > population.solutions[ j ].fitness:
                        swap = population.solutions[ j ]
                        population.solutions[ j ] = population.solutions[ i ]
                        population.solutions[ i ] = swap
                        
        else:    
            for i in range (0, len( population )):
                for j in range (i, len (population )):
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
    firstCrossPoint = np.random.randint(0,len(solution1)-2)
    secondCrossPoint = np.random.randint(firstCrossPoint+1,len(solution1)-1)
    
    parent1MiddleCross = solution1[firstCrossPoint:secondCrossPoint]
    parent2MiddleCross = solution2[firstCrossPoint:secondCrossPoint]

    child1 = solution1[:firstCrossPoint] + parent2MiddleCross + solution1[secondCrossPoint:]
    child2 = solution2[:firstCrossPoint] + parent1MiddleCross + solution2[secondCrossPoint:]

    relations = []

    for i in range(len(parent1MiddleCross)):
        relations.append([parent2MiddleCross[i], parent1MiddleCross[i]])

    counts1 = [child1.count(i) for i in child1]
    counts2 = [child2.count(i) for i in child2]

    while len([x for x in counts1 if x > 1]) > 0:
        for i in child1[:firstCrossPoint]:
            for j in parent2MiddleCross:
                if i == j:
                    index_j = parent2MiddleCross.index(j)
                    relation = relations[index_j]
                    index_i = child1.index(i)
                    child1[index_i] = relation[1]
        
        for i in child1[secondCrossPoint:]:
            for j in parent2MiddleCross:
                if i == j:
                    index_j = parent2MiddleCross.index(j)
                    relation = relations[index_j]
                    index_i = child1.index(i,secondCrossPoint)
                    child1[index_i] = relation[1]

        counts1 = [child1.count(i) for i in child1]

    while len([x for x in counts2 if x > 1]) > 0:
        for i in child2[:firstCrossPoint]:
            for j in parent1MiddleCross:
                if i == j:
                    index_j = parent1MiddleCross.index(j)
                    relation = relations[index_j]
                    index_i = child2.index(i)
                    child2[index_i] = relation[0]
        
        for i in child2[secondCrossPoint:]:
            for j in parent1MiddleCross:
                if i == j:
                    index_j = parent1MiddleCross.index(j)
                    relation = relations[index_j]
                    index_i = child2.index(i,secondCrossPoint)
                    child2[index_i] = relation[0]

        counts2 = [child2.count(i) for i in child2]

    return child1, child2

# -------------------------------------------------------------------------------------------------
# Cycle Crossover
# -------------------------------------------------------------------------------------------------
# TODO: implement Cycle Crossover
def cycle_crossover( problem, solution1, solution2):
    cycles = []
    considered = []

    # finding the cycles
    while len(considered) < len(solution1):
        i = 1

        while i in considered:
            i += 1
        
        cycle =  []
        full_cycle = False

        while full_cycle == False:
            cycle = cycle.append(i)
            considered = considered.append(i)
            i = solution1.index(solution2[i])

            if i in considered:
                full_cycle = True

        cycles = cycles.append(cycle)
    
    child1 =  [None] * len(solution1)
    child2 =  [None] * len(solution1)
    
    # getting the children
    for i, cycle in enumerate(cycles):
        # note that here cycle 1 is the cycle with index 0
        if i % 2 == 0:
            child1[i] = solution1[i]
            child2[i] = solution2[i]
        else:
            child1[i] = solution2[i]
            child2[i] = solution1[i]

    return child1, child2

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
            if len(temp) > 1 : gene = choices( temp )  

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
    point1 = randint( 0, len( solution.representation )-1 )
    point2 = point1

    while point1 == point2:
        point2 = randint( 0, len( solution.representation )-1 )
    #print(f" >> singlepoint: {singlepoint}")
     
    sol_point1 = solution[point1]
    sol_point2 = solution[point2]

    solution[point1] = sol_point2
    solution[point2] = sol_point1

    return solution

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