#import HillClimbing somehow

def initialize_using_hc( problem, population_size ):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm
    
    Required:
    
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    
    @ population_size - to define the size of the population to be returned. 
    """
    solution_list = []
    params = { "Maximum-Iterations" : 100 , "Stop-Conditions" : "Classical", "Neighborhood-Size": -1}

    hc = HillClimbing(problem, problem.neighborhood_function, params)

    # generate a population of admissible solutions (individuals)
    for _ in range(0, population_size):
        s = hc.search()
        
        # check if the solution is admissible
        while not problem.is_admissible( s ):
            s = hc.search()
        
        problem.evaluate_solution ( s )
        
        solution_list.append( s )

    population = Population( 
        problem = problem , 
        maximum_size = population_size, 
        solution_list = solution_list )
    
    return population