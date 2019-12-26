from cifo.algorithm.hill_climbing import HillClimbing
from cifo.custom_problem.travel_salesman_problem import pip_bitflip_get_neighbors
from cifo.problem.population import Population

def initialize_using_hc( problem, population_size, params={} ):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm
    
    Required:
    
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    
    @ population_size - to define the size of the population to be returned. 

    @ params - the parameters to create the Hill Climbing object (default: empty dictionary)
    """
    solution_list = []

    hc = HillClimbing(
        problem_instance = problem,
        neighborhood_function = pip_bitflip_get_neighbors,
        params = params
        )

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
        solution_list = solution_list
        )
    
    return population