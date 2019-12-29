from cifo.algorithm.tabu_search import TabuSearch
from cifo.problem.population import Population

def initialize_using_ts( problem, population_size, params={} ):
    """
    Initialize a population of solutions (feasible solution) for an evolutionary algorithm
    
    Required:
    
    @ problem - problem's build solution function knows how to create an individual in accordance with the encoding.
    
    @ population_size - to define the size of the population to be returned. 

    @ params - the parameters to create the Tabu Search object (default: empty dictionary)
    """
    solution_list = []

    # we need to do this since the neighborhood functions are specific to the type of problem and they're not
    # part of the problem class
    if problem.name == "Travel Salesman Problem":
        from cifo.custom_problem.travel_salesman_problem import pip_bitflip_get_neighbors
    elif problem.name == "Portfolio Investment Problem":
        from cifo.custom_problem.portfolio_investment_problem import pip_bitflip_get_neighbors

    ts = TabuSearch(
        problem_instance = problem,
        neighborhood_function = pip_bitflip_get_neighbors,
        params = params
        )

    # generate a population of admissible solutions (individuals)
    for _ in range(0, population_size):
        s = ts.search()
        
        # check if the solution is admissible
        while not problem.is_admissible( s ):
            s = ts.search()
        
        problem.evaluate_solution ( s )
        
        solution_list.append( s )

    population = Population( 
        problem = problem , 
        maximum_size = population_size, 
        solution_list = solution_list
        )
    
    return population