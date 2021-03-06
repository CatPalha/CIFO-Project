from copy import deepcopy
from random import choice, randint

from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import LinearSolution, Encoding

tsp_encoding_rule = {
    "Size"         : -1, # It must be defined by the size of DV (Number of products)
    "Is ordered"   : True,
    "Can repeat"   : False,
    "Data"         : [0,0], # must be defined by the data
    "Data Type"    : ""
}


# REMARK: There is no constraint

# -------------------------------------------------------------------------------------------------
# TSP - Travel Salesman Problem
# -------------------------------------------------------------------------------------------------
class TravelSalesmanProblem( ProblemTemplate ):
    """
    Given a list of cities and the distances between each pair of cities,
    what is the shortest possible route that visits each city exactly once and returns to the origin city?
    """

    # Constructor
    #----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables, constraints = {} , encoding_rule = {}):
        """

            Travel Salesman Problem CONSTRUCTOR
        
            Parameters:

            @decision_variables
            
            Expected Decision Variables, so the dictionary must have the following keys and values of them must be lists:
            
            e.g:

            decision_variables_example = {

                "Distances" : data, #<< Matrix, Mandatory - the matrix containing the distances between all cities

                "Cities"    : [i for i in range(0, len(data))], #<< List, Mandatory - he list of all the cities we need to cross

            }
            
            @constraints
            
            The TSP has no constraints

            @encoding_rule

            tsp_encoding_rule = {
                
                "Size"         : -1, # This number must be redefined using the size of DV (Number of products contained in the instance of the problem)
                
                "Is ordered"   : FaTruelse,
                
                "Can repeat"   : False,
                
                "Data"         : [i for i in range(0, len(data))],
                
                "Data Type"    : "Choices"
            }

        """
        # optimize the access to the decision variables
        # ...

        # Call the Parent-class constructor to store these values and to execute  any other logic to be implemented by the constructor of the super-class
        super().__init__(
            decision_variables = decision_variables, 
            constraints = constraints, 
            encoding_rule = encoding_rule
        )

        # 1. Define the Name of the Problem
        self._name = "Travel Salesman Problem"
        
        # 2. Define the Problem Objective
        self._objective = ProblemObjective.Minimization

        # optimize the access to the decision variables
        # this was added by us
        self._distances = []
        if "Distances" in decision_variables:
            self._distances = decision_variables["Distances"]

        self._cities = []
        if "Cities" in decision_variables:
            self._weights = decision_variables["Cities"]

    # Build Solution for TSP
    #----------------------------------------------------------------------------------------------
    def build_solution(self):
        """
        Builds a linear solution for TSP that is an ordered list of numbers, with no repetitions
        
        Where: 
            
            each number i corresponds to the city of index i in the distance matrix
        """
        solution_representation = []
        encoding_data = self._encoding.encoding_data[:]

        for _ in range(0, self._encoding.size):
            # choose a random city
            city = choice(encoding_data)
            solution_representation.append( city )
            # remove it from the list of possibilities
            encoding_data.remove(city)
        
        # create a LinearSolution object
        solution = LinearSolution(
            representation = solution_representation, 
            encoding_rule = self._encoding_rule
        )
        
        return solution

    # Solution Admissibility Function - is_admissible()
    #----------------------------------------------------------------------------------------------
    def is_admissible( self, solution ): #<< use this signature in the sub classes, the meta-heuristic 
        """
        Check if the solution is admissible, considering the no cities can be repeated
        """
        solution_representation = list(solution.representation)
        
        counts = [solution_representation.count(i) for i in solution_representation]
        
        repeated = False
        i = 0
        
        while repeated == False and i < len(counts):
            
            if counts[i] > 1:
                repeated = True
            
            i += 1
        
        result = not(repeated)

        return result

    # Evaluate_solution()
    #-------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    def evaluate_solution(self, solution, feedback = None):# << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        Calculate the "distance" that is crossed in the solution
        """
        distances = self._distances
        solution_representation = list(solution.representation)

        fitness = 0
    
        for city in solution_representation:
            i = solution_representation.index(city)
            if i < len(solution_representation)-1:
                city2 = solution_representation[i+1]
            else:
                city2 = solution_representation[0]

            dist = distances[city][city2]

            fitness += dist
    
        solution.fitness = fitness

        return solution


# -------------------------------------------------------------------------------------------------
# OPTIONAL - it onlu+y is needed if you will implement Local Search Methods
#            (Hill Climbing and Simulated Annealing)
# -------------------------------------------------------------------------------------------------
def tsp_bitflip_get_neighbors( solution, problem, neighborhood_size = 0 ):
    neighborhood = []
    
    # if the neighborhood size is -1 we get all neighbors
    if neighborhood_size == -1:
        for i in range(0, len(solution.representation)):
            for j in range(0, len(solution.representation)):
                if i != j:
                    # swap two different cities
                    neighbor = solution.representation[:]
                    neighbor[i] = solution.representation[j]
                    neighbor[j] = solution.representation[i]
                    
                    # check if this neighbor is not repeated
                    if neighbor not in neighborhood:
                        neighborhood.append(neighbor)
    else:
        while len(neighborhood) < neighborhood_size:
            # swap two different random cities
            i = randint(0, len(solution.representation)-1)
            j = randint(0, len(solution.representation)-1)
            
            while i == j:
                j = randint(0, len(solution.representation)-1)
            
            # deep copy of solution
            neighbor = solution.representation[:]
            neighbor[i] = solution.representation[j]
            neighbor[j] = solution.representation[i]
            
            # check if this neighbor is not repeated
            if neighbor not in neighborhood:
                neighborhood.append(neighbor)

    neighbors = []
    
    # create LinearSolution objects
    for neighbor in neighborhood:
        neigh = LinearSolution(neighbor, solution.encoding_rule)
        neighbors.append(neigh)
    
    return neighbors