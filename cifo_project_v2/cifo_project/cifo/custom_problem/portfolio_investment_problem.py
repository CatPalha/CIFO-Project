from copy import deepcopy
from random import choice, randint, uniform

from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.problem.solution import LinearSolution, Encoding

import pandas as pd
import numpy as np

pip_encoding_rule = {
    "Size"         : -1, # It must be defined by the size of DV (Number of products)
    "Is ordered"   : False,
    "Can repeat"   : True,
    "Data"         : [0,0], # an interval of integers where the largest is 'budget'//'cheapest stock' 
    "Data Type"    : "" # "Choices"
}

pip_constraints_example = {
    "Risk-Tolerance" : 1,
    "Budget": 10000
}

# -------------------------------------------------------------------------------------------------
# PIP - Portfolio Investment Problem 
# -------------------------------------------------------------------------------------------------
class PortfolioInvestmentProblem( ProblemTemplate ):
    """
    """

    # Constructor
    #----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables, constraints , encoding_rule = {}):
        """
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
        self._name = "Problem Name"
        
        # 2. Define the Problem Objective
        self._objective = ProblemObjective.Maximization

        self._symbols = []
        if "Symbols" in self._decision_variables:
            self._symbols = self._decision_variables["Symbols"]
        
        self._assets = []
        if "Assets" in self._decision_variables:
            self._assets = self._decision_variables["Assets"]

        self._prices = []
        if "Prices" in self._decision_variables:
            self._prices = self._decision_variables["Prices"]

        self._exp_return = []
        if "Expected-Returns" in self._decision_variables:
            self._exp_return = self._decision_variables["Expected-Returns"]

        self._std = []
        if "Standard-Deviations" in self._decision_variables:
            self._std = self._decision_variables["Standard-Deviations"]

        self._hist_data = pd.DataFrame(columns = self._symbols)
        if "Historical-Data" in self._decision_variables:
            self._hist_data = self._decision_variables["Historical-Data"]

        self._risk_tolerance = 1
        if "Risk-Tolerance" in self._constraints:
            self._risk_tolerance = self._constraints["Risk-Tolerance"]

        self._budget = 10000
        if "Budget" in self._constraints:
            self._budget = self._constraints["Budget"]

    # Build Solution for PIP
    #----------------------------------------------------------------------------------------------
    def build_solution(self):
        """
        """
        solution_representation = []
        encoding_data = self._encoding.encoding_data

        for _ in range(0, self._encoding.size):
            solution_representation.append( choice(encoding_data) )
        
        solution = LinearSolution(
            representation = solution_representation, 
            encoding_rule = self._encoding_rule
        )
        
        return solution

    # Solution Admissibility Function - is_admissible()
    #----------------------------------------------------------------------------------------------
    def is_admissible( self, solution ): #<< use this signature in the sub classes, the meta-heuristic 
        """
        """
        prices = self._prices

        price = 0
        for i in range(0, len( prices )):
            price += (prices[ i ] * solution.representation[i])

        if price > self._budget:
            return False
        
        weights = []

        for i in range(0, len( prices )):
            weights[ i ] = (prices[ i ] * solution.representation[ i ]) / price
        
        returns = self._exp_return

        exp_return = sum([returns[ i ] * weights[ i ] for i in range(0, len( weights ))])

        std_deviations = self._std
        
        variance = sum([weights[ i ]**2 * std_deviations[ i ]**2 for i in range(0, len( weights ))])

        correlation = self._hist_data.corr()
        
        for i in range(0, len( weights )):
            for j in range(i, len( weights )):
                aditional_variance =  weights[i] * weights[j] * std_deviations[i] * std_deviations[j] * correlationiloc[i,j]
                variance += aditional_variance

        standard_deviation = np.sqrt(variance)

        sharpe = exp_return / standard_deviation
        
        result = (sharpe >= self._risk_tolerance)
        
        return result
        
    # Evaluate_solution()
    #-------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    def evaluate_solution(self, solution, feedback = None):# << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        """
        weights = []

        for i in range(0, len( prices )):
            weights[ i ] = (prices[ i ] * solution.representation[ i ]) / price
        
        returns = self._exp_return

        fitness = sum([returns[ i ] * weights[ i ] for i in range(0, len( weights ))])
        
        solution.fitness = fitness

        return solution


# -------------------------------------------------------------------------------------------------
# OPTIONAL - it onlu+y is needed if you will implement Local Search Methods
#            (Hill Climbing and Simulated Annealing)
# -------------------------------------------------------------------------------------------------
def pip_bitflip_get_neighbors( solution, problem, neighborhood_size = 0 ):
    neighbors = []

    while len(neighbors) < neighborhood_size:
        # deep copy of solution.representation
        neighbor = solution.representation[:]
        i = randint(0, len(solution)-1)
        mx = max(solution.encoding_rule.encoding_data)

        if solution.representation[i] == mx:
            neighbor[i] = solution.representation[i] - 1
        elif solution.representation[i] == 0:
            neighbor[i] = solution.representation[i] + 1
        else:
            op = choice([-1,1])
            neighbor[i] = solution.representation[i] + op

        if neighbor not in neighbors:
            neighbors.append(neighbor)

    neighborhood = []
    
    for neighbor in neighbors:
        neigh = LinearSolution(neighbor, solution.encoding_rule)
        neighborhood.append(neigh)

    return neighborhood



