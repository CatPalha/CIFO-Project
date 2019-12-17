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
    "Can repeat"   : True,  # this doesn't really make sense
    "Data"         : [0,0], # the companies codes
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
        
        self._assets = pd.Series(index=self._symbols)
        if "Assets" in self._decision_variables:
            self._assets = self._decision_variables["Assets"]

        self._prices = pd.Series(index=self._symbols)
        if "Prices" in self._decision_variables:
            self._prices = self._decision_variables["Prices"]

        self._exp_return = pd.Series(index=self._symbols)
        if "Expected-Returns" in self._decision_variables:
            self._exp_return = self._decision_variables["Expected-Returns"]

        self._std = pd.Series(index=self._symbols)
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
        encoding_data = self._encoding.encoding_data
        solution_representation = dict.fromkeys(encoding_data, 0)
        randoms = uniform(size = self._encoding.size)
        weights = [random / sum(randoms) for random in randoms]

        for i in range(0, self._encoding.size):
            asset = choice(encoding_data)
            solution_representation[asset] += weights[i]        
        
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
        investment = 0
        exp_return = 0
        assets = list(solution.representation.keys())

        for asset in assets:
            investment += self._prices[asset]
            exp_return += self._exp_return[asset] * solution.representation[asset]
            
        weights_times_deviations = [solution.representation[asset]**2 * self._std[asset]**2 for asset in assets]
        variance = sum(weights_times_deviations)

        correlation = self._hist_data.corr()
        
        for asset1 in assets:
            for asset2 in assets:
                if asset1.index() < asset2.index():
                    aditional_variance =  solution.representation[asset1] * solution.representation[asset2] * self._std[asset1] * self._std[asset2] * correlation[asset1][asset2]
                    variance += aditional_variance

        standard_deviation = np.sqrt(variance)

        sharpe = exp_return / standard_deviation

        result = (investment <= self._budget and sharpe >= self._risk_tolerance)

        return result

    # Evaluate_solution()
    #-------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    def evaluate_solution(self, solution, feedback = None):# << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        """
        exp_return = 0
        assets = list(solution.representation.keys())

        for asset in assets:
            exp_return += self._exp_return[asset] * solution.representation[asset]
    
        solution.fitness = exp_return

        return solution


# -------------------------------------------------------------------------------------------------
# OPTIONAL - it onlu+y is needed if you will implement Local Search Methods
#            (Hill Climbing and Simulated Annealing)
# -------------------------------------------------------------------------------------------------
def pip_bitflip_get_neighbors( solution, problem, neighborhood_size = 0 ):
    assets = list(solution.representation.keys())
    neighborhood = []
    i = 0

    while i < neighborhood_size:
        neighbor = solution.copy()
        asset = choice(assets)
        neighbor.representation[asset] = uniform()
        neighborhood.append(neighbor)
        i += 1

    return solution