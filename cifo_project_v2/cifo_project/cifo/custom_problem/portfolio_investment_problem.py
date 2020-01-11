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
    The Portfolio Investment Problem looked to assemble a portfolio of assets such that the
    expected return was maximized, while the risk of this investment was minimized.
    """

    # Constructor
    #----------------------------------------------------------------------------------------------
    def __init__(self, decision_variables, constraints , encoding_rule = {}):
        """
        Portfolio Investment Problem CONSTRUCTOR
        
            Parameters:

            @decision_variables
            
            Expected Decision Variables, so the dictionary must have the following keys and values of them must be lists:
            
            e.g:
            
            df : Pandas DataFrame with the data

            decision_variables_example = {

                "Symbols"             : list(df['symbol']), #<< List, Mandatory - a list with the symbolic representation of each asset
                
                "Assets"              : list(df['name']), #<< List, Mandatory - a list with the name of each asset
                
                "Prices"              : list(df['price']), #<< List, Mandatory - a list with the price of each asset
                
                "Expected-Returns"    : list(df['exp_return_3m']), #<< List, Mandatory - a list with the expected return from the investment in each asset
                
                "Standard-Deviations" : list(df['standard_deviation']), #<< List, Mandatory - a list with the risk (standard deviation) that comes from investing in each asset
                
                "Historical-Data"     : df_hist #<< Pandas DataFrame, Mandatory - a Pandas DataFrame with the returns dating back to some predefined period
            }            
            
            @constraints
            
            The budget and risk tolerance of the portfolio

            e.g.:
            
            constraints = {

                "Risk-Tolerance" : 1, #<< Number, Mandatory - The lowest ratio between return and risk that we allowed the portfolio to have
                
                "Budget"         : 10000 #<< Number, Mandatory - The maximum amount of money that we could invest
            }

            @encoding_rule

            n_max = int(constraints["Budget"] // min(decision_variables_example["Prices"]))

            pip_encoding_rule = {

                "Size"         : -1, # It must be defined by the size of DV (Number of products)
                
                "Is ordered"   : True,
                
                "Can repeat"   : False,
                
                "Data"         : [i for i in range(0, n_max+1)],
                
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
        self._name = "Portfolio Investment Problem"
        
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
        Builds a linear solution for PIP with same size of decision variables integers.
        
        Where: 
            
            This integer represents the number of stocks we bought for that asset
        """
        solution_representation = []
        encoding_data = self._encoding.encoding_data
        randoms = []

        # assign random numbers between 0 and 1 to each asset
        for _ in range(0, self._encoding.size):
            randoms.append( uniform(0,1) )

        weights = []
        
        # get the investment weights for each asset based on the random numbers
        for random in randoms:
            weights.append( random / sum(randoms) )

        prices = self._prices
        budget = self._budget

        # calculate how much that investment weight costs related to our budget
        for i in range(0,len(weights)):
            investment = weights[i] * budget
            # divide the money we're investing by the price of the stocks to find out how many we can buy
            solution_representation.append( round(investment/prices[i]) )
        
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
        Check if the solution is admissible, considering the budget and risk tolerance
        """
        prices = self._prices

        # find how much we're investing with the portfolio
        price = 0
        for i in range(0, len( prices )):
            price += (prices[ i ] * solution.representation[i])

        # check if it's greater than our budget
        if price > self._budget:
            return False
        
        weights = []

        # find the investment weights
        for i in range(0, len( prices )):
            weights.append((prices[ i ] * solution.representation[ i ]) / price)
        
        returns = self._exp_return

        # calculate the expected return of the investment
        exp_return = sum([returns[ i ] * weights[ i ] for i in range(0, len( weights ))]) - 1.56

        std_deviations = self._std
        
        # calculate the expected risk of the investment
        variance = sum([weights[ i ]**2 * std_deviations[ i ]**2 for i in range(0, len( weights ))])

        correlation = self._hist_data.corr()
        
        for i in range(0, len( weights )):
            if weights[i] > 0:
                for j in range(i, len( weights )):
                    if weights[j] > 0:
                        aditional_variance =  2 * weights[i] * weights[j] * std_deviations[i] * std_deviations[j] * correlation.iloc[i,j]
                        variance += aditional_variance

        standard_deviation = np.sqrt(variance)

        # calculate the return / risk ratio
        sharpe = exp_return / standard_deviation
        
        result = (sharpe >= self._risk_tolerance)
        
        return result
        
    # Evaluate_solution()
    #-------------------------------------------------------------------------------------------------------------
    # It should be seen as an abstract method 
    def evaluate_solution(self, solution, feedback = None):# << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
        """
        Calculate the expected return of the portfolio
        """
        prices = self._prices

        # find how much we're investing
        price = 0
        for i in range(0, len( prices )):
            price += (prices[ i ] * solution.representation[i])
        
        weights = []

        # calculate the investment weights
        for i in range(0, len( prices )):
            weights.append((prices[ i ] * solution.representation[ i ]) / price)
        
        returns = self._exp_return

        # get the expected return
        fitness = sum([returns[ i ] * weights[ i ] for i in range(0, len( weights ))])
        
        solution.fitness = fitness

        return solution


# -------------------------------------------------------------------------------------------------
# OPTIONAL - it onlu+y is needed if you will implement Local Search Methods
#            (Hill Climbing and Simulated Annealing)
# -------------------------------------------------------------------------------------------------
def pip_bitflip_get_neighbors( solution, problem, neighborhood_size = 0 ):
    """
    The basic idea for the neighborhood function we defined was that a neighbor was equal to the
    original solution but for a random asset we either bought one more stock or one less stock.
    """

    neighbors = []

    # if the neighborhood size is -1 we get all neighbors
    if neighborhood_size == -1:
        admissible = False
        for i in range(0, len(solution.representation)):
            # copy the current solution
            neighbor1 = deepcopy(solution)
            neighbor2 = deepcopy(solution)

            mx = max(solution.encoding_rule["Data"])

            # if we reached the investment limit with this asset, we subtact 1
            if solution.representation[i] == mx:
                neighbor1.representation[i] = solution.representation[i] - 1

            # if the current investment is 0 for this asset, we add 1
            elif solution.representation[i] == 0:
                neighbor1.representation[i] = solution.representation[i] + 1
                
            # if neither of these limits is met, we create two neighbors
            else:
                neighbor1.representation[i] = solution.representation[i] + 1
                neighbor2.representation[i] = solution.representation[i] - 1
                
                # we need to check if the neighbor is admissible and not repeated
                # and that we have at least one admissible solution in the neighborhood
                if (admissible == False) and (problem.is_admissible(neighbor2)) and (neighbor2 not in neighbors):
                    neighbors.append(neighbor2)
                    admissible = True
                elif neighbor2 not in neighbors:
                    neighbors.append(neighbor2)

            # we need to check if the neighbor is admissible and not repeated
            # and that we have at least one admissible solution in the neighborhood
            if (admissible == False) and (problem.is_admissible(neighbor1)) and (neighbor1 not in neighbors):
                neighbors.append(neighbor1)
                admissible = True
            elif (neighbor1 not in neighbors) and (admissible == True):
                neighbors.append(neighbor1)


    else:
        admissible = False
        while len(neighbors) < neighborhood_size:
            # deep copy of solution.representation
            neighbor = deepcopy(solution)
            # choose a random asset
            i = randint(0, len(solution.representation)-1)
            mx = max(solution.encoding_rule["Data"])

            # if we reached the investment limit with this asset, we subtact 1
            if solution.representation[i] == mx:
                neighbor.representation[i] = solution.representation[i] - 1
            # if the current investment is 0 for this asset, we add 1
            elif solution.representation[i] == 0:
                neighbor.representation[i] = solution.representation[i] + 1
            # if neither of these limits is met, we randomly add or subtract 1
            else:
                op = choice([-1,1])
                neighbor.representation[i] = solution.representation[i] + op

            # we need to check if the neighbor is admissible and not repeated
            # and that we have at least one admissible solution in the neighborhood
            if (admissible == False) and (problem.is_admissible(neighbor)) and (neighbor not in neighbors):
                neighbors.append(neighbor)
                admissible = True
            elif (neighbor not in neighbors) and (admissible == True):
                neighbors.append(neighbor)

    return neighbors