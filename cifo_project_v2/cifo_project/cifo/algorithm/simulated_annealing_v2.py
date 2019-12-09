# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------
"""
Simulated Annealing Meta-Heuristic
----------------------------------

Content: 

▶ class Simulated Annealing

─────────────────────────────────────────────────────────────────────────

CIFO - Computation Intelligence for Optimization

Author: Fernando A J Peres - fperes@novaims.unl.pt - (2019) version L4.0
"""
# -------------------------------------------------------------------------------------------------

# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/
# C O D E
# /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/


# -------------------------------------------------------------------------------------------------
# Class: Simulated Annealing
# -------------------------------------------------------------------------------------------------

from cifo.problem.problem_template import ProblemTemplate
from cifo.problem.objective import ProblemObjective
from cifo.util.observer            import LocalSearchMessage

class SimulatedAnnealing:
    """
    Classic Implementation of Simulated Annealing with some improvements.

    Improvements:
    ------------
    1. Memory - avoiding to lose the best
    2. C / Minimum C Calibration

    Algorithm:
    ---------
    1: Initialize
    2: Repeat while Control Parameter >= Minimum Control Parameter 
        2.1. Internal Looping
            2.1.1: Get the best neighbor
            2.1.2: Select the best, between the current best and current best neighbor
            2.1.3: Check stop condition ***
        2.2. Update C (Control Parameter)
        2.3: Check stop condition ***
    3: Return the Solution
    """

    # Constructor
    #----------------------------------------------------------------------------------------------
    def __init__(self, problem_instance, neighborhood_function, feedback = None, config = {}):
        
        """
        Simulated Annealing Constructor

        Parameters:
        -----------
        ▶ problem_instance     - the instance of the problem that the algorithm will search a solution

        ▶ neighborhood_function - it is expected a function that must follow the signature:
           
            neighborhood_function( solution, problem, neighborhood_size = 0 )
        
        ▶ feedback 

        ▶ config - dictionary with configurations for Simulated Annealing
            e.g.: params = { "Maximum-Iterations" : 100 , "Internal-Loop" : 5, "Neighborhood-Size": -1, "Initialize-C" : "Classical"}

            A. "Maximum-Iterations" - the number of maximum iterations (used to stop the search, even there are neighbors better than the current solution)
            
            B. "Internal-Loop" - the number of iterations that the internal loop runs before updating C

            C. "Neighborhood-Size" - the size of the neighborhood, the default is -1, which means the neighborhood will return all neighbors found

            D. "Initialize-C" - the approach used to initialize the control parameter C
        """
        # set
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        self._name             = "Simulated Annealing"
        self._description      = ""
        self._problem_instance = problem_instance
        self._get_neighbors    = neighborhood_function
        self._feedback         = feedback
        self._observers        = []
        self._iteration        = 0
        self._solution = None
        self._neighbor = None

        # parse params
        # Default params: 
        # { "Maximum-Iterations" : 10 , "Internal-Loop" : 5, "Target-Fitness" : None, "Neighborhood-Size": 0, "Initialize-C" : "Classical"}
        # Motivation: Enables the user to change some Simulated Annealing Behaviors
        # (Flexibility)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        # max iterations
        self._max_iterations = 10
        if "Maximum-Iterations" in params: 
            self._max_iterations = params["Maximum-Iterations"]
        
        # internal loop iterations
        self._internal_loop = 10
        if "Internal-Loop" in params: 
            self._internal_loop = params["Internal-Loop"]
        
        # target fitness
        self._target_fitness = None
        if "Target-Fitness" in params: self._target_fitness = params["Target-Fitness" ]

        # neighborhood size
        self._neighborhood_size = 0
        if "Neighborhood-Size" in params: 
            self._neighborhood_size = params["Neighborhood-Size" ]
        
        # initialize C
        self._initialize_C_approach = "Classical"
        if "Initialize-C" in params: 
            self._initialize_C_approach = params["Initialize-C" ]

        self._description = f"Maximum-Iterations: {self._max_iterations} | Internal-Loop: {self._internal_loop} | Initialize-C: {self._initialize_C_approach}"

        # memory (to avoid lost the best)
        self._best_solution     = None

        # Prepare the internal methods for multi-objective / single-objective:
        # Motivation: Avoid in each selection step check if it is multi-single or min/max 
        # (Optimization)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if self._problem_instance.is_multi_objective:
            # Multi-objective
            print("NOT IMPLEMENTED.")
        else: 
            # Single-Objective
            if self._problem_instance.objective == ProblemObjective.Maximization:
                self._get_best_neighbor = self._get_best_neighbor_maximization
                self._select = self._select_maximization
            else:
                self._get_best_neighbor = self._get_best_neighbor_minimization
                self._select = self._select_minimization

    # Search
    #----------------------------------------------------------------------------------------------
    def search(self):
        """
        Simulated Annealing Search Method
        ----------------------------------

        Algorithm:

        1: Initialize
        2: Repeat while Control Parameter >= Minimum Control Parameter 
            2.1. Internal Looping
                2.1.1: Get the best neighbor
                2.1.2: Select the best, between the current best and current best neighbor
                2.1.3: Check stop condition ***
            2.2. Update C (Control Parameter)
            2.3: Check stop condition ***
        3: Return the Solution

        """
        self._notify( message = LocalSearchMessage.Started)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        searching = True
        self._solution = None
        self._neighbor = None

        # Search
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 1. Initialize
        self._initialize()
        self._notify(message = LocalSearchMessage.Initialized)

        # 2. Repeat while Control Parameter >= Minimum Control Parameter
        c = self._initialize_C(self._initialize_C_approach)
        self._iteration = 0
        while searching: #self._min_C needs to be defined in _initialize_C
            self._iteration += 1
            searching_internal = True
            # 2.1: Internal Looping
            self._internal_iteration = 0
            while searching_internal:
                self._internal_iteration += 1
                # 2.1.1: Get the best neighbor
                self._get_best_neighbor()
                # 2.1.2: Select the best, between the current best and best neighbor
                changed = self._select()
                # 2.1.3: Check stop conditions
                searching_internal = self._check_internal_stop_conditions
            # 2.2: Update C
            c = self._update_c(self._initialize_C_approach) #self._update_C needs to be defined
            # 2.3: Check stop conditions
            searching = self._check_stop_conditions
        
        #3: Return the best solution
        self._notify(message = "FINISHED")
        return self._solution

    # Constructor
    #----------------------------------------------------------------------------------------------
    def _initialize(self):
        """
        Initialize the initial solution, start C and Minimum C
        """
        pass

    # Constructor
    #----------------------------------------------------------------------------------------------
    def _select(self): 
        """
        Select the solution for the next iteration
        """
        pass

    # Constructor
    #----------------------------------------------------------------------------------------------
    def _get_random_neighbor(self, solution):
        """
        Get a random neighbor of the neighborhood (internally it will call the neighborhood provided to the algorithm)
        """
        pass
    
        # Constructor
    #----------------------------------------------------------------------------------------------
    def _initialize_C(self):
        """
        Use one of the available approaches to initialize C and Minimum C
        """
        pass

# -------------------------------------------------------------------------------------------------
# Class: Simulated Annealing
# -------------------------------------------------------------------------------------------------
def initialize_C_approach1():
    return 0, 0