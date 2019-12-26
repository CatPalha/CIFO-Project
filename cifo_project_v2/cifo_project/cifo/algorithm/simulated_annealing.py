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
from cifo.util.observer import LocalSearchMessage

from random import randint, choice, uniform
import numpy as np
import math

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

    def __init__(self, problem_instance, neighborhood_function, feedback = None, params = {}):
        """
        Simulated Annealing Constructor

        Parameters:
        -----------
        ▶ problem_instance      - the instance of the problem that the algorithm will search a solution

        ▶ neighborhood_function - it is expected a function that must follow the signature:
           
            neighborhood_function( solution, problem, neighborhood_size = 0 )
        
        ▶ feedback 

        ▶ params                - dictionary with configurations for Simulated Annealing
            e.g.: params = { "Maximum-Internal-Iterations" : 5, "Maximum-Iterations" : 10, "Initial-C" : 1, "Minimum-C" : 0.01, "Update-Method" : "Geometric", "Update-Rate" : 0.9, "Neighborhood-Size" : -1}

            A. "Maximum-Internal-Iterations" - the number of maximum internal iterations
            
            B. "Maximum-Iterations" - the number of maximum iterations (used to stop the search, even there are neighbors better than the current solution)

            C. "Initial-C" - the initial value of the control parameter C

            D. "Minimum-C" - the value for the minimum value that C can hit

            E. "Update-Method" - the method that is used to update the control parameter
            Possible "Update-Methods" : ["Geometric", "Linear", "Logarithmic"]
            1. Geometric - Ci = Ci-1 * "Update-Rate", where i is the iteration number and the "Update-Rate" has to be a number between 0 and 1
            2. Linear - Ci = Ci-1 - "Update-Rate", where i is the iteration number
            3. Logarithmic - Ci = C0/log(i), where i is the iteration number and C0 is the initial value of C

            F. "Update-Rate" - the rate of change of C. If the "Update-Method" is "Geometric" then the "Update-Rate" has to be between 0 and 1. If the "Update-Method" is "Logarithmic" there's no need to define an "Update-Rate"
            
            G. "Initialize-Method-C" - the method that is used to initialize the control parameter. If "Initial-C" is already defined there's no need to define "Initialize-Method-C"
            Possible "Initialize-Method-C" : ["Classical", "Fitness Dependent"]
            1. Classical - C starts with the value 1
            2. Fitness Dependent - C starts equal to the fitness of the initial solution

            H. "Initialize-Method-Minimum-C" - the method that is used to initialize the minimum value that the control parameter can hit. If "Minimum-C" is already defined there's no need to define "Initialize-Method-Minimum-C"
            Possible "Initialize-Method-Minimum-C" : ["Classical", "Proportional"]
            1. Classical - minimum C starts with the value 0.01
            2. Proportional - minimum C starts as 0.01 * C

            I. "Neighborhood-Size" the size of the neighborhood, the default is -1, which means the neighborhood will return all neighbors found
        """
        # set
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        self._name               = "Simulated Annealing"
        self._description        = ""
        self._problem_instance   = problem_instance   
        self._objective          = problem_instance.objective
        self._get_neighbors      = neighborhood_function
        self._feedback           = feedback
        self._observers          = []
        self._internal_iteration = 0
        self._external_iteration = 0
        self._solution           = None
        self._neighbor           = None
        self._internal_searching = True
        self._external_searching = True
        self._best_solution      = None
        self._initial_c          = None

        # parse params
        # Default params: 
        # { "Maximum-Internal-Iterations" : 5, "Maximum-Iterations" : 10, "Update-Method" : "Geometric", "Initialize-Method-C" : "Classical", "Initialize-Method-Minimum-C" : "Classical", "Neighborhood-Size" : 0}
        # Motivation: Enables the user to change some Simulated Annealing Behaviors
        # (Flexibility)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        self._max_internal_iterations = 5
        if "Maximum-Internal-Iterations" in params:
            self._max_internal_iterations = params["Maximum-Internal-Iterations"]
            
        self._max_external_iterations = 10
        if "Maximum-Iterations" in params:
            self._max_external_iterations = params["Maximum-Iterations"]

        self._c = None
        if "Initial-C" in params:
            self._c = params["Initial-C"]

        self._min_c = None
        if "Minimum-C" in params:
            self._min_c = params["Minimum-C"]

        self._update_method = "Geometric"
        if "Update-Method" in params:
            self._update_method = params["Update-Method"]

        self._rate = None
        if "Update-Rate" in params:
            self._rate = params["Update-Rate"]

        self._initialize_method_c = "Classical"
        if "Initialize-Method-C" in params:
            self._initialize_method_c = params["Initialize-Method-C"]

        self._initialize_method_min_c = "Classical"
        if "Initialize-Method-Minimum-C" in params:
            self._initialize_method_min_c = params["Initialize-Method-Minimum-C"]

        self._neighborhood_size = 0
        if "Neighborhood-Size" in params:
            self._neighborhood_size = params["Neighborhood-Size"]

        # Prepare the internal methods for multi-objective / single-objective:
        # Motivation: Avoid in each selection step check if it is multi-single or min/max 
        # (Optimization)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if self._problem_instance.is_multi_objective:
            # Multi-objective
            print("NOT IMPLEMENTED.")

    # Search Method
    #----------------------------------------------------------------------------------------------    
    def search(self):
        """
        Simulated Annealing Search Method
        ----------------------------------

        Algorithm:

        1: Initialize
        2: Repeat while Control Parameter >= Minimum Control Parameter 
            2.1. Internal Looping
                2.1.1: Get a random neighbor
                2.1.2: Select the best, between the current best and current best neighbor
                2.1.3: Check stop condition ***
            2.2. Update C (Control Parameter)
            2.3: Check stop condition ***
        3: Return the Solution

        """

        self._notify( message = LocalSearchMessage.Started)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Search
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 1. Initialize
        self._initialize()
        self._notify(message = LocalSearchMessage.Initialized)

        self._external_searching = True
        self._external_iteration = 0

        while self._external_searching == True:
            self._external_iteration += 1

            # 2.1. Internal Looping
            self._internal_searching = True
            self._internal_iteration = 0
            
            while self._internal_searching == True:
                self._internal_iteration += 1
                # 2.1.1: Get a random neighbor
                self._get_random_neighbor()
                # 2.1.2: Select the best, between the current best and current best neighbor
                self._select()
                # 2.1.3: Check stop condition
                self._internal_stop_condition()
            
            # 2.2. Update C (Control Parameter)
            self._update_C()
            # 2.3. Check stop condition
            self._external_stop_condition()

        # 3: Return the Solution
        self._notify(message = "FINISHED")
        return self._best_solution

    # Initialize:  create an initial solution
    #----------------------------------------------------------------------------------------------
    def _initialize(self):
        """
        Create a feasible initial solution and initilize C and minimum C if they're not yet initialized
        """
        self._solution = self._problem_instance.build_solution()

        while self._problem_instance.is_admissible(self._solution) == False:
            self._solution = self._problem_instance.build_solution()

        self._best_solution = self._solution
        self._problem_instance.evaluate_solution(self._solution, feedback = self._feedback)

        # initialize C if it's not initialized
        if self._c is None:
            self._c = self._initialize_C()
            
        # save the initial C if the update method is logarithmic
        if self._update_method == "Logarithmic":
            self._initial_c = self._c

        # initialize minimum C if it's not initialized
        if self._min_c is None:
            self._min_c = self._initialize_min_C()

    # _get_random_neighbor:  get a random, admissible, neighbor
    #----------------------------------------------------------------------------------------------
    def _get_random_neighbor(self):
        """
        Get a random, admissible, neighbor of the neighborhood
        """
        # Get Neighbors of the current solution
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        neighborhood = self._get_neighbors(
            solution = self._solution,
            problem = self._problem_instance,
            neighborhood_size = self._neighborhood_size
            )

        
        # Select a random neighbor in neighborhood of the current solution
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        i = randint(0,len(neighborhood)-1)
        self._neighbor = neighborhood[i]

        while self._problem_instance.is_admissible(self._neighbor) == False:
            i = np.random.randint(0,len(neighborhood)-1)
            self._neighbor = neighborhood[i]

        self._problem_instance.evaluate_solution(self._neighbor, feedback = self._feedback)

    # _select:  select the better solution (or a worse one with a certain probability)
    #----------------------------------------------------------------------------------------------    
    def _select(self):
        """
        Select the better solution, or a worse one with a certain probability
        """
        if self._objective == "Maximization":
            if self._neighbor.fitness >= self._solution.fitness:
                self._solution = self._neighbor
                
                if self._solution.fitness >= self._best_solution.fitness:
                    self._best_solution = self._solution

            else:
                rand_number = uniform(0,1)
                prob = np.exp(-abs(self._neighbor.fitness-self._solution.fitness)/self._c)
                
                if rand_number <= prob:
                    self._solution = self._neighbor
        else:
            if self._neighbor.fitness <= self._solution.fitness:
                self._solution = self._neighbor

                if self._solution.fitness <= self._best_solution.fitness:
                    self._best_solution = self._solution

            else:
                rand_number = uniform(0,1)
                prob = np.exp(-abs(self._neighbor.fitness-self._solution.fitness)/self._c)
                
                if rand_number <= prob:
                    self._solution = self._neighbor

    def _internal_stop_condition(self):
        """
        Stops when the maximum number of internal iterations is achieved
        """
        if self._internal_iteration >= self._max_internal_iterations:
            self._internal_searching = False

    def _external_stop_condition(self):
        """
        Stops when the maximum number of external iterations is achieved or when C is smaller than the minimum C
        """
        if self._external_iteration >= self._max_external_iterations or self._c < self._min_c:
            self._external_searching = False

    def _update_C(self):
        """
        Updates C according to the method that was set
        """
        if self._update_method == "Geometric":
            self._c *= self._rate
        if self._update_method == "Linear":
            self._c -= self._rate
        elif self._update_method == "Logarithmic":
            self._c = self._initial_c / math.log(self._external_iteration+1)

    def _initialize_C(self):
        """
        Initializes C according to the method that was set
        """
        if self._initialize_method_c == "Classical":
            self._c = 1
        if self._initialize_method_c == "Fitness Dependent":
            self._c = self._solution.fitness

    def _initialize_min_C(self):
        """
        Initializes Minimum C according to the method that was set
        """
        if self._initialize_method_min_c == "Classical":
            self._min_c = 0.01
        if self._initialize_method_min_c == "Proportional":
            self._min_c = 0.01 * self._c

    # Problem Instance
    #----------------------------------------------------------------------------------------------
    @property
    def problem(self):
        return self._problem_instance
    
    # Solution
    #----------------------------------------------------------------------------------------------
    @property
    def solution(self):
        return self._solution

    # Name
    #----------------------------------------------------------------------------------------------
    @property
    def name(self):
        return self._name

    # Description
    #----------------------------------------------------------------------------------------------
    @property
    def description(self):
        return self._description

    #----------------------------------------------------------------------------------------------
    # Observable Interface
    #----------------------------------------------------------------------------------------------
    def get_state( self ):
        return self._state 

    #----------------------------------------------------------------------------------------------
    def register_observer(self, observer):
        self._observers.append( observer )

    #----------------------------------------------------------------------------------------------
    def unregister_observer(self, observer ):
        self._observers.remove( observer) 

    #----------------------------------------------------------------------------------------------
    def _notify( self, message, neighbors = None ):

        self._state = {
            "internal_iteration" : self._internal_iteration,
            "external_iteration" : self._external_iteration,
            "message"            : message,
            "neighbor"           : self._neighbor,
            "neighbors"          : neighbors
        }

        for observer in self._observers:
            observer.update()