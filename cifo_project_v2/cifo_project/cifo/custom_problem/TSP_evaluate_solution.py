def evaluate_solution(self, solution, feedback = None):# << This method does not need to be extended, it already automated solutions evaluation, for Single-Objective and for Multi-Objective
    """
    Calculate the "distance" that is crossed in the solution
    """
    distances = self._distances

    fitness = 0
    
    for city in solution:
        i = solution.index(city)
        if i < len(solution)-1:
            city2 = solution[i+1]
        else:
            city2 = solution[0]

        dist = distances[city][city2]

        fitness += dist
    
    solution.fitness = fitness

    return solution