def is_admissible( self, solution ): #<< use this signature in the sub classes, the meta-heuristic 
    """
    Check if the solution is admissible, considering the no cities can be repeated
    """
    counts = [solution.representation.count(i) for i in solution.representation]
    
    repeated = False
    i = 0
    
    while repeated == False and i < len(counts):
        
        if counts[i] > 1:
            repeated = True
        
        i += 1
    
    result = not(repeated)

    return result