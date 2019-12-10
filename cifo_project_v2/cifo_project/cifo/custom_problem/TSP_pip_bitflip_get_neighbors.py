
from random import randint

def pip_bitflip_get_neighbors(solution, problem = {}, neighborhood_size = 0):
    neighborhood = []
    
    if neighborhood_size == -1:
        for i in range(0, len(solution)):
            for j in range(0, len(solution)):
                if i != j:
                    neighbor = solution[:]
                    neighbor[i] = solution[j]
                    neighbor[j] = solution[i]
                    
                    if neighbor not in neighborhood:
                        neighborhood.append(neighbor)
    else:
        while len(neighborhood) < neighborhood_size:
            i = randint(0, len(solution)-1)
            j = randint(0, len(solution)-1)
            
            while i == j:
                j = randint(0, len(solution)-1)
            
            # deep copy of solution
            neighbor = solution[:]
            neighbor[i] = solution[j]
            neighbor[j] = solution[i]
            
            if neighbor not in neighborhood:
                neighborhood.append(neighbor)
    
    return neighborhood