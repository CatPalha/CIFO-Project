def cycle_crossover(solution1, solution2):

    cycles = []
    #
    considered = []
    
    # finding the cycles
    while len(considered) < len(solution1):
        i = 0

        while i in considered:
            i += 1
        
        cycle =  []
        full_cycle = False

        while full_cycle == False:
            print(i)
            #is not appending
            cycle.append(i)
            #is not appending

            considered.append(i)
            i = solution1.index(solution2[i])

            if i in considered:
                full_cycle = True

        cycles.append(cycle)
    
    child1 =  [None] * len(solution1)
    child2 =  [None] * len(solution1)
    
    # getting the children
    print(cycles)
    #for j in range(0, len(cycles)):

    for i, cycle in enumerate(cycles):
        # note that here cycle 1 is the cycle with index 0
        if i % 2 == 0:
            for j in cycle:
                child1[j] = solution1[j]
                child2[j] = solution2[j]
        else:
            for j in cycle:
                child1[j] = solution2[j]
                child2[j] = solution1[j]

    return child1, child2


p1 = [8, 4, 7, 3, 6, 2, 5, 1 ,9, 0]
p2 = [0, 1, 2, 3 ,4 ,5 ,6, 7, 8, 9]

b = cycle_crossover(p1, p2)
print(b)

# Child 1:  8 1 2 3 4 5 6 7 9 0
# Child 2:  0 4 7 3 6 2 5 1 8 9 