import numpy as np

def pmx_crossover(solution1, solution2):
    firstCrossPoint = np.random.randint(0,len(solution1)-2)
    secondCrossPoint = np.random.randint(firstCrossPoint+1,len(solution1)-1)

    #firstCrossPoint = 1
    #secondCrossPoint = 8
    
    parent1MiddleCross = solution1[firstCrossPoint:secondCrossPoint]
    parent2MiddleCross = solution2[firstCrossPoint:secondCrossPoint]

    child1 = solution1[:firstCrossPoint] + parent2MiddleCross + solution1[secondCrossPoint:]
    child2 = solution2[:firstCrossPoint] + parent1MiddleCross + solution2[secondCrossPoint:]

    relations = []

    for i in range(len(parent1MiddleCross)):
        relations.append([parent2MiddleCross[i], parent1MiddleCross[i]])

    counts1 = [child1.count(i) for i in child1]
    counts2 = [child2.count(i) for i in child2]

    while len([x for x in counts1 if x > 1]) > 0:
        for i in child1[:firstCrossPoint]:
            for j in parent2MiddleCross:
                if i == j:
                    index_j = parent2MiddleCross.index(j)
                    relation = relations[index_j]
                    index_i = child1.index(i)
                    child1[index_i] = relation[1]
        
        for i in child1[secondCrossPoint:]:
            for j in parent2MiddleCross:
                if i == j:
                    index_j = parent2MiddleCross.index(j)
                    relation = relations[index_j]
                    index_i = child1.index(i,secondCrossPoint)
                    child1[index_i] = relation[1]

        counts1 = [child1.count(i) for i in child1]

    while len([x for x in counts2 if x > 1]) > 0:
        for i in child2[:firstCrossPoint]:
            for j in parent1MiddleCross:
                if i == j:
                    index_j = parent1MiddleCross.index(j)
                    relation = relations[index_j]
                    index_i = child2.index(i)
                    child2[index_i] = relation[0]
        
        for i in child2[secondCrossPoint:]:
            for j in parent1MiddleCross:
                if i == j:
                    index_j = parent1MiddleCross.index(j)
                    relation = relations[index_j]
                    index_i = child2.index(i,secondCrossPoint)
                    child2[index_i] = relation[0]

        counts2 = [child2.count(i) for i in child2]

    return child1, child2

print(pmx_crossover([0,1,2,3,4,5,6,7,8,9],[3,1,2,8,4,7,0,9,6,5]))