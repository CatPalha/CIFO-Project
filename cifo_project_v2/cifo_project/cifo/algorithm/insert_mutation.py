from random import randint

def insert_mutation( solution):
    # choose two different random points in the solution
    #point1 = randint( 0, len( solution.representation )-1 )
    point1 = randint( 0, len( solution )-1 )
    point2 = point1

    while point1 == point2:
        point2 = randint( 0, len( solution )-1 )
    #print(f" >> singlepoint: {singlepoint}")

    print(point1)
    print(point2)

    # guarantee that point 1 is the smallest and vice versa
    #point_1 = min(point1,point2)
    #point_2 = max(point1,point2)
    
    #sol_1 = solution.representation[:point_1+1]
    #sol_2 = solution.representation[point_1+1:]
    #point = solution.representation[point_2]
    
    if point1 < point2:
        sol_1 = solution[:point1]
        sol_2 = solution[point1:]
    else:
        sol_1 = solution[:point1+1]
        sol_2 = solution[point1+1:]

    point = solution[point2]

    if point in sol_1:
        sol_1.remove( point )
    else:
        sol_2.remove( point )

    solution = sol_1 + [point] + sol_2

    return solution

s = [0,1,2,3,4,5,6,7,8,9]

print(insert_mutation(s))