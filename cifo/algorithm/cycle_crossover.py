parent1 = [8, 4, 7, 3, 6, 2, 5, 1, 9, 0]
parent2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

cycles = [-1]*len(parent1)
cycle_no = 1
cyclestart = (i for i,v in enumerate(cycles) if v < 0)

for pos in cyclestart:

    while cycles[pos] < 0:
        cycles[pos] = cycle_no
        pos = parent1.index(parent2[pos])cycles = [-1]*len(parent1)
cycle_no = 1
cyclestart = (i for i,v in enumerate(cycles) if v < 0)

for pos in cyclestart:

    while cycles[pos] < 0:
        cycles[pos] = cycle_no
        pos = parent1.index(parent2[pos])

    cycle_no += 1

child1 = [parent1[i] if n%2 else parent2[i] for i,n in enumerate(cycles)]
child2 = [parent2[i] if n%2 else parent1[i] for i,n in enumerate(cycles)]


print("parent1:", parent1)
print("parent2:", parent2)
print("cycles:", cycles)
print("child1:", child1)
print("child2:", child2)