#TSP
import numpy as np


### NOTE: THE ORDERED CROSSOVER CAN ALSO SOLVE THIS PROBLEM (to the optional approach)###


### ROULLETE WHEEL FOR MINIMIZATION ###
### SELECTION APPROACH - we can also apply the tournament selection to this ###
"""

For using Roulette wheel selection for minimization, you have to do two pre-processing steps:

1º)
You have to get rid of the negative fitness values, because the fitness value will represent 
the selection probability, which can't be negative. The easiest way for doing this, is to 
subtract the lowest (negative) value from all fitness values. The lowest fitness value is 
now zero.

2º)
For minimizing, you have to revert the fitness values. This is done by setting the fitness 
values to max fitness - fitness. The individual with the best fitness has now the highest 
fitness value.

The transformed fitness values are now feed into the normal Roulette wheel selector, 
which selects the individual with the highest fitness. But essentially you are doing a
minimization.

"""

#The population has to be a vector with the cities numbers (?)
#Admissible: a valid solution would need to represent a route where every location is included at least once and only once.
#Fitness: fitness function calculates the total distance between each city in the chromosome’s permutation


class RouletteWheelSelection:
    """
    Main idea: better individuals get higher chance
    Chances proportional to fitness
    Implementation: roulette wheel technique
    Assign to each individual a part of the roulette wheel
    Spin the wheel n times to select n individuals
    """
    def select(self, population):
        """
        select two different parents using roulette wheel
        """
        index1 = self.select_index(population = population)
        index2 = index1
        
        while index2 == index1:
            index2 = self.select_index(population = population)

        return population[index1], population[index2]

    def uniform(self, a, b):
        s = np.random.uniform(a,b)
        "Get a random number in the range [a, b) or [a, b] depending on rounding."
        return a + (b - a) * s

    def select_index(self, population):

        # Get the Total Fitness (all solutions in the population) to calculate the chances proportional to fitness
        total_fitness = 0
        for sol in population:
            total_fitness += sol

        # spin the wheel
        wheel_position = self.uniform(0, 1)

        # calculate the position which wheel should stop
        stop_position = 0
        index = 0
        for sol in population :
            stop_position += (sol / total_fitness)
            if stop_position > wheel_position :
                break
            index += 1    

        return index    


a = RouletteWheelSelection()

list_test = [1, 2, 3, 4, 5, 6, 7, 8]

b = a.select(list_test)

print(b)


### PARTIAL MAPPED CROSSOVER ###

data = [
[​ 0 ​ , ​ 2451​ , ​ 713​ , ​ 1018​ , ​ 1631​ , ​ 1374​ , ​ 2408​ , ​ 213​ , ​ 2571​ , ​ 875​ , ​ 1420​ , ​ 2145​ , ​ 1972​ ],
[​ 2451​ , ​ 0 ​ , ​ 1745​ , ​ 1524​ , ​ 831​ , ​ 1240​ , ​ 959​ , ​ 2596​ , ​ 403​ , ​ 1589​ , ​ 1374​ , ​ 357​ , ​ 579​ ],
[​ 713​ , ​ 1745​ , ​ 0 ​ , ​ 355​ , ​ 920​ , ​ 803​ , ​ 1737​ , ​ 851​ , ​ 1858​ , ​ 262​ , ​ 940​ , ​ 1453​ , ​ 1260​ ],
[​ 1018​ , ​ 1524​ , ​ 355​ , ​ 0 ​ , ​ 700​ , ​ 862​ , ​ 1395​ , ​ 1123​ , ​ 1584​ , ​ 466​ , ​ 1056​ , ​ 1280​ , ​ 987​ ],
[​ 1631​ , ​ 831​ , ​ 920​ , ​ 700​ , ​ 0 ​ , ​ 663​ , ​ 1021​ , ​ 1769​ , ​ 949​ , ​ 796​ , ​ 879​ , ​ 586​ , ​ 371​ ],
[​ 1374​ , ​ 1240​ , ​ 803​ , ​ 862​ , ​ 663​ , ​ 0 ​ , ​ 1681​ , ​ 1551​ , ​ 1765​ , ​ 547​ , ​ 225​ , ​ 887​ , ​ 999​ ],
[​ 2408​ , ​ 959​ , ​ 1737​ , ​ 1395​ , ​ 1021​ , ​ 1681​ , ​ 0 ​ , ​ 2493​ , ​ 678​ , ​ 1724​ , ​ 1891​ , ​ 1114​ , ​ 701​ ],
[​ 213​ , ​ 2596​ , ​ 851​ , ​ 1123​ , ​ 1769​ , ​ 1551​ , ​ 2493​ , ​ 0 ​ , ​ 2699​ , ​ 1038​ , ​ 1605​ , ​ 2300​ , ​ 2099​ ],
[​ 2571​ , ​ 403​ , ​ 1858​ , ​ 1584​ , ​ 949​ , ​ 1765​ , ​ 678​ , ​ 2699​ , ​ 0 ​ , ​ 1744​ , ​ 1645​ , ​ 653​ , ​ 600​ ],
[​ 875​ , ​ 1589​, ​ 262​ , ​ 466​ , ​ 796​ , ​ 547​ , ​ 1724​ , ​ 1038​ , ​ 1744​ , ​ 0 ​ , ​ 679​ , ​ 1272​ , ​ 1162​ ],
[​ 1420​ , ​ 1374​ , ​ 940​ , ​ 1056​ , ​ 879​ , ​ 225​ , ​ 1891​ , ​ 1605​ , ​ 1645​ , ​ 679​ , ​ 0 ​ , ​ 1017​ , ​ 1200​ ],
[​ 2145​ , ​ 357​ , ​ 1453​ , ​ 1280​ , ​ 586​ , ​ 887​ , ​ 1114​ , ​ 2300​ , ​ 653​ , ​ 1272​ , ​ 1017​ , ​ 0 ​ , ​ 504​ ],
[​ 1972​ , ​ 579​ , ​ 1260​ , ​ 987​ , ​ 371​ , ​ 999​ , ​ 701​ , ​ 2099​ , ​ 600​ , ​ 1162​ , ​ 1200​ , ​ 504​ , ​ 0 ​ ],
]
]