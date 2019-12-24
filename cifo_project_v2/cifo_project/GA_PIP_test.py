from cifo.algorithm.genetic_algorithm import GeneticAlgorithm
from cifo.custom_problem.portfolio_investment_problem import PortfolioInvestmentProblem

from cifo.algorithm.ga_operators import (initialize_randomly, 
    RankSelection, RouletteWheelSelection, TournamentSelection, 
    singlepoint_crossover, pmx_crossover, cycle_crossover,
    single_point_mutation, swap_mutation, 
    standard_replacement, elitism_replacement
)

import pandas as pd

df = pd.read_excel('C:/Users/Mafalda/CIFO/CIFO-Project/cifo_project_v2/cifo_project/data/sp500.xlsx')
df_hist = pd.read_excel('C:/Users/Mafalda/CIFO/CIFO-Project/cifo_project_v2/cifo_project/data/sp_12_weeks.xlsx')

pip_decision_variables_example = {
    "Symbols"             : list(df['symbol']),
    "Assets"              : list(df['name']),
    "Prices"              : list(df['price']),
    "Expected-Returns"    : list(df['exp_return_3m']),
    "Standard-Deviations" : list(df['standard_deviation']),
    "Historical-Data"     : df_hist
}

pip_constraints_example = {
    "Risk-Tolerance" : 1,
    "Budget"         : 10000
}

n_max = int(pip_constraints_example["Budget"] // min(pip_decision_variables_example["Prices"]))

pip_encoding_rule = {
    "Size"         : df.shape[0], # It must be defined by the size of DV (Number of products)
    "Is ordered"   : True,
    "Can repeat"   : False,
    "Data"         : [i for i in range(0, n_max+1)],
    "Data Type"    : "Choices"
}

pip = PortfolioInvestmentProblem(
    decision_variables = pip_decision_variables_example,
    constraints = pip_constraints_example,
    encoding_rule = pip_encoding_rule
    )

params = {
    "Population-Size"           : 5,
    "Number-of-Generations"     : 10,
    
    "Crossover-Probability"     : 0.8,
    "Mutation-Probability"      : 0.5,
    
    "Initialization-Approach"   : initialize_randomly,
    "Selection-Approach"        : RankSelection(),
    "Tournament-Size"           : 5,
    "Crossover-Approach"        : pmx_crossover,
    "Mutation-Aproach"          : swap_mutation,
    "Replacement-Approach"      : standard_replacement
}


solution = GeneticAlgorithm(
    problem_instance = pip
)

"""
solution = GeneticAlgorithm(
    problem_instance = tsp,
    params = params
)
"""

import matplotlib.pyplot as plt

fit = []
for i in range(1,20):
    print(i)
    print(solution.search())
    fit.append(solution.search().fitness)

avg = sum(fit)/len(fit)

plt.plot(range(1,20),fit)
plt.axhline(y=avg, color='r', linestyle='-', label=str(avg))
plt.legend()
plt.show()