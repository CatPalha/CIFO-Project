from cifo.custom_problem.portfolio_investment_problem import PortfolioInvestmentProblem
from cifo.custom_problem.portfolio_investment_problem import pip_bitflip_get_neighbors

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

solution = pip.build_solution()
admissible = pip.is_admissible(solution)

#while admissible == False:
    #solution = pip.build_solution()
    #admissible = pip.is_admissible(solution)

fitness = pip.evaluate_solution(solution)
neighborhood = pip_bitflip_get_neighbors(solution, pip, 5)

print("Solution", solution)
print("Admissible", admissible)
#print("Fitness", fitness)
#if admissible == True:
    #print("Neighborhood", neighborhood)