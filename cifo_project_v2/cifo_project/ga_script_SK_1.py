# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------------------

from cifo.algorithm.genetic_algorithm import GeneticAlgorithm
from cifo.algorithm.hill_climbing import HillClimbing
from cifo.custom_problem.knapsack_problem import (
    KnapsackProblem, knapsack_decision_variables_example, knapsack_constraints_example, 
    knapsack_bitflip_get_neighbors
)
from cifo.custom_problem.travel_salesman_problem import (
    TravelSalesmanProblem, tsp_bitflip_get_neighbors
)
from cifo.custom_problem.portfolio_investment_problem import (
    PortfolioInvestmentProblem, pip_bitflip_get_neighbors
)
from cifo.problem.objective import ProblemObjective
from cifo.algorithm.ga_operators import (
    initialize_randomly, initialize_using_hc, initialize_using_sa, initialize_using_ts,
    RouletteWheelSelection, RankSelection, TournamentSelection, 
    singlepoint_crossover, pmx_crossover, cycle_crossover, n_point_crossover,
    order_1_crossover, uniform_crossover,
    single_point_mutation, swap_mutation, insert_mutation, inversion_mutation,
    scramble_mutation, uniform_mutation,
    elitism_replacement, standard_replacement 
)    
from cifo.util.terminal import Terminal, FontColor
from cifo.util.observer import GeneticAlgorithmObserver
from random import randint

import pandas as pd

def plot_performance_chart( df ):
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    x = df["Generation"] 
    x_rev = x[::-1]
    y1 = df["Fitness_Mean"] 
    y1_upper = df["Fitness_Lower"]
    y1_lower = df["Fitness_Upper"]



    # line
    trace1 = go.Scatter(
        x = x,
        y = y1,
        line=dict(color='rgb(0,100,80)'),
        mode='lines',
        name='Fair',
    )

    trace2 = go.Scatter(
        x = x,
        y = y1_upper,
        fill='tozerox',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Fair',
    )

    trace3 = go.Scatter(
        x = x,
        y = y1_lower,
        fill='tozerox',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        name='Fair',
    )

    data = [trace1]
    
    layout = go.Layout(
        paper_bgcolor='rgb(255,255,255)',
        plot_bgcolor='rgb(229,229,229)',
        xaxis=dict(
            gridcolor='rgb(255,255,255)',
            range=[1,10],
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False
        ),
        yaxis=dict(
            gridcolor='rgb(255,255,255)',
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    fig.show()

# Problem
#--------------------------------------------------------------------------------------------------
# Decision Variables
dv = {
    "Values"    : [360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147, 
    78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28, 87, 73, 78, 15, 
    26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276, 312], 

    "Weights"   : [7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
    42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
    3, 86, 66, 31, 65, 0, 79, 20, 65, 52, 13]
}

# Problem Instance
knapsack_problem_instance = KnapsackProblem( 
    decision_variables = dv,
    constraints = { "Max-Weight" : 400 })

# Problem
#--------------------------------------------------------------------------------------------------
# Decision Variables
data = [
    [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
    [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
    [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
    [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
    [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
    [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
    [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
    [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
    [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
    [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
    [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
    [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
    [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
]

tsp_dv = {
    "Distances" : data, #<< Number, Mandatory
    "Cities"    : [i for i in range(0, len(data))], #<< Number, Mandatory
}

tsp_encoding_rule = {
    "Size"         : len(data), # It must be defined by the size of DV (Number of products)
    "Is ordered"   : True,
    "Can repeat"   : False,
    "Data"         : [i for i in range(0, len(data))],
    "Data Type"    : "Choices"
}

# Problem Instance
tsp = TravelSalesmanProblem(
    decision_variables = tsp_dv,
    encoding_rule = tsp_encoding_rule
    )

# Problem
#--------------------------------------------------------------------------------------------------
# Decision Variables

df = pd.read_excel('C:/Users/Mafalda/CIFO/CIFO-Project/cifo_project_v2/cifo_project/data/sp500.xlsx')
df_hist = pd.read_excel('C:/Users/Mafalda/CIFO/CIFO-Project/cifo_project_v2/cifo_project/data/sp_12_weeks.xlsx')

pip_decision_variables = {
    "Symbols"             : list(df['symbol']),
    "Assets"              : list(df['name']),
    "Prices"              : list(df['price']),
    "Expected-Returns"    : list(df['exp_return_3m']),
    "Standard-Deviations" : list(df['standard_deviation']),
    "Historical-Data"     : df_hist
}

pip_constraints = {
    "Risk-Tolerance" : 1,
    "Budget"         : 10000
}

n_max = int(pip_constraints["Budget"] // min(pip_decision_variables["Prices"]))

pip_encoding_rule = {
    "Size"         : df.shape[0], # It must be defined by the size of DV (Number of products)
    "Is ordered"   : True,
    "Can repeat"   : False,
    "Data"         : [i for i in range(0, n_max+1)],
    "Data Type"    : "Choices"
}

# Problem Instance
pip = PortfolioInvestmentProblem(
    decision_variables = pip_decision_variables,
    constraints = pip_constraints,
    encoding_rule = pip_encoding_rule
    )

# Configuration
#--------------------------------------------------------------------------------------------------
# parent selection object
parent_selection = TournamentSelection()
#parent_selection = RouletteWheelSelection()

params = {
        # params
        "Population-Size"           : 10,
        "Number-of-Generations"     : 10,
        "Crossover-Probability"     : 0.8,
        "Mutation-Probability"      : 0.8,
        # operators / approaches
        "Initialization-Approach"   : initialize_using_hc,
        "Selection-Approach"        : parent_selection,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : singlepoint_crossover,
        "Mutation-Aproach"          : single_point_mutation,
        "Replacement-Approach"      : elitism_replacement
    }

hc_init_params = {
    "Maximum-Iterations" : 20,
    "Stop-Conditions" : "Alternative-01",
    "Neighborhood-Size": 10,
    "Neighborhood-Function": pip_bitflip_get_neighbors
}

log_name = "mp0-8"

number_of_runs = 30

# Run the same configuration many times
#--------------------------------------------------------------------------------------------------
for run in range(1,number_of_runs + 1):
    # Genetic Algorithm
    # ga = GeneticAlgorithm( 
        # problem_instance = knapsack_problem_instance,
        # params =  params,
        # run = run,
        # log_name = log_name )

    ga = GeneticAlgorithm( 
        problem_instance = pip,
        params =  params,
        init_params = hc_init_params,
        run = run,
        log_name = log_name )
    
    ga_observer = GeneticAlgorithmObserver( ga )
    ga.register_observer( ga_observer )
    ga.search()    
    ga.save_log()

# Consolidate the runs
#--------------------------------------------------------------------------------------------------

# save the config

# consolidate the runs information
from os import listdir, path, mkdir
from os.path import isfile, join
from pandas import pandas as pd
import numpy as np

log_dir   = f"./log/{log_name}" 

log_files = [f for f in listdir(log_dir) if isfile(join(log_dir, f))]
print(log_files)

fitness_runs = []
columns_name = []
counter = 0
generations = []

for log_name in log_files:
    if log_name.startswith('run_'):
        df = pd.read_excel( log_dir + "/" + log_name)
        fitness_runs.append( list ( df.Fitness ) )
        columns_name.append( log_name.strip(".xslx") )
        counter += 1

        if not generations:
            generations = list( df["Generation"] )
        
#fitness_sum = [sum(x) for x in zip(*fitness_runs)]   

df = pd.DataFrame(list(zip(*fitness_runs)), columns = columns_name)

fitness_sd   = list( df.std( axis = 1 ) )
fitness_mean = list( df.mean( axis = 1 ) )

#df["Fitness_Sum"] = fitness_sum
df["Generation"]  = generations
df["Fitness_SD"]  = fitness_sd
df["Fitness_Mean"]  = fitness_mean
df["Fitness_Lower"]  = df["Fitness_Mean"] + df["Fitness_SD"]
df["Fitness_Upper"]  = df["Fitness_Mean"] - df["Fitness_SD"]


if not path.exists( log_dir ):
    mkdir( log_dir )

df.to_excel( log_dir + "/all.xlsx", index = False, encoding = 'utf-8' )

plot_performance_chart( df )




#[sum(sublist) for sublist in itertools.izip(*myListOfLists)]



