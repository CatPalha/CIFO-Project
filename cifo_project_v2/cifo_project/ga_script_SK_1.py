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

def plot_performance_chart( df0,df1,df2,df3,df4,df5,df6,df7,df8,df9,df10 ):
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    x0 = df0["Generation"] 
    x_rev0 = x0[::-1]
    y1_0 = df0["Fitness_Mean"] 
    #y1_upper = df["Fitness_Lower"]
    #y1_lower = df["Fitness_Upper"]
    
    x1 = df1["Generation"] 
    x_rev1 = x1[::-1]
    y1_1 = df1["Fitness_Mean"] 
    
    x2 = df2["Generation"] 
    x_rev2 = x2[::-1]
    y1_2 = df2["Fitness_Mean"] 
    
    x3 = df3["Generation"] 
    x_rev3 = x3[::-1]
    y1_3 = df3["Fitness_Mean"] 
    
    x4 = df4["Generation"] 
    x_rev4 = x4[::-1]
    y1_4 = df4["Fitness_Mean"] 

    x5 = df5["Generation"] 
    x_rev5 = x5[::-1]
    y1_5 = df5["Fitness_Mean"] 

    x6 = df6["Generation"] 
    x_rev6 = x6[::-1]
    y1_6 = df6["Fitness_Mean"] 

    x7 = df7["Generation"] 
    x_rev7 = x7[::-1]
    y1_7 = df7["Fitness_Mean"] 

    x8 = df8["Generation"] 
    x_rev8 = x8[::-1]
    y1_8 = df8["Fitness_Mean"] 
    
    x9 = df9["Generation"] 
    x_rev9 = x9[::-1]
    y1_9 = df9["Fitness_Mean"] 
        
    x10 = df10["Generation"] 
    x_rev10 = x10[::-1]
    y1_10 = df10["Fitness_Mean"] 
    
    # line
    trace0 = go.Scatter(
        x = x0,
        y = y1_0,
        line=dict(color='rgb(255,0,0)'),
        mode='lines',
        name='0.0',
    )
    
    trace1 = go.Scatter(
        x = x1,
        y = y1_1,
        line=dict(color='rgb(255,128,0)'),
        mode='lines',
        name='0.1',
    )
    
    trace2 = go.Scatter(
        x = x2,
        y = y1_2,
        line=dict(color='rgb(255,255,0)'),
        mode='lines',
        name='0.2',
    )
    
    trace3 = go.Scatter(
        x = x3,
        y = y1_3,
        line=dict(color='rgb(128,255,0)'),
        mode='lines',
        name='0.3',
    )
    
    trace4 = go.Scatter(
        x = x4,
        y = y1_4,
        line=dict(color='rgb(0,255,0)'),
        mode='lines',
        name='0.4',
    )

    trace5 = go.Scatter(
        x = x5,
        y = y1_5,
        line=dict(color='rgb(0,255,128)'),
        mode='lines',
        name='0.5',
    )

    trace6 = go.Scatter(
        x = x6,
        y = y1_6,
        line=dict(color='rgb(0,255,255)'),
        mode='lines',
        name='0.6',
    )

    trace7 = go.Scatter(
        x = x7,
        y = y1_7,
        line=dict(color='rgb(0,128,255)'),
        mode='lines',
        name='0.7',
    )

    trace8 = go.Scatter(
        x = x8,
        y = y1_8,
        line=dict(color='rgb(0,0,255)'),
        mode='lines',
        name='0.8',
    )
    
    trace9 = go.Scatter(
        x = x9,
        y = y1_9,
        line=dict(color='rgb(128,0,255)'),
        mode='lines',
        name='0.9',
    )
    
    trace10 = go.Scatter(
        x = x10,
        y = y1_10,
        line=dict(color='rgb(255,0,255)'),
        mode='lines',
        name='1.0',
    )
    
    #trace2 = go.Scatter(
        #x = x,
        #y = y1_upper,
        #fill='tozerox',
        #fillcolor='rgba(0,100,80,0.2)',
        #line=dict(color='rgba(255,255,255,0)'),
        #showlegend=False,
        #name='Fair',
    #)

    #trace3 = go.Scatter(
        #x = x,
        #y = y1_lower,
        #fill='tozerox',
        #fillcolor='rgba(0,100,80,0.2)',
        #line=dict(color='rgba(255,255,255,0)'),
        #showlegend=False,
        #name='Fair',
    #)
    
    data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10]
    
    layout = go.Layout(
        paper_bgcolor='rgb(255,255,255)',
        plot_bgcolor='rgb(229,229,229)',
        xaxis=dict(
            gridcolor='rgb(255,255,255)',
            range=[0,250],
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
#parent_selection = TournamentSelection()
parent_selection = RouletteWheelSelection()

params0 = {
        # params
        "Population-Size"           : 20,
        "Number-of-Generations"     : 250,
        "Crossover-Probability"     : 0.0,
        "Mutation-Probability"      : 0.5,
        # operators / approaches
        "Initialization-Approach"   : initialize_randomly,
        "Selection-Approach"        : parent_selection,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : cycle_crossover,
        "Mutation-Aproach"          : swap_mutation,
        "Replacement-Approach"      : elitism_replacement
    }

params1 = {
        # params
        "Population-Size"           : 20,
        "Number-of-Generations"     : 250,
        "Crossover-Probability"     : 0.1,
        "Mutation-Probability"      : 0.5,
        # operators / approaches
        "Initialization-Approach"   : initialize_randomly,
        "Selection-Approach"        : parent_selection,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : cycle_crossover,
        "Mutation-Aproach"          : swap_mutation,
        "Replacement-Approach"      : elitism_replacement
    }

params2 = {
        # params
        "Population-Size"           : 20,
        "Number-of-Generations"     : 250,
        "Crossover-Probability"     : 0.2,
        "Mutation-Probability"      : 0.5,
        # operators / approaches
        "Initialization-Approach"   : initialize_randomly,
        "Selection-Approach"        : parent_selection,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : cycle_crossover,
        "Mutation-Aproach"          : swap_mutation,
        "Replacement-Approach"      : elitism_replacement
    }

params3 = {
        # params
        "Population-Size"           : 20,
        "Number-of-Generations"     : 250,
        "Crossover-Probability"     : 0.3,
        "Mutation-Probability"      : 0.5,
        # operators / approaches
        "Initialization-Approach"   : initialize_randomly,
        "Selection-Approach"        : parent_selection,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : cycle_crossover,
        "Mutation-Aproach"          : swap_mutation,
        "Replacement-Approach"      : elitism_replacement
    }

params4 = {
        # params
        "Population-Size"           : 20,
        "Number-of-Generations"     : 250,
        "Crossover-Probability"     : 0.4,
        "Mutation-Probability"      : 0.5,
        # operators / approaches
        "Initialization-Approach"   : initialize_randomly,
        "Selection-Approach"        : parent_selection,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : cycle_crossover,
        "Mutation-Aproach"          : swap_mutation,
        "Replacement-Approach"      : elitism_replacement
    }

params5 = {
        # params
        "Population-Size"           : 20,
        "Number-of-Generations"     : 250,
        "Crossover-Probability"     : 0.5,
        "Mutation-Probability"      : 0.5,
        # operators / approaches
        "Initialization-Approach"   : initialize_randomly,
        "Selection-Approach"        : parent_selection,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : cycle_crossover,
        "Mutation-Aproach"          : swap_mutation,
        "Replacement-Approach"      : elitism_replacement
    }

params6 = {
        # params
        "Population-Size"           : 20,
        "Number-of-Generations"     : 250,
        "Crossover-Probability"     : 0.6,
        "Mutation-Probability"      : 0.5,
        # operators / approaches
        "Initialization-Approach"   : initialize_randomly,
        "Selection-Approach"        : parent_selection,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : cycle_crossover,
        "Mutation-Aproach"          : swap_mutation,
        "Replacement-Approach"      : elitism_replacement
    }

params7 = {
        # params
        "Population-Size"           : 20,
        "Number-of-Generations"     : 250,
        "Crossover-Probability"     : 0.7,
        "Mutation-Probability"      : 0.5,
        # operators / approaches
        "Initialization-Approach"   : initialize_randomly,
        "Selection-Approach"        : parent_selection,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : cycle_crossover,
        "Mutation-Aproach"          : swap_mutation,
        "Replacement-Approach"      : elitism_replacement
    }

params8 = {
        # params
        "Population-Size"           : 20,
        "Number-of-Generations"     : 250,
        "Crossover-Probability"     : 0.8,
        "Mutation-Probability"      : 0.5,
        # operators / approaches
        "Initialization-Approach"   : initialize_randomly,
        "Selection-Approach"        : parent_selection,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : cycle_crossover,
        "Mutation-Aproach"          : swap_mutation,
        "Replacement-Approach"      : elitism_replacement
    }

params9 = {
        # params
        "Population-Size"           : 20,
        "Number-of-Generations"     : 250,
        "Crossover-Probability"     : 0.9,
        "Mutation-Probability"      : 0.5,
        # operators / approaches
        "Initialization-Approach"   : initialize_randomly,
        "Selection-Approach"        : parent_selection,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : cycle_crossover,
        "Mutation-Aproach"          : swap_mutation,
        "Replacement-Approach"      : elitism_replacement
    }

params10 = {
        # params
        "Population-Size"           : 20,
        "Number-of-Generations"     : 250,
        "Crossover-Probability"     : 1.0,
        "Mutation-Probability"      : 0.5,
        # operators / approaches
        "Initialization-Approach"   : initialize_randomly,
        "Selection-Approach"        : parent_selection,
        "Tournament-Size"           : 5,
        "Crossover-Approach"        : cycle_crossover,
        "Mutation-Aproach"          : swap_mutation,
        "Replacement-Approach"      : elitism_replacement
    }

neighborhood_function = tsp_bitflip_get_neighbors
#neighborhood_function = pip_bitflip_get_neighbors

hc_init_params_0 = {
    "Maximum-Iterations" : 10,
    "Stop-Conditions" : "Classical",
    "Neighborhood-Size": 10,
    "Neighborhood-Function": neighborhood_function
}

hc_init_params_1 = {
    "Maximum-Iterations" : 20,
    "Stop-Conditions" : "Classical",
    "Neighborhood-Size": 10,
    "Neighborhood-Function": neighborhood_function
}

hc_init_params_2 = {
    "Maximum-Iterations" : 30,
    "Stop-Conditions" : "Classical",
    "Neighborhood-Size": 10,
    "Neighborhood-Function": neighborhood_function
}

hc_init_params_3 = {
    "Maximum-Iterations" : 40,
    "Stop-Conditions" : "Classical",
    "Neighborhood-Size": 10,
    "Neighborhood-Function": neighborhood_function
}

hc_init_params_4 = {
    "Maximum-Iterations" : 50,
    "Stop-Conditions" : "Classical",
    "Neighborhood-Size": 10,
    "Neighborhood-Function": neighborhood_function
}

hc_init_params_5 = {
    "Maximum-Iterations" : 60,
    "Stop-Conditions" : "Classical",
    "Neighborhood-Size": 10,
    "Neighborhood-Function": neighborhood_function
}

hc_init_params_6 = {
    "Maximum-Iterations" : 70,
    "Stop-Conditions" : "Classical",
    "Neighborhood-Size": 10,
    "Neighborhood-Function": neighborhood_function
}

hc_init_params_7 = {
    "Maximum-Iterations" : 80,
    "Stop-Conditions" : "Classical",
    "Neighborhood-Size": 10,
    "Neighborhood-Function": neighborhood_function
}

hc_init_params_8 = {
    "Maximum-Iterations" : 90,
    "Stop-Conditions" : "Classical",
    "Neighborhood-Size": 10,
    "Neighborhood-Function": neighborhood_function
}

hc_init_params_9 = {
    "Maximum-Iterations" : 100,
    "Stop-Conditions" : "Classical",
    "Neighborhood-Size": 10,
    "Neighborhood-Function": neighborhood_function
}
"""
hc_init_params_10 = {
    "Maximum-Iterations" : 20,
    "Stop-Conditions" : "Classical",
    "Neighborhood-Size": 10,
    "Neighborhood-Function": neighborhood_function
}
"""

sa_init_params0 = {
    "Maximum-Internal-Iterations" : 5,
    "Maximum-Iterations" : 10,
    #"Initial-C" : 200,
    #"Minimum-C" : 0.01,
    "Update-Method" : "Geometric",
    "Update-Rate" : 0.9,
    "Initialize-Method-C" : "Classical",
    "Initialize-Method-Minimum-C" : "Classical",
    "Neighborhood-Size" : 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

sa_init_params1 = {
    "Maximum-Internal-Iterations" : 10,
    "Maximum-Iterations" : 10,
    #"Initial-C" : 200,
    #"Minimum-C" : 0.01,
    "Update-Method" : "Geometric",
    "Update-Rate" : 0.9,
    "Initialize-Method-C" : "Classical",
    "Initialize-Method-Minimum-C" : "Classical",
    "Neighborhood-Size" : 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

sa_init_params2 = {
    "Maximum-Internal-Iterations" : 20,
    "Maximum-Iterations" : 10,
    #"Initial-C" : 200,
    #"Minimum-C" : 0.01,
    "Update-Method" : "Geometric",
    "Update-Rate" : 0.9,
    "Initialize-Method-C" : "Classical",
    "Initialize-Method-Minimum-C" : "Classical",
    "Neighborhood-Size" : 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

sa_init_params3 = {
    "Maximum-Internal-Iterations" : 30,
    "Maximum-Iterations" : 10,
    #"Initial-C" : 200,
    #"Minimum-C" : 0.01,
    "Update-Method" : "Geometric",
    "Update-Rate" : 0.9,
    "Initialize-Method-C" : "Classical",
    "Initialize-Method-Minimum-C" : "Classical",
    "Neighborhood-Size" : 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

sa_init_params4 = {
    "Maximum-Internal-Iterations" : 40,
    "Maximum-Iterations" : 10,
    #"Initial-C" : 200,
    #"Minimum-C" : 0.01,
    "Update-Method" : "Geometric",
    "Update-Rate" : 0.9,
    "Initialize-Method-C" : "Classical",
    "Initialize-Method-Minimum-C" : "Classical",
    "Neighborhood-Size" : 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

sa_init_params5 = {
    "Maximum-Internal-Iterations" : 50,
    "Maximum-Iterations" : 10,
    #"Initial-C" : 200,
    #"Minimum-C" : 0.01,
    "Update-Method" : "Geometric",
    "Update-Rate" : 0.9,
    "Initialize-Method-C" : "Classical",
    "Initialize-Method-Minimum-C" : "Classical",
    "Neighborhood-Size" : 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

sa_init_params6 = {
    "Maximum-Internal-Iterations" : 60,
    "Maximum-Iterations" : 10,
    #"Initial-C" : 200,
    #"Minimum-C" : 0.01,
    "Update-Method" : "Geometric",
    "Update-Rate" : 0.9,
    "Initialize-Method-C" : "Classical",
    "Initialize-Method-Minimum-C" : "Classical",
    "Neighborhood-Size" : 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

sa_init_params7 = {
    "Maximum-Internal-Iterations" : 70,
    "Maximum-Iterations" : 10,
    #"Initial-C" : 200,
    #"Minimum-C" : 0.01,
    "Update-Method" : "Geometric",
    "Update-Rate" : 0.9,
    "Initialize-Method-C" : "Classical",
    "Initialize-Method-Minimum-C" : "Classical",
    "Neighborhood-Size" : 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

sa_init_params8 = {
    "Maximum-Internal-Iterations" : 80,
    "Maximum-Iterations" : 10,
    #"Initial-C" : 200,
    #"Minimum-C" : 0.01,
    "Update-Method" : "Geometric",
    "Update-Rate" : 0.9,
    "Initialize-Method-C" : "Classical",
    "Initialize-Method-Minimum-C" : "Classical",
    "Neighborhood-Size" : 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

sa_init_params9 = {
    "Maximum-Internal-Iterations" : 90,
    "Maximum-Iterations" : 10,
    #"Initial-C" : 200,
    #"Minimum-C" : 0.01,
    "Update-Method" : "Geometric",
    "Update-Rate" : 0.9,
    "Initialize-Method-C" : "Classical",
    "Initialize-Method-Minimum-C" : "Classical",
    "Neighborhood-Size" : 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

sa_init_params10 = {
    "Maximum-Internal-Iterations" : 100,
    "Maximum-Iterations" : 10,
    #"Initial-C" : 200,
    #"Minimum-C" : 0.01,
    "Update-Method" : "Geometric",
    "Update-Rate" : 0.9,
    "Initialize-Method-C" : "Classical",
    "Initialize-Method-Minimum-C" : "Classical",
    "Neighborhood-Size" : 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

ts_init_params0 = {
    "Maximum-Iterations" : 10,
    "Stop-Conditions" : "Classical",
    #"Target-Fitness"
    "Neighborhood-Size": 5,
    "Memory-Size": 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

ts_init_params1 = {
    "Maximum-Iterations" : 20,
    "Stop-Conditions" : "Classical",
    #"Target-Fitness"
    "Neighborhood-Size": 5,
    "Memory-Size": 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

ts_init_params2 = {
    "Maximum-Iterations" : 30,
    "Stop-Conditions" : "Classical",
    #"Target-Fitness"
    "Neighborhood-Size": 5,
    "Memory-Size": 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

ts_init_params3 = {
    "Maximum-Iterations" : 40,
    "Stop-Conditions" : "Classical",
    #"Target-Fitness"
    "Neighborhood-Size": 5,
    "Memory-Size": 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

ts_init_params4 = {
    "Maximum-Iterations" : 50,
    "Stop-Conditions" : "Classical",
    #"Target-Fitness"
    "Neighborhood-Size": 5,
    "Memory-Size": 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

ts_init_params5 = {
    "Maximum-Iterations" : 60,
    "Stop-Conditions" : "Classical",
    #"Target-Fitness"
    "Neighborhood-Size": 5,
    "Memory-Size": 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

ts_init_params6 = {
    "Maximum-Iterations" : 70,
    "Stop-Conditions" : "Classical",
    #"Target-Fitness"
    "Neighborhood-Size": 5,
    "Memory-Size": 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

ts_init_params7 = {
    "Maximum-Iterations" : 80,
    "Stop-Conditions" : "Classical",
    #"Target-Fitness"
    "Neighborhood-Size": 5,
    "Memory-Size": 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

ts_init_params8 = {
    "Maximum-Iterations" : 90,
    "Stop-Conditions" : "Classical",
    #"Target-Fitness"
    "Neighborhood-Size": 5,
    "Memory-Size": 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}

ts_init_params9 = {
    "Maximum-Iterations" : 100,
    "Stop-Conditions" : "Classical",
    #"Target-Fitness"
    "Neighborhood-Size": 5,
    "Memory-Size": 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}
"""
ts_init_params10 = {
    "Maximum-Iterations" : 10,
    "Stop-Conditions" : "Classical",
    #"Target-Fitness"
    "Neighborhood-Size": 5,
    "Memory-Size": 5,
    "Neighborhood-Function": tsp_bitflip_get_neighbors
}
"""
log_name0 = "mp0-8_0"
log_name1 = "mp0-8_1"
log_name2 = "mp0-8_2"
log_name3 = "mp0-8_3"
log_name4 = "mp0-8_4"
log_name5 = "mp0-8_5"
log_name6 = "mp0-8_6"
log_name7 = "mp0-8_7"
log_name8 = "mp0-8_8"
log_name9 = "mp0-8_9"
log_name10 = "mp0-8_10"

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

    ga0 = GeneticAlgorithm( 
        problem_instance = tsp,
        params =  params0,
        # init_params = hc_init_params,
        run = run,
        log_name = log_name0 )
    
    ga1 = GeneticAlgorithm( 
        problem_instance = tsp,
        params =  params1,
        # init_params = hc_init_params,
        run = run,
        log_name = log_name1 )
    
    ga2 = GeneticAlgorithm( 
        problem_instance = tsp,
        params =  params2,
        # init_params = hc_init_params,
        run = run,
        log_name = log_name2 )
    
    ga3 = GeneticAlgorithm( 
        problem_instance = tsp,
        params =  params3,
        # init_params = hc_init_params,
        run = run,
        log_name = log_name3 )
    
    ga4 = GeneticAlgorithm( 
        problem_instance = tsp,
        params =  params4,
        # init_params = hc_init_params,
        run = run,
        log_name = log_name4 )

    ga5 = GeneticAlgorithm( 
        problem_instance = tsp,
        params =  params5,
        # init_params = hc_init_params,
        run = run,
        log_name = log_name5 )

    ga6 = GeneticAlgorithm( 
        problem_instance = tsp,
        params =  params6,
        # init_params = hc_init_params,
        run = run,
        log_name = log_name6 )

    ga7 = GeneticAlgorithm( 
        problem_instance = tsp,
        params =  params7,
        # init_params = hc_init_params,
        run = run,
        log_name = log_name7 )

    ga8 = GeneticAlgorithm( 
        problem_instance = tsp,
        params =  params8,
        # init_params = hc_init_params,
        run = run,
        log_name = log_name8 )
    
    ga9 = GeneticAlgorithm( 
        problem_instance = tsp,
        params =  params9,
        # init_params = hc_init_params,
        run = run,
        log_name = log_name9 )
    
    ga10 = GeneticAlgorithm( 
        problem_instance = tsp,
        params =  params10,
        # init_params = hc_init_params,
        run = run,
        log_name = log_name10 )
            
    ga_observer0 = GeneticAlgorithmObserver( ga0 )
    ga0.register_observer( ga_observer0 )
    ga0.search()    
    ga0.save_log()
    
    ga_observer1 = GeneticAlgorithmObserver( ga1 )
    ga1.register_observer( ga_observer1 )
    ga1.search()    
    ga1.save_log()

    ga_observer2 = GeneticAlgorithmObserver( ga2 )
    ga2.register_observer( ga_observer2 )
    ga2.search()    
    ga2.save_log()
    
    ga_observer3 = GeneticAlgorithmObserver( ga3 )
    ga3.register_observer( ga_observer3 )
    ga3.search()    
    ga3.save_log()
    
    ga_observer4 = GeneticAlgorithmObserver( ga4 )
    ga4.register_observer( ga_observer4 )
    ga4.search()    
    ga4.save_log()

    ga_observer5 = GeneticAlgorithmObserver( ga5 )
    ga5.register_observer( ga_observer5 )
    ga5.search()    
    ga5.save_log()

    ga_observer6 = GeneticAlgorithmObserver( ga6 )
    ga6.register_observer( ga_observer6 )
    ga6.search()    
    ga6.save_log()

    ga_observer7 = GeneticAlgorithmObserver( ga7 )
    ga7.register_observer( ga_observer7 )
    ga7.search()    
    ga7.save_log()

    ga_observer8 = GeneticAlgorithmObserver( ga8 )
    ga8.register_observer( ga_observer8 )
    ga8.search()    
    ga8.save_log()
    
    ga_observer9 = GeneticAlgorithmObserver( ga9 )
    ga9.register_observer( ga_observer9 )
    ga9.search()    
    ga9.save_log()
    
    ga_observer10 = GeneticAlgorithmObserver( ga10 )
    ga10.register_observer( ga_observer10 )
    ga10.search()    
    ga10.save_log()
    
# Consolidate the runs
#--------------------------------------------------------------------------------------------------

# save the config

# consolidate the runs information
from os import listdir, path, mkdir
from os.path import isfile, join
from pandas import pandas as pd
import numpy as np

log_dir0   = f"./log/{log_name0}" 
log_dir1   = f"./log/{log_name1}" 
log_dir2   = f"./log/{log_name2}" 
log_dir3   = f"./log/{log_name3}" 
log_dir4   = f"./log/{log_name4}" 
log_dir5   = f"./log/{log_name5}" 
log_dir6   = f"./log/{log_name6}" 
log_dir7   = f"./log/{log_name7}" 
log_dir8   = f"./log/{log_name8}" 
log_dir9   = f"./log/{log_name9}" 
log_dir10   = f"./log/{log_name10}" 

log_files0 = [f for f in listdir(log_dir0) if isfile(join(log_dir0, f))]
log_files1 = [f for f in listdir(log_dir1) if isfile(join(log_dir1, f))]
log_files2 = [f for f in listdir(log_dir2) if isfile(join(log_dir2, f))]
log_files3 = [f for f in listdir(log_dir3) if isfile(join(log_dir3, f))]
log_files4 = [f for f in listdir(log_dir4) if isfile(join(log_dir4, f))]
log_files5 = [f for f in listdir(log_dir5) if isfile(join(log_dir5, f))]
log_files6 = [f for f in listdir(log_dir6) if isfile(join(log_dir6, f))]
log_files7 = [f for f in listdir(log_dir7) if isfile(join(log_dir7, f))]
log_files8 = [f for f in listdir(log_dir8) if isfile(join(log_dir8, f))]
log_files9 = [f for f in listdir(log_dir9) if isfile(join(log_dir9, f))]
log_files10 = [f for f in listdir(log_dir10) if isfile(join(log_dir10, f))]

#print(log_files)

fitness_runs0 = []
columns_name0 = []
counter0 = 0
generations0 = []

for log_name in log_files0:
    if log_name.startswith('run_'):
        df0 = pd.read_excel( log_dir0 + "/" + log_name)
        fitness_runs0.append( list ( df0.Fitness ) )
        columns_name0.append( log_name.strip(".xslx") )
        counter0 += 1

        if not generations0:
            generations0 = list( df0["Generation"] )

fitness_runs1 = []
columns_name1 = []
counter1 = 0
generations1 = []

for log_name in log_files1:
    if log_name.startswith('run_'):
        df1 = pd.read_excel( log_dir1 + "/" + log_name)
        fitness_runs1.append( list ( df1.Fitness ) )
        columns_name1.append( log_name.strip(".xslx") )
        counter1 += 1

        if not generations1:
            generations1 = list( df1["Generation"] )

fitness_runs2 = []
columns_name2 = []
counter2 = 0
generations2 = []

for log_name in log_files2:
    if log_name.startswith('run_'):
        df2 = pd.read_excel( log_dir2 + "/" + log_name)
        fitness_runs2.append( list ( df2.Fitness ) )
        columns_name2.append( log_name.strip(".xslx") )
        counter2 += 1

        if not generations2:
            generations2 = list( df2["Generation"] )

fitness_runs3 = []
columns_name3 = []
counter3 = 0
generations3 = []

for log_name in log_files3:
    if log_name.startswith('run_'):
        df3 = pd.read_excel( log_dir3 + "/" + log_name)
        fitness_runs3.append( list ( df3.Fitness ) )
        columns_name3.append( log_name.strip(".xslx") )
        counter3 += 1

        if not generations3:
            generations3 = list( df3["Generation"] )

fitness_runs4 = []
columns_name4 = []
counter4 = 0
generations4 = []

for log_name in log_files4:
    if log_name.startswith('run_'):
        df4 = pd.read_excel( log_dir4 + "/" + log_name)
        fitness_runs4.append( list ( df4.Fitness ) )
        columns_name4.append( log_name.strip(".xslx") )
        counter4 += 1

        if not generations4:
            generations4 = list( df4["Generation"] )

fitness_runs5 = []
columns_name5 = []
counter5 = 0
generations5 = []

for log_name in log_files5:
    if log_name.startswith('run_'):
        df5 = pd.read_excel( log_dir5 + "/" + log_name)
        fitness_runs5.append( list ( df5.Fitness ) )
        columns_name5.append( log_name.strip(".xslx") )
        counter5 += 1

        if not generations5:
            generations5 = list( df5["Generation"] )

fitness_runs6 = []
columns_name6 = []
counter6 = 0
generations6 = []

for log_name in log_files6:
    if log_name.startswith('run_'):
        df6 = pd.read_excel( log_dir6 + "/" + log_name)
        fitness_runs6.append( list ( df6.Fitness ) )
        columns_name6.append( log_name.strip(".xslx") )
        counter6 += 1

        if not generations6:
            generations6 = list( df6["Generation"] )

fitness_runs7 = []
columns_name7 = []
counter7 = 0
generations7 = []

for log_name in log_files7:
    if log_name.startswith('run_'):
        df7 = pd.read_excel( log_dir7 + "/" + log_name)
        fitness_runs7.append( list ( df7.Fitness ) )
        columns_name7.append( log_name.strip(".xslx") )
        counter7 += 1

        if not generations7:
            generations7 = list( df7["Generation"] )

fitness_runs8 = []
columns_name8 = []
counter8 = 0
generations8 = []

for log_name in log_files8:
    if log_name.startswith('run_'):
        df8 = pd.read_excel( log_dir8 + "/" + log_name)
        fitness_runs8.append( list ( df8.Fitness ) )
        columns_name8.append( log_name.strip(".xslx") )
        counter8 += 1

        if not generations8:
            generations8 = list( df8["Generation"] )

fitness_runs9 = []
columns_name9 = []
counter9 = 0
generations9 = []

for log_name in log_files9:
    if log_name.startswith('run_'):
        df9 = pd.read_excel( log_dir9 + "/" + log_name)
        fitness_runs9.append( list ( df9.Fitness ) )
        columns_name9.append( log_name.strip(".xslx") )
        counter9 += 1

        if not generations9:
            generations9 = list( df9["Generation"] )

fitness_runs10 = []
columns_name10 = []
counter10 = 0
generations10 = []

for log_name in log_files10:
    if log_name.startswith('run_'):
        df10 = pd.read_excel( log_dir10 + "/" + log_name)
        fitness_runs10.append( list ( df10.Fitness ) )
        columns_name10.append( log_name.strip(".xslx") )
        counter10 += 1

        if not generations10:
            generations10 = list( df10["Generation"] )

#fitness_sum = [sum(x) for x in zip(*fitness_runs)]   

df0 = pd.DataFrame(list(zip(*fitness_runs0)), columns = columns_name0)
df1 = pd.DataFrame(list(zip(*fitness_runs1)), columns = columns_name1)
df2 = pd.DataFrame(list(zip(*fitness_runs2)), columns = columns_name2)
df3 = pd.DataFrame(list(zip(*fitness_runs3)), columns = columns_name3)
df4 = pd.DataFrame(list(zip(*fitness_runs4)), columns = columns_name4)
df5 = pd.DataFrame(list(zip(*fitness_runs5)), columns = columns_name5)
df6 = pd.DataFrame(list(zip(*fitness_runs6)), columns = columns_name6)
df7 = pd.DataFrame(list(zip(*fitness_runs7)), columns = columns_name7)
df8 = pd.DataFrame(list(zip(*fitness_runs8)), columns = columns_name8)
df9 = pd.DataFrame(list(zip(*fitness_runs9)), columns = columns_name9)
df10 = pd.DataFrame(list(zip(*fitness_runs10)), columns = columns_name10)

fitness_sd0   = list( df0.std( axis = 1 ) )
fitness_mean0 = list( df0.mean( axis = 1 ) )

fitness_sd1   = list( df1.std( axis = 1 ) )
fitness_mean1 = list( df1.mean( axis = 1 ) )

fitness_sd2   = list( df2.std( axis = 1 ) )
fitness_mean2 = list( df2.mean( axis = 1 ) )

fitness_sd3   = list( df3.std( axis = 1 ) )
fitness_mean3 = list( df3.mean( axis = 1 ) )

fitness_sd4   = list( df4.std( axis = 1 ) )
fitness_mean4 = list( df4.mean( axis = 1 ) )

fitness_sd5   = list( df5.std( axis = 1 ) )
fitness_mean5 = list( df5.mean( axis = 1 ) )

fitness_sd6   = list( df6.std( axis = 1 ) )
fitness_mean6 = list( df6.mean( axis = 1 ) )

fitness_sd7   = list( df7.std( axis = 1 ) )
fitness_mean7 = list( df7.mean( axis = 1 ) )

fitness_sd8   = list( df8.std( axis = 1 ) )
fitness_mean8 = list( df8.mean( axis = 1 ) )

fitness_sd9   = list( df9.std( axis = 1 ) )
fitness_mean9 = list( df9.mean( axis = 1 ) )

fitness_sd10   = list( df10.std( axis = 1 ) )
fitness_mean10 = list( df10.mean( axis = 1 ) )

#df["Fitness_Sum"] = fitness_sum
df0["Generation"]  = generations0
df0["Fitness_SD"]  = fitness_sd0
df0["Fitness_Mean"]  = fitness_mean0
df0["Fitness_Lower"]  = df0["Fitness_Mean"] + df0["Fitness_SD"]
df0["Fitness_Upper"]  = df0["Fitness_Mean"] - df0["Fitness_SD"]

df1["Generation"]  = generations1
df1["Fitness_SD"]  = fitness_sd1
df1["Fitness_Mean"]  = fitness_mean1
df1["Fitness_Lower"]  = df1["Fitness_Mean"] + df1["Fitness_SD"]
df1["Fitness_Upper"]  = df1["Fitness_Mean"] - df1["Fitness_SD"]

df2["Generation"]  = generations2
df2["Fitness_SD"]  = fitness_sd2
df2["Fitness_Mean"]  = fitness_mean2
df2["Fitness_Lower"]  = df2["Fitness_Mean"] + df2["Fitness_SD"]
df2["Fitness_Upper"]  = df2["Fitness_Mean"] - df2["Fitness_SD"]

df3["Generation"]  = generations3
df3["Fitness_SD"]  = fitness_sd3
df3["Fitness_Mean"]  = fitness_mean3
df3["Fitness_Lower"]  = df3["Fitness_Mean"] + df3["Fitness_SD"]
df3["Fitness_Upper"]  = df3["Fitness_Mean"] - df3["Fitness_SD"]

df4["Generation"]  = generations4
df4["Fitness_SD"]  = fitness_sd4
df4["Fitness_Mean"]  = fitness_mean4
df4["Fitness_Lower"]  = df4["Fitness_Mean"] + df4["Fitness_SD"]
df4["Fitness_Upper"]  = df4["Fitness_Mean"] - df4["Fitness_SD"]

df5["Generation"]  = generations5
df5["Fitness_SD"]  = fitness_sd5
df5["Fitness_Mean"]  = fitness_mean5
df5["Fitness_Lower"]  = df5["Fitness_Mean"] + df5["Fitness_SD"]
df5["Fitness_Upper"]  = df5["Fitness_Mean"] - df5["Fitness_SD"]

df6["Generation"]  = generations6
df6["Fitness_SD"]  = fitness_sd6
df6["Fitness_Mean"]  = fitness_mean6
df6["Fitness_Lower"]  = df6["Fitness_Mean"] + df6["Fitness_SD"]
df6["Fitness_Upper"]  = df6["Fitness_Mean"] - df6["Fitness_SD"]

df7["Generation"]  = generations7
df7["Fitness_SD"]  = fitness_sd7
df7["Fitness_Mean"]  = fitness_mean7
df7["Fitness_Lower"]  = df7["Fitness_Mean"] + df7["Fitness_SD"]
df7["Fitness_Upper"]  = df7["Fitness_Mean"] - df7["Fitness_SD"]

df8["Generation"]  = generations8
df8["Fitness_SD"]  = fitness_sd8
df8["Fitness_Mean"]  = fitness_mean8
df8["Fitness_Lower"]  = df8["Fitness_Mean"] + df8["Fitness_SD"]
df8["Fitness_Upper"]  = df8["Fitness_Mean"] - df8["Fitness_SD"]

df9["Generation"]  = generations9
df9["Fitness_SD"]  = fitness_sd9
df9["Fitness_Mean"]  = fitness_mean9
df9["Fitness_Lower"]  = df9["Fitness_Mean"] + df9["Fitness_SD"]
df9["Fitness_Upper"]  = df9["Fitness_Mean"] - df9["Fitness_SD"]

df10["Generation"]  = generations10
df10["Fitness_SD"]  = fitness_sd10
df10["Fitness_Mean"]  = fitness_mean10
df10["Fitness_Lower"]  = df10["Fitness_Mean"] + df10["Fitness_SD"]
df10["Fitness_Upper"]  = df10["Fitness_Mean"] - df10["Fitness_SD"]

if not path.exists( log_dir0 ):
    mkdir( log_dir0 )

df0.to_excel( log_dir0 + "/all.xlsx", index = False, encoding = 'utf-8' )

if not path.exists( log_dir1 ):
    mkdir( log_dir1 )

df1.to_excel( log_dir1 + "/all.xlsx", index = False, encoding = 'utf-8' )

if not path.exists( log_dir2 ):
    mkdir( log_dir2 )

df2.to_excel( log_dir2 + "/all.xlsx", index = False, encoding = 'utf-8' )

if not path.exists( log_dir3 ):
    mkdir( log_dir3 )

df3.to_excel( log_dir3 + "/all.xlsx", index = False, encoding = 'utf-8' )

if not path.exists( log_dir4 ):
    mkdir( log_dir4 )

df4.to_excel( log_dir4 + "/all.xlsx", index = False, encoding = 'utf-8' )

if not path.exists( log_dir5 ):
    mkdir( log_dir5 )

df5.to_excel( log_dir5 + "/all.xlsx", index = False, encoding = 'utf-8' )

if not path.exists( log_dir6 ):
    mkdir( log_dir6 )

df6.to_excel( log_dir6 + "/all.xlsx", index = False, encoding = 'utf-8' )

if not path.exists( log_dir7 ):
    mkdir( log_dir7 )

df7.to_excel( log_dir7 + "/all.xlsx", index = False, encoding = 'utf-8' )

if not path.exists( log_dir8 ):
    mkdir( log_dir8 )

df8.to_excel( log_dir8 + "/all.xlsx", index = False, encoding = 'utf-8' )

if not path.exists( log_dir9 ):
    mkdir( log_dir9 )

df9.to_excel( log_dir9 + "/all.xlsx", index = False, encoding = 'utf-8' )

if not path.exists( log_dir10 ):
    mkdir( log_dir10 )

df10.to_excel( log_dir10 + "/all.xlsx", index = False, encoding = 'utf-8' )


print("Cycle Crossover with probability 0.0:", df0["Fitness_Mean"].loc[df0["Generation"] == 250])
print("Cycle Crossover with probability 0.1:", df1["Fitness_Mean"].loc[df1["Generation"] == 250])
print("Cycle Crossover with probability 0.2:", df2["Fitness_Mean"].loc[df2["Generation"] == 250])
print("Cycle Crossover with probability 0.3:", df3["Fitness_Mean"].loc[df3["Generation"] == 250])
print("Cycle Crossover with probability 0.4:", df4["Fitness_Mean"].loc[df4["Generation"] == 250])
print("Cycle Crossover with probability 0.5:", df5["Fitness_Mean"].loc[df5["Generation"] == 250])
print("Cycle Crossover with probability 0.6:", df6["Fitness_Mean"].loc[df6["Generation"] == 250])
print("Cycle Crossover with probability 0.7:", df7["Fitness_Mean"].loc[df7["Generation"] == 250])
print("Cycle Crossover with probability 0.8:", df8["Fitness_Mean"].loc[df8["Generation"] == 250])
print("Cycle Crossover with probability 0.9:", df9["Fitness_Mean"].loc[df9["Generation"] == 250])
print("Cycle Crossover with probability 1.0:", df10["Fitness_Mean"].loc[df10["Generation"] == 250])

plot_performance_chart( df0,df1,df2,df3,df4,df5,df6,df7,df8,df9,df10 )

#[sum(sublist) for sublist in itertools.izip(*myListOfLists)]



