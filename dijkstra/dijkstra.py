# %%
    
# distributions change/sensitityv w.r.t map/scenario variation
    # add scenario for straight path on grass
    # add sceanrio for straight path on water
    # add scenario where we have to fly up a cliff (without rolling)
    # add scenario with two slopes
    
    # 3 random ones from a maps


# sensitivity to robot/parameter changes
    # delta 12 parameters

    # take all above scenarios, 
    # look at the paths of the pareto front
    # vary one parameter at a time, 
    # see how energy, time, mode change second pareto front 
    # makes correspondece between good paths in both runs

# sensitivity to multimodality
    # take each scenario
    # get the pareto points, min time and min energy, no. solutions
    # remove each modality afterwards
    # recalculate metrics
    # result is a table scenario 1,2,3,4 vs. no mode
    # no flying, no rolling, no swimming, no driving, no recharging


# how to select good costs and how does that change the path
# time in motion / resting
# minimal distance vs. minimal time
# how to quantify risk?




# %%

%reload_ext autoreload
%autoreload 2

%matplotlib widget

import os
import copy
import math
import random
import networkx as nx
from joblib import Memory, Parallel, delayed

# Imports from your modules
from dijkstra_algorithm import layered_dijkstra_with_battery, find_all_feasible_paths, analyze_paths, compute_pareto_front, build_layered_graph
from dijkstra_visualize import visualize_world_with_multiline_3D, plot_basic_metrics, plot_stacked_bars, visualize_param_variations



# %%

class SensitivityConstants:
    def __init__(self, constants, variation):
        self.constants = constants
        self.variation = variation

    def __iter__(self):
        # Yield baseline as a deep copy.
        baseline = copy.deepcopy(self.constants)
        yield baseline
        
        # Recursively collect all paths to numeric parameters.
        numeric_paths = []
        def traverse(d, path):
            for key, value in d.items():
                if isinstance(value, (int, float)):
                    numeric_paths.append(path + [key])
                elif isinstance(value, dict):
                    traverse(value, path + [key])
        traverse(self.constants, [])
        
        # For each numeric parameter, yield two variations: +variation and -variation.
        for path in numeric_paths:
            for factor in [1 + self.variation, 1 - self.variation]:
                new_constants = copy.deepcopy(self.constants)
                # Navigate to the value to be modified.
                d = new_constants
                for key in path[:-1]:
                    d = d[key]
                d[path[-1]] *= factor
                yield new_constants


CONSTANTS = {
    'SWITCH_TIME': 100.0,    # s
    'SWITCH_ENERGY': 1.0,    # Wh
    'BATTERY_CAPACITY': 30.0,  # Wh
    'RECHARGE_TIME': 30000.0,   # s or 30000
    'MODES': {
        'fly':   {'speed': 10.0,  'power': 1000.0},  # m/s, W
        'swim':  {'speed': 0.5,  'power':   10.0},
        'roll':  {'speed': 3.0,  'power':    1.0},
        'drive': {'speed': 1.0,  'power':   30.0},
    }
}

# print("Parameter variations: ")
# for idx, const in enumerate(SensitivityConstants(CONSTANTS)):
#     print(f"Variation {idx}: {const}")

# print("-" * 40)
    

START = (0, 'drive')
GOAL = (7, 'drive')
ENERGY_VS_TIME = 0.0

# Create a Joblib Memory object for caching.
memory = Memory("cache_dir", verbose=0)

@memory.cache
def compute_for_scenario(name, graph, constants):
    print(f"Processing scenario: {name} with constants: {constants}")
    G_world = graph
    L = build_layered_graph(G_world, constants)
    optimal_path = layered_dijkstra_with_battery(G_world, L, START, GOAL, constants, energy_vs_time=ENERGY_VS_TIME)
    all_feasible_paths = find_all_feasible_paths(G_world, L, START, GOAL, constants, energy_vs_time=ENERGY_VS_TIME)
    meta_paths = analyze_paths(all_feasible_paths, constants)
    pareto_front = compute_pareto_front(meta_paths)   
    return name, {
        "G_world": G_world,
        "L": L,
        "optimal_path": optimal_path,
        "all_feasible_paths": all_feasible_paths,
        "meta_paths": meta_paths,
        "pareto_front": pareto_front
    }


# %%

import pickle
from dijkstra_scenario import PremadeScenarios

all_scenarios = PremadeScenarios.get_all()
all_variations = list(SensitivityConstants(CONSTANTS, variation=0.5)) # 0.2, 0.5


def process_variation(idx, var_constants):
    print(f"\n--- Processing parameter variation {idx} ---")
    
    results_list = []
    for name, graph in all_scenarios.items():
        results_list.append(compute_for_scenario(name, graph, constants=var_constants))

    scenario_results = {name: data for name, data in results_list}
    return idx, {"constants": var_constants, "results": scenario_results}



recompute = True
pickle_file = "all_results.pkl"

if os.path.exists(pickle_file) and not recompute:
    print("Loading all_results from pickle file...")
    with open(pickle_file, "rb") as f:
        all_results = pickle.load(f)
else:
    print("Computing all_results...")
   
    all_results_list = Parallel(n_jobs=-1)(
        delayed(process_variation)(idx, var_constants)
        for idx, var_constants in enumerate(all_variations)
    )

    all_results = {idx: data for idx, data in all_results_list}

    with open(pickle_file, "wb") as f:
        pickle.dump(all_results, f)
    print("Computed and saved all_results.")


# %%

###############################################################################
# Visualization of parameter variations for a single scenario
###############################################################################
for scenario in all_scenarios:
    print(f"Scenario: {scenario}")
    visualize_param_variations(all_results, scenario)



# %% 


# For singel example scneario

# name = "scenario_2"
# graph = all_scenarios[name]
# var_constants = all_variations[0]

# results_list = [compute_for_scenario(name, graph, constants=var_constants)]
# scenario_results = {name: data for name, data in results_list}

# all_results_list = [(0, {"constants": var_constants, "results": scenario_results})]
# all_results = {idx: data for idx, data in all_results_list}


# %%
###############################################################################
# Visualization of a single scenario for single parameter variation
###############################################################################
selected_variation = 0
selected_scenario = "scenario_2"
if selected_scenario in all_results[selected_variation]["results"]:
    constants = all_results[selected_variation]["constants"]
    data = all_results[selected_variation]["results"][selected_scenario]
    G_world = data["G_world"]
    L = data["L"]
    optimal_path = data["optimal_path"]
    meta_paths = data["meta_paths"]
    pareto_front = data["pareto_front"]

    # # For example, visualize metrics for baseline:
    visualize_world_with_multiline_3D(G_world, L, optimal_path, constants, label_option="all_edges")
    # print("Optimal Path:")
    # print(optimal_path)
    # print("-----")

    # plot_basic_metrics(meta_paths, pareto_front, optimal_path)
else:
    print(f"Scenario {selected_scenario} not found in variation {selected_variation}.")

# %%