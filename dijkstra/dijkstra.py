# %%
    
# distributions change/sensitityv w.r.t map/scenario variation
    # add scenario for straight path on grass
    # add sceanrio for straight path on water
    # add scenario with cliff
    # add scenario with slopes

    # 3 random ones from a maps
        # 1 same traveltime for all modes
        # 1 same energy for all modes
        # 1 where robot must recharge?
        # 1 random one that shows complexity of scenarios

# I can make scenarios that require each mode and or otherwise infeasible



# sensitivity to robot/parameter changes
    # delta 12 parameters
    # take all above scenarios, 
    # look at the paths of the pareto front
    # vary one parameter at a time, 
    # see how energy, time, mode change
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



# plot pareto fronts for parameter variations -> Done
# get rid of crosses -> Done

# Paretor front is only optimal for a certain metric
# plot pareto front with loss of modality


# resource and risk variance in a scenario
# each pareto path makes sense for certain environmental conditions
# optimal for given risk/resource conditions


# How does the behavavioral richness change with the number of modes?



# Paretofront: Multi-objective optimization
#  What are our objectives?

# Minimize travel time / Get from A to B
# Maximize travel distance? Proxy for maximize explored area?

# For all: 
# Maximize survival likelihood / Minimize risk of failure
# -> minimize energy consumption / resource uncertainty




# %%

def setup_autoreload():
    try:
        ip = get_ipython()
        if ip is not None:
            ip.run_line_magic("reload_ext", "autoreload")
            ip.run_line_magic("autoreload", "2")
            ip.run_line_magic("matplotlib", "widget")
    except Exception:
        pass

setup_autoreload()



import os
import copy
import math
import random
import pickle
import networkx as nx
from joblib import Memory, Parallel, delayed
import matplotlib.pyplot as plt


# Imports from your modules
from dijkstra_algorithm import layered_dijkstra_with_battery, find_all_feasible_paths, analyze_paths, compute_pareto_front, build_layered_graph
from dijkstra_visualize import visualize_world_with_multiline_3D, plot_basic_metrics, plot_stacked_bars, visualize_param_variations, visualize_pareto_fronts, plot_pareto_front_distance_vs_time, plot_path_distance_vs_time


# %%

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



from dijkstra_scenario import PremadeScenarios

clear_cache = False
if clear_cache:
    memory.clear()

recompute = True
n_jobs = -1
pickle_file = "all_results.pkl"

all_scenarios = PremadeScenarios.get_all()
all_variations = list(SensitivityConstants(CONSTANTS, variation=0.3))[:1]

if os.path.exists(pickle_file) and not recompute:
    print("Loading all_results from pickle file...")
    with open(pickle_file, "rb") as f:
        all_results = pickle.load(f)
else:
    print("Computing all_results...")

    def process_variation(idx, var_constants):
        print(f"\n--- Processing parameter variation {idx} ---")
        
        results_list = []
        for name, graph in all_scenarios.items():
            results_list.append(compute_for_scenario(name, graph, constants=var_constants))

        scenario_results = {name: data for name, data in results_list}
        return idx, {"constants": var_constants, "results": scenario_results}

    all_results_list = Parallel(n_jobs=n_jobs)(
        delayed(process_variation)(idx, var_constants)
        for idx, var_constants in enumerate(all_variations)
    )

    all_results = {idx: data for idx, data in all_results_list}

    with open(pickle_file, "wb") as f:
        pickle.dump(all_results, f)
    print("Computed and saved all_results.")


# %%


selected_keys = list(all_scenarios.keys())[0:1]

for selected_scenario in selected_keys:
    selected_variation = 0
    if selected_scenario in all_results[selected_variation]["results"]:
        constants = all_results[selected_variation]["constants"]
        data = all_results[selected_variation]["results"][selected_scenario]
        G_world = data["G_world"]
        L = data["L"]
        optimal_path = data["optimal_path"]
        meta_paths = data["meta_paths"]
        pareto_front = data["pareto_front"]
        
        print("Optimal Path:")
        print(optimal_path)
        print("-----")

        # print first 10 paths
        selected_paths = sorted(meta_paths, key=lambda m: m.total_time)[:10]

        for meta_path in selected_paths:
            print(meta_path.path_obj)
            print("-----")

        visualize_world_with_multiline_3D(G_world, L, optimal_path, constants, label_option="all_edges")
        plot_basic_metrics(meta_paths, pareto_front, optimal_path)
        plot_stacked_bars(meta_paths)
        visualize_param_variations(all_results, selected_scenario)
        plot_pareto_front_distance_vs_time(pareto_front, L, constants)
        visualize_pareto_fronts(all_results, selected_scenario)
        plot_path_distance_vs_time(meta_paths, L, constants, n_paths=100)
        plt.show()

    else:
        print(f"Scenario {selected_scenario} not found in variation {selected_variation}.")




# ('roll',) <- makes no sense
# ('swim',) <- makes no sense
# ('drive',)
# ('fly',) 
# ('roll', 'swim') <- makes no sense
# ('roll', 'drive')
# ('roll', 'fly')
# ('swim', 'drive')
# ('swim', 'fly')
# ('drive', 'fly')
# ('roll', 'swim', 'drive')
# ('roll', 'swim', 'fly')
# ('roll', 'drive', 'fly')
# ('swim', 'drive', 'fly')
# ('roll', 'swim', 'drive', 'fly')





# %%