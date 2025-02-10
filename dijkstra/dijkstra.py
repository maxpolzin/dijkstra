# %%

    # distributions change/sensitityv w.r.t map/scenario variation
        # add scenario for straight path on grass DONE
        # add sceanrio for straight path on water DONE
        # add scenario where we have to fly up a cliff (without rolling) DONE
        # add scenario with two slopes DONE
        
        # 3 random ones from a maps DONE


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

    # can we get optimal pareto paths 0.1/0.9 for energy/cost from higher resolution grids with complex graph



# %%

%reload_ext autoreload
%autoreload 2

%matplotlib widget

import os
import math
import random
import networkx as nx
from joblib import Memory

# Imports from your modules
from dijkstra_scenario import build_world_graph, build_layered_graph, PremadeScenarios
from dijkstra_visualize import visualize_world_with_multiline_3D, plot_basic_metrics, plot_stacked_bars
from dijkstra_algorithm import layered_dijkstra_with_battery, find_all_feasible_paths, analyze_paths

# Define your modes and constants
MODES = {
    'fly':   {'speed': 10.0,  'power': 1000.0},  # m/s, W
    'swim':  {'speed': 0.5,  'power':   10.0},
    'roll':  {'speed': 3.0,  'power':    1.0},
    'drive': {'speed': 1.0,  'power':   30.0},
}

CONSTANTS = {
    'SWITCH_TIME': 100.0,    # s time penalty for mode switch
    'SWITCH_ENERGY': 1.0,    # Wh penalty for switching
    'BATTERY_CAPACITY': 30.0,  # Wh
    'RECHARGE_TIME': 1000.0,   # s
}

start = (0, 'drive')
goal = (7, 'drive')

# Create a Joblib Memory object for caching.
memory = Memory("cache_dir", verbose=0)

# Now define a function to compute all scenario results and decorate it.
@memory.cache
def compute_all_results(modes, constants, start, goal):
    results = {}
    all_scenarios = PremadeScenarios.get_all()
    for name, graph in all_scenarios.items():
        print(f"Processing scenario: {name}")
        G_world = graph
        L = build_layered_graph(G_world, modes, constants)
        optimal_path = layered_dijkstra_with_battery(G_world, L, start, goal, modes, constants, energy_vs_time=0.5)
        all_feasible_paths = find_all_feasible_paths(G_world, L, start, goal, constants=constants)
        meta_paths = analyze_paths(all_feasible_paths, constants)
        results[name] = {
            "G_world": G_world,
            "L": L,
            "optimal_path": optimal_path,
            "all_feasible_paths": all_feasible_paths,
            "meta_paths": meta_paths
        }
    return results

# Now, get the results (this call will load from disk if already computed).
results = compute_all_results(MODES, CONSTANTS, start, goal)

###############################################################################
# Visualization of a single scenario
###############################################################################
# For example, visualize the results for "scenario_0"
selected_scenario = "scenario_0"
if selected_scenario in results:
    data = results[selected_scenario]
    G_world = data["G_world"]
    L = data["L"]
    optimal_path = data["optimal_path"]
    meta_paths = data["meta_paths"]

    visualize_world_with_multiline_3D(G_world, L, optimal_path, CONSTANTS, label_option="traveled_only")
    print("Optimal Path:")
    print(optimal_path)

    plot_basic_metrics(meta_paths)
    plot_stacked_bars(meta_paths, sort_xticks_interval=10)
else:
    print(f"Scenario {selected_scenario} not found in the results.")
