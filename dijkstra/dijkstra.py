# %%

# main_experiment.py
import os
import pickle
import math
import random
import networkx as nx

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

###############################################################################
# Load (or compute) results for all scenarios
###############################################################################
RESULTS_FILE = "premade_scenarios_results.pkl"

if os.path.exists(RESULTS_FILE):
    print("Loading results from file...")
    with open(RESULTS_FILE, "rb") as f:
        results = pickle.load(f)
else:
    print("Computing results for all premade scenarios...")
    results = {}
    all_scenarios = PremadeScenarios.get_all()
    for name, graph in all_scenarios.items():
        print(f"Processing scenario: {name}")
        G_world = graph
        L = build_layered_graph(G_world, MODES, CONSTANTS)
        optimal_path = layered_dijkstra_with_battery(G_world, L, start, goal, MODES, CONSTANTS, energy_vs_time=0.5)
        all_feasible_paths = find_all_feasible_paths(G_world, L, start, goal, constants=CONSTANTS)
        meta_paths = analyze_paths(all_feasible_paths, CONSTANTS)
        # Store all data you might later need.
        results[name] = {
            "G_world": G_world,
            "L": L,
            "optimal_path": optimal_path,
            "all_feasible_paths": all_feasible_paths,
            "meta_paths": meta_paths
        }
    print("Saving results to file...")
    with open(RESULTS_FILE, "wb") as f:
        pickle.dump(results, f)

#%%

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


# %%