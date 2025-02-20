# %%   

import os
import copy
import math
import random
import pickle
import networkx as nx
from joblib import Memory, Parallel, delayed
import matplotlib.pyplot as plt

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
memory = Memory("cache_dir", verbose=0)

# %%

from dijkstra_algorithm import find_all_feasible_paths, analyze_paths, build_layered_graph
from dijkstra_scenario import PremadeScenarios

clear_cache = False
if clear_cache:
    memory.clear()


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

START = (0, 'drive')
GOAL = (7, 'drive')


# @memory.cache
def compute_for_scenario(graph, constants):
    print(f"Processing scenario with constants: {constants}")
    G_world = graph
    L = build_layered_graph(G_world, constants)
    all_feasible_paths = find_all_feasible_paths(G_world, L, START, GOAL, constants)
    meta_paths = analyze_paths(all_feasible_paths, constants)
    return G_world, L, meta_paths


scenario = PremadeScenarios.scenario_1()

G_world, L, meta_paths = compute_for_scenario(scenario, constants=CONSTANTS)





from dijkstra_visualize import visualize_world_with_multiline_3D


import itertools

def group_meta_paths_by_modes(meta_paths, mode_list=["roll", "swim", "drive", "fly"]):
    combos = [frozenset(combo) for r in range(1, len(mode_list) + 1) for combo in itertools.combinations(mode_list, r)]
    groups = {combo: [] for combo in combos}
    for p in meta_paths:
        used = frozenset(mode for (_, mode) in p.path_obj.path)
        if used in groups:
            groups[used].append(p)
    return {k: v for k, v in groups.items() if v}

grouped = group_meta_paths_by_modes(meta_paths)
for k, v in grouped.items():
    print(k, len(v))

#  %%


# for path in grouped[frozenset({'swim', 'fly', 'drive', 'roll'})]:
#     print(path.path_obj)
#     visualize_world_with_multiline_3D(G_world, L, path.path_obj, CONSTANTS, label_option="traveled_only")



import matplotlib.pyplot as plt
from itertools import cycle

grouped = group_meta_paths_by_modes(meta_paths)
markers = cycle(['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+'])
colors = cycle(plt.cm.tab10.colors)

fig, axs = plt.subplots(1, 2, figsize=(11, 5))
for combo, paths in grouped.items():
    times = [p.total_time for p in paths]
    distances = [sum(p.mode_distances.values()) for p in paths]
    energies = [p.total_energy for p in paths]
    marker = next(markers)
    color = next(colors)
    label = ",".join(sorted(combo))
    axs[0].scatter(times, distances, marker=marker, color=color, label=label)
    axs[1].scatter(times, energies, marker=marker, color=color, label=label)
axs[0].set_xlabel("Travel Time (s)")
axs[0].set_ylabel("Travel Distance (m)")
axs[0].set_title("Travel Time vs. Distance")
axs[1].set_xlabel("Travel Time (s)")
axs[1].set_ylabel("Total Energy (Wh)")
axs[1].set_title("Travel Time vs. Energy")
axs[0].legend(title="Mode Combination", fontsize=8)
plt.tight_layout()
plt.show()












# %%


import itertools

import matplotlib.pyplot as plt

def get_modes_used(p):
    return set(mode for (_, mode) in p.path_obj.path)

def plot_paths(meta_paths, ax=None, ):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    mode_list=["roll", "swim", "drive", "fly"]
    combos = []
    for r in range(1, len(mode_list) + 1):
        combos.extend(itertools.combinations(mode_list, r))
    
    cmap = plt.get_cmap("tab20")
    total = len(combos)
    print(combos)
    
    for idx, combo in enumerate(combos):
        combo_set = set(combo)
        filtered = [p for p in meta_paths if get_modes_used(p) <= combo_set]
        if not filtered:
            continue
        
        # plot





selected_keys = list(all_scenarios.keys())[:]

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

        # visualize_world_with_multiline_3D(G_world, L, optimal_path, constants, label_option="all_edges")
        # plot_basic_metrics(meta_paths, pareto_front, optimal_path)
        # plot_stacked_bars(meta_paths)

        # visualize_param_variations(all_results, selected_scenario)

        # visualize_pareto_fronts(all_results, selected_scenario)

        plot_pareto_front_distance_vs_time(pareto_front, L, constants)
        # plot_path_distance_vs_time(meta_paths, L, constants, n_paths=100)
        
        plot_pareto_fronts_all_combinations(meta_paths, mode_list=["fly", "swim", "roll", "drive"])
        plt.show()

    else:
        print(f"Scenario {selected_scenario} not found in variation {selected_variation}.")




# ('roll',)
# ('swim',)
# ('drive',)
# ('fly',) 
# ('roll', 'swim')
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