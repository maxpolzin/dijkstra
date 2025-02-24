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
    # return G_world, L, []
    return G_world, L, meta_paths



def test1():
    nodes = {
        0: (0, 0, 0),
        1: (0, 0, 0),
        2: (0, 0, 0),
        3: (0, 0, 0),
        4: (0, 0, 100),
        
        5: (0, 0, 100),
        6: (0, 0, 100),
       
        7: (0, 0, 0),

        8: (0, 0, 0),
        9: (0, 0, 0),

        10: (0, 0, 0),
        11: (0, 0, 0),
        12: (0, 0, 0),
        13: (0, 0, 0),


    }
    edges = [
        (0, 1, "grass", 1040),
        (1, 2, "grass", 1040),
        (2, 3, "water", 1000),
        (3, 4, "cliff", 100),
        (4, 7, "slope", 3600),
        
        (2, 5, "grass", 2000),
        (5, 4, "grass", 3400),

        
        (3, 6, "grass", 1100),
        (6, 4, "grass", 2600),

        (0, 8, "water", 4800),
        (8, 9, "water", 4800),
        (9, 7, "water", 4800),

        (0, 10, "cliff", 825),
        (10, 11, "cliff", 825),
        (11, 13, "cliff", 825),
        (13, 7, "cliff", 825)

    ]
    G = nx.Graph()
    for node, (x, y, height) in nodes.items():
        G.add_node(node, x=x, y=y, height=height)
    for u, v, terrain, distance in edges:
        G.add_edge(u, v, terrain=terrain, distance=distance)
    print("Built test1.")
    return G




def test2():
    nodes = {
        0: (0, 0, 0),
        1: (0, 800, 0),
        2: (0, 1150, 357),
        3: (0, 1400, 0),
        4: (0, 1940, 0),
        
        5: (0, 2300, 350),
        16: (0, 2660, 0),
        7: (0, 3700, 0),

        6: (-1259, 800, 0),
        8: (-1259, 950, 250),
        
        9: (0, 0, 0),

        10: (-2890, 0, 0),
        11: (-6000, 1800, 0),
        12: (-3500, 3700, 0),

        13: (3600, 2000, 0),
        15: (3661, 2687, 0),
        17: ((3600+3661)/2, (2000+2687)/2, 238),
        
        14: (7286, 2400, 0),

        18: (4900, 2400, 0),

        19: (-4800, 1100, 150)


    }
    edges = [
        (0, 1, "grass", 800),
        (1, 2, "cliff", 500),
        (2, 3, "cliff", 500),
        (3, 4, "grass", 540),
        (4, 5, "cliff", 500),
        (5, 16, "cliff", 500),

        # (4, 16, "grass", 2400),
        (8, 16, "grass", 2400),

        (13, 18, "grass", 2400),
        (18, 15, "grass", 2400),

        (6, 19, "grass", 2400),
        (19, 7, "slope", 2400),


        (0, 6, "water", 1200),
        # (1, 6, "grass", 2150),

        (6, 8, "cliff", 180),
        (8, 16, "slope", 2300),
        (16, 7, "grass", 1040),
        
        # (10, 15, "grass", 1400),
        # (16, 15, "grass", 1400),

        (0, 10, "grass", 3600),
        (10, 11, "grass", 3600),
        (11, 12, "grass", 3600),
        (12, 7, "grass", 1800),

        (0, 13, "water", 4700),
        (13, 14, "water", 4700),
        (14, 15, "water", 4700),
        (15, 7, "water", 3800),

        (13, 17, "cliff", 125),
        (17, 15, "cliff", 125),

    ]

    G = nx.Graph()
    for node, (x, y, height) in nodes.items():
        G.add_node(node, x=x, y=y, height=height)
    # for u, v, terrain, distance in edges:
    #     G.add_edge(u, v, terrain=terrain, distance=distance)

    for (u, v, terrain, _) in edges:
        G.add_edge(u, v, terrain=terrain)
    for (u, v) in G.edges():
        x_u, y_u, z_u = G.nodes[u]['x'], G.nodes[u]['y'], G.nodes[u]['height']
        x_v, y_v, z_v = G.nodes[v]['x'], G.nodes[v]['y'], G.nodes[v]['height']
        dx = x_u - x_v
        dy = y_u - y_v
        dz = z_u - z_v
        G[u][v]['distance'] = math.sqrt(dx*dx + dy*dy + dz*dz)

    return G


# scenario = PremadeScenarios.test1()
scenario = test2()

G_world, L, meta_paths = compute_for_scenario(scenario, constants=CONSTANTS)

from dijkstra_visualize import visualize_world_with_multiline_3D





import itertools
import matplotlib.pyplot as plt
from itertools import cycle

def group_meta_paths_by_modes(meta_paths, mode_list=["roll", "swim", "drive", "fly"]):
    combos = [frozenset(combo) for r in range(1, len(mode_list) + 1) for combo in itertools.combinations(mode_list, r)]
    groups = {combo: [] for combo in combos}
    for p in meta_paths:
        used = frozenset(mode for (_, mode) in p.path_obj.path[1:-1])
        if used in groups:
            groups[used].append(p)
    return {k: v for k, v in groups.items() if v}

# grouped = group_meta_paths_by_modes(meta_paths)
# for k, v in grouped.items():
#     print(k, len(v))


def group_meta_paths_by_mode_number(meta_paths):
    groups = {}
    for p in meta_paths:
        used = frozenset(mode for (_, mode) in p.path_obj.path[1:-1])
        count = len(used)
        if count:
            groups.setdefault(count, []).append(p)
    return groups

grouped_by_number = group_meta_paths_by_mode_number(meta_paths)
for k, v in grouped_by_number.items():
    print(k, len(v))

def compute_pareto_front(meta_paths):
    pareto = []
    for m in meta_paths:
        dominated = False
        for n in meta_paths:
            if n == m:
                continue
            if (n.total_time <= m.total_time and n.total_energy <= m.total_energy and
                (n.total_time < m.total_time or n.total_energy < m.total_energy)):
                dominated = True
                break
        if not dominated:
            pareto.append(m)
    return pareto



grouped = group_meta_paths_by_modes(meta_paths)
grouped_by_number = group_meta_paths_by_mode_number(meta_paths)

# for path in grouped_by_number[4]:
#     print(path.path_obj)
#     print("-----")

pf = compute_pareto_front(grouped_by_number[3])
for path in pf:
    visualize_world_with_multiline_3D(G_world, L, path.path_obj, CONSTANTS, label_option="traveled_only")


# pf = compute_pareto_front(grouped_by_number[4])
# for path in pf:
#     visualize_world_with_multiline_3D(G_world, L, path.path_obj, CONSTANTS, label_option="all_edges")



markers = cycle(['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+'])
colors = cycle(plt.cm.tab10.colors)

# fig, axs = plt.subplots(2, 1, figsize=(5, 7))

fig, ax = plt.subplots(figsize=(5, 4))
axs = [None, ax]

# Scatter plots for each mode combination (grouped by exact modes used)
for combo, paths in grouped.items():
    marker = next(markers)
    color = next(colors)
    label = ",".join(sorted(combo))
    times = [p.total_time for p in paths]
    distances = [sum(p.mode_distances.values()) for p in paths]
    energies = [p.total_energy for p in paths]
    # axs[0].scatter(times, distances, marker=marker, color=color, label=label)
    axs[1].scatter(times, energies, marker=marker, color=color, label=label)

# Overlay dashed lines for Pareto fronts computed per mode count
for count, paths in grouped_by_number.items():
    pf = compute_pareto_front(paths)
    if not pf:
        continue
    pf_sorted = sorted(pf, key=lambda p: p.total_time)
    times_pf = [p.total_time for p in pf_sorted]
    energies_pf = [p.total_energy for p in pf_sorted]
    label = f"{count} mode{'s' if count > 1 else ''} Pareto Front"
    axs[1].plot(times_pf, energies_pf, linestyle="--", marker=None, color="black", label=None)






# for ax in axs:
#     ax.set_xlim(0, 5000)
#     ax.grid(True, linestyle="--", alpha=0.5)

# axs[0].set_xlabel("Travel Time (s)")
# axs[0].set_ylabel("Travel Distance (m)")
# axs[0].set_title("Travel Time vs. Distance")
axs[1].set_xlabel("Travel Time (s)")
axs[1].set_ylabel("Total Energy (Wh)")
axs[1].set_title("Travel Time vs. Energy")
# axs[0].legend(title="Mode Combination", fontsize=8)
axs[1].legend(title="Mode Combo / Pareto Front", fontsize=8)
for ax in axs:
    if ax is not None:
        ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()



# for path in grouped[frozenset({'fly', 'drive'})]:
#     print(path.path_obj)
#     




# %%

import itertools

import matplotlib.pyplot as plt



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