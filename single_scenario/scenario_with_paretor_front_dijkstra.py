# %%   

import math
import networkx as nx
from joblib import Memory
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

clear_cache = False
if clear_cache:
    memory.clear()


CONSTANTS = {
    'SWITCH_TIME': 100.0,    # s
    'SWITCH_ENERGY': 1.0,    # Wh
    'BATTERY_CAPACITY': 30.0,  # Wh
    'RECHARGE_TIME': 27000.0,   # s or 30000
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


def scenario_with_pareto_front():
    nodes = {
        1: (0, 800, 0),
        2: (0, 1150, 357),
        3: (0, 1400, 0),
        4: (0, 1940, 0),
        
        5: (0, 2300, 350),
        16: (0, 2660, 0),

        6: (-1259, 800, 0),
        8: (-1259, 950, 250),
        
        9: (0, 0, 0),

        0: (0, 0, 0),
        7: (0, 3700, 0),

        10: (-3658, -692, 0),
        11: (-5800, 1100, 0),
        12: (-3460, 3200, 0),

        13: (3600, 2000, 0),
        15: (3661, 2687, 0),
        17: ((3600+3661)/2, (2000+2687)/2, 238),
        
        14: (7286, 2400, 0),

        18: (4900, 2400, 0),

        19: (-4800, 1100, 150),
        
        20: (-1700, -1200, 0),

        21: (500, 3550, 0), # increases number of paths, but not pareto front

        22: (-50+1200, 700, 0),
        23: (-50+1000, 1250, 0),
        24: (-50+1000, 1350, 160),
        26: (-50+560, 2150, 202),


    }
    edges = [

        (0, 22, "water"),
        (22, 23, "water"),
        (23, 24, "cliff"),
        (24, 26, "grass"),
        (26, 7, "slope"),

        (0, 1, "grass"),
        (1, 2, "cliff"),
        (2, 3, "cliff"),
        (3, 4, "grass"),
        (4, 5, "cliff"),
        (5, 16, "cliff"),

        (8, 16, "grass"),
        # (11, 19, "cliff"), # adding this looks nice, but doesnt change anything computation 120s
        # (12, 19, "slope"), # adding this looks nice, but doesnt change anything computation 120s

        (13, 18, "grass"),
        (18, 15, "grass"),

        (6, 19, "grass"),
        (19, 7, "slope"),

        (0, 20, "grass"),
        (20, 6, "grass"),

        # (16, 21, "grass"),
        # (21, 7, "water"),
        # (2, 8, "grass"),

        (0, 6, "water"),

        (6, 8, "cliff"),
        (8, 16, "slope"),
        (16, 7, "grass"),
        
        # (0, 10, "grass"),
        (0, 20, "grass"),
        (20, 10, "grass"),
        (10, 11, "grass"),
        (11, 12, "grass"),
        (12, 7, "grass"),

        (0, 13, "water"),
        (0, 22, "water"),
        (22, 13, "water"),
        (13, 14, "water"),
        (14, 15, "water"),
        (15, 7, "water"),

        (13, 17, "cliff"),
        (17, 15, "cliff"),

    ]
  

    G = nx.Graph()
    for node, (x, y, height) in nodes.items():
        G.add_node(node, x=x, y=y, height=height)

    for (u, v, terrain) in edges:
        G.add_edge(u, v, terrain=terrain)
    for (u, v) in G.edges():
        x_u, y_u, z_u = G.nodes[u]['x'], G.nodes[u]['y'], G.nodes[u]['height']
        x_v, y_v, z_v = G.nodes[v]['x'], G.nodes[v]['y'], G.nodes[v]['height']
        dx = x_u - x_v
        dy = y_u - y_v
        dz = z_u - z_v
        G[u][v]['distance'] = math.sqrt(dx*dx + dy*dy + dz*dz)

    return G


scenario = scenario_with_pareto_front()

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

def group_meta_paths_by_mode_number(meta_paths):
    groups = {}
    for p in meta_paths:
        used = frozenset(mode for (_, mode) in p.path_obj.path[1:-1])
        count = len(used)
        if count:
            groups.setdefault(count, []).append(p)
    return groups

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


pf = compute_pareto_front(grouped_by_number[4])
for path in pf:
    visualize_world_with_multiline_3D(G_world, L, path.path_obj, CONSTANTS, label_option="all_edges")


# Define markers and colors
markers = cycle(['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', '+'])
colors = cycle(plt.cm.tab10.colors)

# Figure 1: Travel Time vs. Distance
fig1, ax1 = plt.subplots(figsize=(5, 7))
for combo, paths in grouped.items():
    marker = next(markers)
    color = next(colors)
    label = ",".join(sorted(combo))
    times = [p.total_time for p in paths]
    distances = [sum(p.mode_distances.values()) for p in paths]
    ax1.scatter(times, distances, marker=marker, color=color, label=label)
ax1.set_xlabel("Travel Time (s)")
ax1.set_ylabel("Travel Distance (m)")
ax1.set_title("Travel Time vs. Distance")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(title="Mode Combination", fontsize=8)
plt.tight_layout()

# Figure 2: Travel Time vs. Energy
fig2, ax2 = plt.subplots(figsize=(5, 7))
for combo, paths in grouped.items():
    marker = next(markers)
    color = next(colors)
    label = ",".join(sorted(combo))
    times = [p.total_time for p in paths]
    energies = [p.total_energy for p in paths]
    ax2.scatter(times, energies, marker=marker, color=color, label=label)

for count, paths in grouped_by_number.items():
    pf = compute_pareto_front(paths)
    if not pf:
        continue
    pf_sorted = sorted(pf, key=lambda p: p.total_time)
    times_pf = [p.total_time for p in pf_sorted]
    energies_pf = [p.total_energy for p in pf_sorted]
    ax2.plot(times_pf, energies_pf, linestyle="--", marker=None, color="black")

ax2.set_xlabel("Travel Time (s)")
ax2.set_ylabel("Total Energy (Wh)")
ax2.set_title("Travel Time vs. Energy")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(title="Mode Combo / Pareto Front", fontsize=8)
plt.tight_layout()

plt.show()


# %%