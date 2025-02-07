# %%

%reload_ext autoreload
%autoreload 2

%matplotlib widget


import numpy as np
import matplotlib.pyplot as plt

from dijkstra_scenario import build_world_graph, build_layered_graph
from dijkstra_visualize import visualize_world_with_multiline_3D
from dijkstra_algorithm import layered_dijkstra_with_battery, find_all_feasible_paths, analyze_paths


MODES = {
    'fly':   {'speed': 5.0,  'power': 1000.0},  # m/s, W
    'swim':  {'speed': 0.5,  'power':   10.0}, # Try 0.15 vs 0.16
    'roll':  {'speed': 3.0,  'power':    1.0},
    'drive': {'speed': 1.0,  'power':   30.0},
}

CONSTANTS = {
    'SWITCH_TIME': 100.0,  # s time penalty for mode switch
    'SWITCH_ENERGY': 1.0,  # Wh penalty for switching
    'BATTERY_CAPACITY': 30,  # Wh
    'RECHARGE_TIME': 3000.0,  # s
}


start = (0, 'drive')    
goal = (7, 'drive')

# straight_grass
# straight_water
# flat_slope_flat
# fly_up_cliff


G_world=build_world_graph(id='fly_up_cliff')
L=build_layered_graph(G_world, MODES, CONSTANTS)


optimal_path = layered_dijkstra_with_battery(G_world, L, start, goal, MODES, CONSTANTS,energy_vs_time=0.5)

visualize_world_with_multiline_3D(G_world, L, optimal_path, CONSTANTS, label_option="traveled_only")

print(optimal_path)




# %%

all_feasible_paths = find_all_feasible_paths(G_world, L, start, goal, constants=CONSTANTS)

for path in all_feasible_paths:
    print(path)

meta_paths = analyze_paths(all_feasible_paths, CONSTANTS)

plot_basic_metrics(meta_paths)

plot_stacked_bars(meta_paths, sort_xticks_interval=10)


# %%
