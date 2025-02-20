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
import networkx as nx
from joblib import Memory, Parallel, delayed

# Imports from your modules
from dijkstra_algorithm import layered_dijkstra_with_battery, find_all_feasible_paths, analyze_paths, compute_pareto_front, build_layered_graph
from dijkstra_visualize import visualize_world_with_multiline_3D, plot_basic_metrics, plot_stacked_bars, visualize_param_variations, visualize_pareto_fronts



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
all_variations = list(SensitivityConstants(CONSTANTS, variation=0.3))


def process_variation(idx, var_constants):
    print(f"\n--- Processing parameter variation {idx} ---")
    
    results_list = []
    for name, graph in all_scenarios.items():
        results_list.append(compute_for_scenario(name, graph, constants=var_constants))

    scenario_results = {name: data for name, data in results_list}
    return idx, {"constants": var_constants, "results": scenario_results}



recompute = False
pickle_file = "all_results.pkl"

if os.path.exists(pickle_file) and not recompute:
    print("Loading all_results from pickle file...")
    with open(pickle_file, "rb") as f:
        all_results = pickle.load(f)
else:
    print("Computing all_results...")
   
    all_results_list = Parallel(n_jobs=3)(
        delayed(process_variation)(idx, var_constants)
        for idx, var_constants in enumerate(all_variations)
    )

    all_results = {idx: data for idx, data in all_results_list}

    with open(pickle_file, "wb") as f:
        pickle.dump(all_results, f)
    print("Computed and saved all_results.")


# %%


# For singel example scneario

# name = "scenario_3"
# graph = all_scenarios[name]
# var_constants = all_variations[0]

# results_list = [compute_for_scenario(name, graph, constants=var_constants)]
# scenario_results = {name: data for name, data in results_list}

# all_results_list = [(0, {"constants": var_constants, "results": scenario_results})]
# all_results = {idx: data for idx, data in all_results_list}


# %%



import matplotlib.pyplot as plt

def plot_pareto_front_distance_vs_time(pareto_front, L, constants, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Re-use your existing mode colors, but add 'switching' and 'recharging' if needed:
    mode_colors_local = {
        'fly':        'red',
        'drive':      'lightgreen',
        'roll':       'yellow',
        'swim':       'blue',
        'switching':  'gray',
        'recharging': 'black',
    }
    
    def plot_single_path_distance_time(path_obj, index=0):
        """
        Given a Path object (with path_obj.state_chain),
        draw the piecewise distance-vs-time line on `ax`.
        """
        state_chain = path_obj.state_chain
        if len(state_chain) < 2:
            return
        
        # We'll keep track of the 'current' time & distance in the 2D plot.
        # Start from t=0, d=0
        t_prev = 0.0
        d_prev = 0.0
        
        # For labeling the end of this path
        final_time = state_chain[-1].cum_time
        final_distance = 0.0  # We'll accumulate traveled distance as we go.
        
        for i in range(1, len(state_chain)):
            old_state = state_chain[i - 1]
            new_state = state_chain[i]
            
            # Total time for this step:
            dt = new_state.cum_time - old_state.cum_time
            
            # Check the edge in L to get traveled distance
            dist_travel = 0.0
            if L.has_edge((old_state.node, old_state.mode), (new_state.node, new_state.mode)):
                dist_travel = L[(old_state.node, old_state.mode)][(new_state.node, new_state.mode)].get('distance', 0.0)
            
            # If recharging took place, we must split the time into
            # recharge portion (horizontal, no distance) and travel portion (distance).
            # new_state.recharge_time = how many seconds were spent recharging.
            recharge_t = new_state.recharge_time
            travel_t = dt - recharge_t
            
            # 1) Plot any recharge portion (horizontal line, black color).
            #    This occurs only if recharge_t > 0.
            if recharge_t > 0:
                t_new = t_prev + recharge_t
                # Horizontal line => distance does not change
                ax.plot([t_prev, t_new], [d_prev, d_prev],
                        color=mode_colors_local['recharging'], linewidth=2)
                t_prev = t_new  # Advance time by recharge_t
            
            # 2) If the node changed or we are "moving", plot a slope in distance-time.
            #    If the node is the same but mode changed => that's a "switch" segment.
            if old_state.node == new_state.node and old_state.mode != new_state.mode:
                # No distance gained, but time passes => "switching" segment
                t_new = t_prev + travel_t
                ax.plot([t_prev, t_new], [d_prev, d_prev],
                        color=mode_colors_local['switching'], linewidth=2)
                t_prev = t_new
                # distance remains the same
            else:
                # Actually traveling in some mode
                t_new = t_prev + travel_t
                d_new = d_prev + dist_travel
                color = mode_colors_local.get(old_state.mode, 'black')
                
                ax.plot([t_prev, t_new], [d_prev, d_new],
                        color=color, linewidth=2)
                
                t_prev = t_new
                d_prev = d_new  # distance advanced
                
            final_distance = d_prev
        
        # Optionally label the end of each path:
        ax.text(final_time, final_distance, f"PF{index}",
                fontsize=8, ha='left', va='bottom')
    
    # Plot each path from the Pareto front
    for idx, path_obj in enumerate(pareto_front):
        plot_single_path_distance_time(path_obj, index=idx)
    
    # Cosmetic finishing touches
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Distance (m)", fontsize=11)
    ax.set_title("Pareto-Front Paths: Distance vs. Time", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Build a small color legend
    legend_labels = {
        'drive': "Drive",
        'fly': "Fly",
        'swim': "Swim",
        'roll': "Roll",
        'switching': "Mode Switch",
        'recharging': "Recharging"
    }
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=mode_colors_local[m], lw=3, label=legend_labels[m])
        for m in ['drive','fly','swim','roll','switching','recharging']
    ]
    ax.legend(handles=legend_handles, loc='best', fontsize=9)
    
    return ax




###############################################################################
# Visualization of a single scenario for single parameter variation
###############################################################################
selected_keys = list(all_scenarios.keys())[4:5]

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

        visualize_world_with_multiline_3D(G_world, L, optimal_path, constants, label_option="all_edges")
        plot_basic_metrics(meta_paths, pareto_front, optimal_path)
        plot_stacked_bars(meta_paths)
        visualize_param_variations(all_results, selected_scenario)
        plot_pareto_front_distance_vs_time(pareto_front, L, constants)
        visualize_pareto_fronts(all_results, selected_scenario)
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