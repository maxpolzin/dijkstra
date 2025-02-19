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



# plot pareto fronts for parameter variations
# plot pareto front with loss of modality
# get rid of crosses


# resource and risk variance in a scenario
# each pareto path makes sense for certain environmental conditions
# optimal for given risk/resource conditions






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
all_variations = list(SensitivityConstants(CONSTANTS, variation=0.3)) # 0.2, 0.5


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
   
    all_results_list = Parallel(n_jobs=13)(
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
selected_keys = list(all_scenarios.keys())[:1]

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
        plot_pareto_front_distance_vs_time(pareto_front, L, constants)

        visualize_param_variations(all_results, selected_scenario)

        plt.show()

    else:
        print(f"Scenario {selected_scenario} not found in variation {selected_variation}.")









# %%


def describe_variation(baseline, variant):
    differences = []
    # Compare top-level numeric parameters (excluding MODES)
    for key in baseline:
        if key == "MODES":
            continue
        if isinstance(baseline[key], (int, float)):
            if baseline[key] != variant[key]:
                diff = (variant[key] - baseline[key]) / baseline[key] * 100
                differences.append(f"{key} {'+' if diff >= 0 else ''}{int(round(diff))}%")
    # Compare nested parameters in MODES
    for mode in baseline["MODES"]:
        for param in baseline["MODES"][mode]:
            base_val = baseline["MODES"][mode][param]
            var_val = variant["MODES"][mode][param]
            if base_val != var_val:
                diff = (var_val - base_val) / base_val * 100
                differences.append(f"{mode.capitalize()} {param} {'+' if diff >= 0 else ''}{int(round(diff))}%")
    return ", ".join(differences)


def extract_and_group_min_paths(all_results):
    baseline_constants = all_results[0]["constants"]

    def get_variation_description(variation_id, variant_constants):
        if variation_id == 0:
            return "baseline"
        return describe_variation(baseline_constants, variant_constants)

    def get_min_energy(pareto_front):
        if not pareto_front:
            return (None, None)
        mp = min(pareto_front, key=lambda x: x.total_energy)
        return (mp.total_time, mp.total_energy)

    def get_min_time(pareto_front):
        if not pareto_front:
            return (None, None)
        mp = min(pareto_front, key=lambda x: x.total_time)
        return (mp.total_time, mp.total_energy)

    grouped = {}
    for variation_id, variation_data in all_results.items():
        variant_constants = variation_data.get("constants", {})
        description = get_variation_description(variation_id, variant_constants)
        scenarios = variation_data.get("results", {})
        for scenario_name, scenario_data in scenarios.items():
            pareto_front = scenario_data.get("pareto_front", [])
            entry = {
                "min_energy_path": get_min_energy(pareto_front),
                "min_time_path": get_min_time(pareto_front)
            }
            if scenario_name not in grouped:
                grouped[scenario_name] = {}
            if variation_id == 0:
                grouped[scenario_name]["baseline"] = {"id": 0, **entry}
            else:
                tokens = description.split()
                if len(tokens) >= 2:
                    param = "_".join(token.lower() for token in tokens[:-1])
                    percent = tokens[-1]
                else:
                    param = tokens[0].lower()
                    percent = ""
                if param not in grouped[scenario_name]:
                    grouped[scenario_name][param] = {"ids": []}
                grouped[scenario_name][param]["ids"].append(variation_id)
                grouped[scenario_name][param][percent] = entry
    return grouped

# %matplotlib widget

import matplotlib.pyplot as plt
# %matplotlib widget

grouped = extract_and_group_min_paths(all_results)

def plot_manual_subplots(grouped, xlim=None):
    rows = [
        ("battery_capacity", "Battery Capacity Variation", ["battery_capacity"]),
        ("drive", "Drive Power and Speed Variation", ["drive", "drive_power", "drive_speed"]),
        ("fly", "Fly Power and Speed Variation", ["fly", "fly_power", "fly_speed"]),
        ("swim", "Swim Power and Speed Variation", ["swim", "swim_power", "swim_speed"]),
        ("roll", "Roll Power and Speed Variation", ["roll", "roll_power", "roll_speed"]),
        ("switch", "Switch Energy and Speed Variation", ["switch", "switch_time", "switch_energy"])
    ]
    
    scenarios = list(grouped.keys())
    nrows = len(rows)
    ncols = len(scenarios)
    
    def compute_cross(data):
        t1, e1 = data["min_energy_path"]
        t2, e2 = data["min_time_path"]
        t_min, t_max = min(t1, t2), max(t1, t2)
        e_min, e_max = min(e1, e2), max(e1, e2)
        mean_time = (t_min + t_max) / 2
        mean_energy = (e_min + e_max) / 2
        dt_low = mean_time - t_min
        dt_high = t_max - mean_time
        de_low = mean_energy - e_min
        de_high = e_max - mean_energy
        return mean_time, mean_energy, dt_low, dt_high, de_low, de_high

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows),
                             sharex=True, sharey=True, squeeze=False)
    
    for i, (row_key, row_title, possible_keys) in enumerate(rows):
        for j, scenario in enumerate(scenarios):
            ax = axes[i][j]
            baseline = grouped[scenario].get("baseline")
            if not baseline or "min_energy_path" not in baseline or "min_time_path" not in baseline:
                if i == 0:
                    ax.set_title(scenario)
                ax.text(0.5, 0.5, "No baseline", ha="center", va="center", transform=ax.transAxes)
                continue
            base_mean_time, base_mean_energy, base_dt_low, base_dt_high, base_de_low, base_de_high = compute_cross(baseline)
            ax.errorbar(base_mean_time, base_mean_energy,
                        xerr=[[base_dt_low], [base_dt_high]],
                        yerr=[[base_de_low], [base_de_high]],
                        fmt="x", color="black", capsize=5, label="baseline")
            for key in possible_keys:
                if key in grouped[scenario]:
                    var_group = grouped[scenario][key]
                    for perc, var_data in var_group.items():
                        if perc == "ids":
                            continue
                        if not ("min_energy_path" in var_data and "min_time_path" in var_data):
                            continue
                        mean_time, mean_energy, dt_low, dt_high, de_low, de_high = compute_cross(var_data)
                        if "power" in key or "energy" in key:
                            color = "red"
                        elif "speed" in key or "time" in key:
                            color = "blue"
                        else:
                            color = "green"
                        ax.errorbar(mean_time, mean_energy,
                                    xerr=[[dt_low], [dt_high]],
                                    yerr=[[de_low], [de_high]],
                                    fmt="x", color=color, capsize=5, label=f"{key} {perc}")
            if i == 0:
                ax.set_title(scenario)
            if j == 0:
                ax.set_ylabel(row_title)
            if i == nrows - 1:
                ax.set_xlabel("Time")
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.legend(fontsize="x-small")
    plt.tight_layout()
    plt.show()

plot_manual_subplots(grouped, xlim=(0, 3000))



# %%


