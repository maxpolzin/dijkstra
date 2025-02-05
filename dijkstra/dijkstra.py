#%%

%reload_ext autoreload
%autoreload 2

%matplotlib widget


from dijkstra_scenario import build_world_graph
from dijkstra_visualize import visualize_world_with_multiline_3D
from dijkstra_algorithm import layered_dijkstra_with_battery



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


G_world=build_world_graph(id=0)
visualize_world_with_multiline_3D(G_world)


L, best_time, best_energy, best_path, recharge_nodes, switch_nodes = layered_dijkstra_with_battery(
    G_world, start, goal, MODES, CONSTANTS, energy_vs_time=0.0)


print("=== LAYERED DIJKSTRA WITH BATTERY ===")
print(f"Best time: {best_time:.1f}s")
print(f"Total used energy: {best_energy:.3f} Wh")
print("Path:", best_path)
print("Switch nodes (IDs):", switch_nodes)
print("Recharge events (node, mode):")
for node_mode in recharge_nodes:
    print(f" - Recharged at node {node_mode[0]} in mode '{node_mode[1]}'")

visualize_world_with_multiline_3D(G_world, best_path, switch_nodes, recharge_nodes, L, CONSTANTS)




# %%

from collections import defaultdict

def to_string(path):
    str = ""
    for i in range(len(path)):
        node, mode = path[i]
        if i != 0:
            prev_node, prev_mode = path[i-1]
            if prev_node != node:
                str += f" -> {node}"
        else:
            str += f"{node}"
        str += f"{mode[0].upper()}"
    return str


def find_all_feasible_paths(G_world, L, start, goal):

    speedup = True
    analysed_paths = 0
    feasible_paths = []

    if speedup: # extract subgraphs for all simple paths from the world graph
        simple_paths_in_world = list(nx.all_simple_paths(G_world, source=start[0], target=goal[0]))
        print(f"Found {len(simple_paths_in_world)} simple paths in the world graph.")
        
        subgraphs = []
        for path in simple_paths_in_world:
            print(f"Simple path: {path}")
            path_node_set = set(path)
            selected_nodes = [node for node in L.nodes if node[0] in path_node_set]
            subgraph = L.subgraph(selected_nodes).copy()
            subgraphs.append(subgraph)

        print(f"Created {len(subgraphs)} subgraphs from the layered graph.")

        for subgraph in subgraphs:
            for path in nx.all_simple_paths(subgraph, source=start, target=goal):
                analysed_paths += 1

                node_visit_counts = defaultdict(int)
                is_valid = True

                for node, mode in path:
                    node_visit_counts[node] += 1
                    if node_visit_counts[node] > 2:
                        is_valid = False
                        break

                if is_valid:
                    if not path in feasible_paths:
                        feasible_paths.append(path)

    else: # analyse all paths
        for path in nx.all_simple_paths(L, source=start, target=goal):
            analysed_paths += 1

            node_visit_counts = defaultdict(int)
            is_valid = True

            last_node = start[0]
            for node, mode in path:
                node_visit_counts[node] += 1
                if node_visit_counts[node] == 2 and not last_node == node:
                    is_valid = False
                    break
                else:
                    last_node = node
                if node_visit_counts[node] > 2:
                    is_valid = False
                    break

            if is_valid:
                feasible_paths.append(path)

    print(f"Analysed {analysed_paths} paths and found {len(feasible_paths)} feasible paths.")

    return feasible_paths


paths = find_all_feasible_paths(G_world, L, start, goal)


# %%

import matplotlib.pyplot as plt
import numpy as np

def extract_time_energy(L, path):
    total_time = 0.0
    total_energy = 0.0

    for i in range(len(path) - 1):
        (u_node, u_mode) = path[i]
        (v_node, v_mode) = path[i + 1]

        if L.has_edge((u_node, u_mode), (v_node, v_mode)):
            edge_time = L[(u_node, u_mode)][(v_node, v_mode)]['time']
            edge_energy = L[(u_node, u_mode)][(v_node, v_mode)].get('energy_Wh', 0.0)
            total_time += edge_time
            total_energy += edge_energy

    no_recharges = (total_energy // CONSTANTS['BATTERY_CAPACITY'])
    total_time +=  no_recharges * CONSTANTS['RECHARGE_TIME']

    return total_time, total_energy, no_recharges



def extract_mode_times_and_energies(L, path):
    # Initialize mode times
    mode_times = {}
    mode_energies = {}

    total_energy = 0.0

    for i in range(len(path) - 1):
        (u_node, u_mode) = path[i]
        (v_node, v_mode) = path[i + 1]

        if L.has_edge((u_node, u_mode), (v_node, v_mode)):
            edge_data = L[(u_node, u_mode)][(v_node, v_mode)]
            edge_time = edge_data['time']
            edge_energy = edge_data.get('energy_Wh', 0.0)

            if u_mode == v_mode and u_mode in mode_times:
                mode_times[u_mode] += edge_time
            elif u_mode == v_mode:
                mode_times[u_mode] = edge_time
            elif u_mode != v_mode and 'switching' in mode_times:
                mode_times['switching'] += edge_time
            elif u_mode != v_mode:
                mode_times['switching'] = edge_time
            else:
                print(f"Edge from {u_node} to {v_node} has an unknown mode combination: {u_mode} -> {v_mode}")

            if u_mode == v_mode and u_mode in mode_energies:
                mode_energies[u_mode] += edge_energy
            elif u_mode == v_mode:
                mode_energies[u_mode] = edge_energy
            elif u_mode != v_mode and 'switching' in mode_energies:
                mode_energies['switching'] += edge_energy
            elif u_mode != v_mode:
                mode_energies['switching'] = edge_energy
            else:
                print(f"Edge from {u_node} to {v_node} has an unknown mode combination: {u_mode} -> {v_mode}")

            total_energy += edge_energy

    # Calculate the number of recharges
    no_recharges = int(total_energy // CONSTANTS['BATTERY_CAPACITY'])
    # Add charging time
    mode_times['charging'] = no_recharges * CONSTANTS['RECHARGE_TIME']

    return mode_times, mode_energies



times = []
energies = []
recharges = []
mode_time_list = []  # List to store mode times for each path
mode_energy_list = []

for idx, path in enumerate(paths, start=1):
    total_time, total_energy, no_recharges = extract_time_energy(L, path)
    mode_times, mode_energies = extract_mode_times_and_energies(L, path)

    times.append(total_time)
    energies.append(total_energy)
    recharges.append(no_recharges)

    mode_time_list.append(mode_times)
    mode_energy_list.append(mode_energies)

    print(f"Path {idx}:")
    print(to_string(path))
    print(f"Total Time: {total_time:.2f}s and Energy: {total_energy:.3f} Wh with {no_recharges} recharges")
    print(f"Mode Times: {mode_times}\n")
    print(f"Mode Energies: {mode_energies}\n")

print(f"Total number of paths: {len(paths)}")



# Create histograms
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(times, bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram of Travel Times")
plt.xlabel("Time [s]")
plt.ylabel("Count of Solutions")

plt.subplot(1,2,2)
plt.hist(energies, bins=20, color='salmon', edgecolor='black')
plt.title("Histogram of Energy Consumption")
plt.xlabel("Used Energy [Wh]")
plt.ylabel("Count of Solutions")

plt.tight_layout()
plt.show()

# Scatter plot
plt.figure(figsize=(6, 5))
plt.scatter(times, energies, alpha=0.7, color='blue', edgecolors='black')
plt.xlabel("Travel Time [s]")
plt.ylabel("Energy Consumption [Wh]")
plt.title("Path Time vs. Energy Consumption")
plt.grid(True)
plt.show()


# %%


# Plotting Mode Times as Stacked Bar Chart
# Define the modes in the order you want them to appear in the stack
modes = ['fly', 'drive', 'roll', 'swim', 'charging', 'switching']
colors = {
    'fly': 'skyblue',
    'drive': 'lightgreen',
    'roll': 'orange',
    'swim': 'purple',
    'charging': 'salmon',
    'switching': 'grey'
}



combined_data = list(zip(times, mode_time_list))
sorted_data = sorted(combined_data, key=lambda x: x[0])
sorted_times, sorted_mode_time_list = zip(*sorted_data)
path_indices = np.arange(1, len(sorted_mode_time_list) + 1)




combined_data = list(zip(times, energies, mode_time_list, mode_energy_list))
path_indices = np.arange(1, len(sorted_mode_time_list) + 1)

sort_by_total_time = 0
sort_by_total_energy = 1

## Sorted by time
sorted_data = sorted(combined_data, key=lambda x: x[sort_by_total_time])
sorted_times, sorted_energies, sorted_mode_time_list, sorted_mode_energy_list = zip(*sorted_data)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

bottom_energy = np.zeros(len(sorted_mode_energy_list))

for mode in modes:
    mode_values_energy = [mode_energy.get(mode, 0) for mode_energy in sorted_mode_energy_list]
    axes[0].bar(path_indices, mode_values_energy, bottom=bottom_energy, label=mode.capitalize(), color=colors.get(mode, 'black'))
    bottom_energy += mode_values_energy  # Update the bottom for the next mode

axes[0].set_ylabel("Energy Consumed [Wh]", fontsize=14)
axes[0].set_title("Energy Consumed in Each Mode per Path (Sorted by Travel Time)", fontsize=16)
axes[0].legend(title="Modes", fontsize=8, title_fontsize=8)
axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)
axes[0].set_xlabel("Path Number (Sorted by Travel Time)", fontsize=14)

axes[0].set_xticks(path_indices[::10])
axes[0].set_xticklabels(path_indices[::10], rotation=45, ha='right', fontsize=10)

bottom_time = np.zeros(len(sorted_mode_time_list))

for mode in modes:
    mode_values_time = [mode_time.get(mode, 0) for mode_time in sorted_mode_time_list]
    axes[1].bar(path_indices, mode_values_time, bottom=bottom_time, label=mode.capitalize(), color=colors.get(mode, 'black'))
    bottom_time += mode_values_time  # Update the bottom for the next mode

axes[1].set_xlabel("Path Number (Sorted by Travel Time)", fontsize=14)
axes[1].set_ylabel("Time Spent [s]", fontsize=14)
axes[1].set_title("Time Spent in Each Mode per Path (Sorted by Travel Time)", fontsize=16)
axes[1].legend(title="Modes", fontsize=8, title_fontsize=8)
axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)

axes[1].set_xticks(path_indices[::10])
axes[1].set_xticklabels(path_indices[::10], rotation=45, ha='right', fontsize=10)

plt.tight_layout()
plt.show()


## Sorted by energy
sorted_data = sorted(combined_data, key=lambda x: x[sort_by_total_energy])
sorted_times, sorted_energies, sorted_mode_time_list, sorted_mode_energy_list = zip(*sorted_data)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

bottom_energy = np.zeros(len(sorted_mode_energy_list))

for mode in modes:
    mode_values_energy = [mode_energy.get(mode, 0) for mode_energy in sorted_mode_energy_list]
    axes[0].bar(path_indices, mode_values_energy, bottom=bottom_energy, label=mode.capitalize(), color=colors.get(mode, 'black'))
    bottom_energy += mode_values_energy  # Update the bottom for the next mode

axes[0].set_ylabel("Energy Consumed [Wh]", fontsize=14)
axes[0].set_title("Energy Consumed in Each Mode per Path (Sorted by Total Energy)", fontsize=16)
axes[0].legend(title="Modes", fontsize=8, title_fontsize=8)
axes[0].grid(True, axis='y', linestyle='--', alpha=0.7)
axes[0].set_xlabel("Path Number (Sorted by Total Energy)", fontsize=14)

axes[0].set_xticks(path_indices[::10])
axes[0].set_xticklabels(path_indices[::10], rotation=45, ha='right', fontsize=10)

bottom_time = np.zeros(len(sorted_mode_time_list))

for mode in modes:
    mode_values_time = [mode_time.get(mode, 0) for mode_time in sorted_mode_time_list]
    axes[1].bar(path_indices, mode_values_time, bottom=bottom_time, label=mode.capitalize(), color=colors.get(mode, 'black'))
    bottom_time += mode_values_time  # Update the bottom for the next mode

axes[1].set_xlabel("Path Number (Sorted by Total Energy)", fontsize=14)
axes[1].set_ylabel("Time Spent [s]", fontsize=14)
axes[1].set_title("Time Spent in Each Mode per Path (Sorted by Total Energy)", fontsize=16)
axes[1].legend(title="Modes", fontsize=8, title_fontsize=8)
axes[1].grid(True, axis='y', linestyle='--', alpha=0.7)

axes[1].set_xticks(path_indices[::10])
axes[1].set_xticklabels(path_indices[::10], rotation=45, ha='right', fontsize=10)

plt.tight_layout()
plt.show()

# %%