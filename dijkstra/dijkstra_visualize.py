#!/usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def short_mode_name(mode):
    return {
        'drive': 'D',
        'fly': 'F',
        'swim': 'S',
        'roll': 'R'
    }.get(mode, '?')


mode_colors = {
    'fly': 'red',
    'drive': 'lightgreen',
    'roll': 'yellow',
    'swim': 'blue',
    'recharging': 'black',
    'switching': 'lightgrey'
}


def build_edge_labels_for_world(G_world, L):
    edge_labels = {}
    for (u, v) in G_world.edges():
        edge_distance = G_world[u][v]['distance']
        edge_terrain = G_world[u][v]['terrain']
        forward_options = []
        backward_options = []
        
        if L is not None:
            for (node, mode) in L.nodes():
                
                # Look for forward options: from u to v.
                if node == u and L.has_edge((u, mode), (v, mode)):
                    travel_time = L[(u, mode)][(v, mode)]['time']
                    energy_cost = L[(u, mode)][(v, mode)]['energy_Wh']
                    forward_options.append(f"{short_mode_name(mode)}({travel_time:.0f}s,{energy_cost:.1f}Wh)")
                
                # Look for backward options: from v to u.
                if node == v and L.has_edge((v, mode), (u, mode)):
                    travel_time = L[(v, mode)][(u, mode)]['time']
                    energy_cost = L[(v, mode)][(u, mode)]['energy_Wh']
                    backward_options.append(f"{short_mode_name(mode)}({travel_time:.0f}s,{energy_cost:.1f}Wh)")
        
        label_forward = f"{', '.join(forward_options)}" if forward_options else ""
        label_middle = f"{edge_distance:.0f}m ({edge_terrain})"
        label_backward = f"{', '.join(backward_options)}" if backward_options else ""
        label = "\n".join(line for line in [label_forward, label_middle, label_backward] if line)
        edge_labels[(u, v)] = label
    
    return edge_labels

def build_edge_labels_for_path(G_world, L, path_states):
    if path_states is None:
        return {}

    edge_labels = {}
    # Iterate over consecutive states in the traveled path.
    for i in range(len(path_states) - 1):
        (u, mode_u) = path_states[i]
        (v, mode_v) = path_states[i+1]
        # Only consider edges where both node and mode remain consistent.
        if u == v or mode_u != mode_v:
            continue
        # Get world edge parameters.
        edge_distance = G_world[u][v]['distance']
        edge_terrain = G_world[u][v]['terrain']
        # Retrieve the travel parameters for the actual mode used.
        if L is not None and L.has_edge((u, mode_u), (v, mode_u)):
            travel_time = L[(u, mode_u)][(v, mode_u)]['time']
            energy_cost = L[(u, mode_u)][(v, mode_u)]['energy_Wh']
            option_label = f"{short_mode_name(mode_u)}({travel_time:.0f}s, {energy_cost:.1f}Wh)"
        else:
            option_label = ""
        # Build the label: first line shows distance and terrain, second line (if available) the travel option.
        label = f"{edge_distance:.0f}m ({edge_terrain})"
        if option_label:
            label += "\n" + option_label
        edge_labels[(u, v)] = label
    return edge_labels


def get_recharge_status(path_states, recharge_set, switch_nodes):
    if path_states is None:
        return {}
    status_dict = {node: set() for node, _ in path_states}
    assigned_recharges = set()
    for i in range(len(path_states) - 1):
        current_node, current_mode = path_states[i]
        next_node, next_mode = path_states[i+1]
        if current_node in switch_nodes:
            if (current_node, current_mode) in recharge_set and (current_node, current_mode) not in assigned_recharges:
                status_dict[current_node].add('before')
                assigned_recharges.add((current_node, current_mode))
            if (current_node, next_mode) in recharge_set and (current_node, next_mode) not in assigned_recharges:
                status_dict[current_node].add('after')
                assigned_recharges.add((current_node, next_mode))
    for (node, mode) in recharge_set:
        if node not in switch_nodes and (node, mode) not in assigned_recharges:
            status_dict[node].add('yes')
    final_status = {}
    for node, statuses in status_dict.items():
        if node in switch_nodes:
            if 'before' in statuses and 'after' in statuses:
                final_status[node] = 'both'
            elif 'before' in statuses:
                final_status[node] = 'before'
            elif 'after' in statuses:
                final_status[node] = 'after'
            else:
                final_status[node] = 'no'
        else:
            final_status[node] = 'yes' if 'yes' in statuses else 'no'
    return final_status

def build_node_labels(G_world, path_states, switch_nodes, recharge_nodes):
    recharge_status = get_recharge_status(path_states, recharge_nodes, switch_nodes)
    node_labels = {}
    for n in G_world.nodes():
        height_val = G_world.nodes[n]['height']
        recharge_str = recharge_status.get(n, '')
        if recharge_str:
            node_labels[n] = f"{n}, {height_val}m\n({recharge_str})"
        else:
            node_labels[n] = f"{n}, {height_val}m"
    return node_labels

def layered_path_to_mode_edges(path):
    if path is None:
        return []
    edges_modes = []
    for i in range(len(path)-1):
        (u_node, u_mode) = path[i]
        (v_node, v_mode) = path[i+1]
        if u_node != v_node and u_mode == v_mode:
            edges_modes.append(((u_node, v_node), u_mode))
    return edges_modes



def visualize_world_with_multiline_3D(
    G_world, L=None, path_result=None, constants=None,
    title="World Graph with Costs (3D)", label_option="traveled_only"
):
    if path_result is not None:
        path_states = path_result.path
        switch_nodes = path_result.switch_nodes
        recharge_nodes = path_result.recharge_events
    else:
        path_states = None
        switch_nodes = set()
        recharge_nodes = set()
    
    if label_option == "all_edges":
        edge_labels = build_edge_labels_for_world(G_world, L)
    elif label_option == "traveled_only" and path_states is not None:
        edge_labels = build_edge_labels_for_path(G_world, L, path_states)
    else:
        edge_labels = {}
    
    node_labels = build_node_labels(G_world, path_states, switch_nodes, recharge_nodes)
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)
    
    # Plot nodes.
    for n in G_world.nodes():
        x = G_world.nodes[n].get('x', 0.0)
        y = G_world.nodes[n].get('y', 0.0)
        z = G_world.nodes[n].get('height', 0.0)
        color = 'lightgreen' if z == 100 else 'darkgreen'
        ax.scatter(x, y, z, color=color, s=60, depthshade=True, alpha=0.3)
        ax.text(x, y, z, node_labels.get(n, str(n)), fontsize=8, zorder=1)
    
    # Plot all edges in gray.
    for (u, v) in G_world.edges():
        x_u = G_world.nodes[u].get('x', 0.0)
        y_u = G_world.nodes[u].get('y', 0.0)
        z_u = G_world.nodes[u].get('height', 0.0)
        x_v = G_world.nodes[v].get('x', 0.0)
        y_v = G_world.nodes[v].get('y', 0.0)
        z_v = G_world.nodes[v].get('height', 0.0)
        ax.plot([x_u, x_v], [y_u, y_v], [z_u, z_v], color='gray', alpha=0.5)
    
    # Highlight traveled edges by mode if a path was provided.
    if path_states is not None:
        traveled_edges_info = layered_path_to_mode_edges(path_states)
        edges_by_mode = {'drive': [], 'swim': [], 'roll': [], 'fly': []}
        for ((u, v), mode) in traveled_edges_info:
            edges_by_mode[mode].append((u, v))
        color_map = {'drive': 'lightgreen', 'swim': 'blue', 'roll': 'yellow', 'fly': 'red'}
        for mode, edge_list in edges_by_mode.items():
            for (u, v) in edge_list:
                x_u = G_world.nodes[u].get('x', 0.0)
                y_u = G_world.nodes[u].get('y', 0.0)
                z_u = G_world.nodes[u].get('height', 0.0)
                x_v = G_world.nodes[v].get('x', 0.0)
                y_v = G_world.nodes[v].get('y', 0.0)
                z_v = G_world.nodes[v].get('height', 0.0)
                ax.plot([x_u, x_v], [y_u, y_v], [z_u, z_v], color=color_map.get(mode, 'black'), linewidth=2.5)
    
    # Place edge labels at midpoints.
    for (u, v), lbl in edge_labels.items():
        x_u = G_world.nodes[u].get('x', 0.0)
        y_u = G_world.nodes[u].get('y', 0.0)
        z_u = G_world.nodes[u].get('height', 0.0)
        x_v = G_world.nodes[v].get('x', 0.0)
        y_v = G_world.nodes[v].get('y', 0.0)
        z_v = G_world.nodes[v].get('height', 0.0)
        mx = 0.5*(x_u + x_v)
        my = 0.5*(y_u + y_v)
        mz = 0.5*(z_u + z_v)

        if label_option == "all_edges":
            ax.text(mx, my, mz, lbl, fontsize=5, zorder=1,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        elif label_option == "traveled_only":
            ax.text(mx, my, mz, lbl, fontsize=7, zorder=1,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))


    # 2D Legend.
    if constants is not None:
        legend_text = (
            "Nodes:\n<ID>, <height>\n (recharge=?)\n\n"
            "Modes:\n  D: Driving (green)\n  R: Rolling (yellow)\n  F: Flying (red)\n  S: Swimming (blue)\n\n"
            f"Mode switch: ({constants['SWITCH_TIME']:.0f}s, {constants['SWITCH_ENERGY']:.1f}Wh)\n"
            f"Battery: ({constants['RECHARGE_TIME']:.0f}s, {constants['BATTERY_CAPACITY']:.0f}Wh)\n"
        )
    else:
        legend_text = "Nodes:\n<ID>, <height>"
    
    ax.text2D(0.0, 0.0, legend_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
              fontsize=8, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    plt.show(block=False)


def visualize_world_and_graph(dem, terrain, G):
    """
    Visualize a 3D world from:
      - dem: 2D numpy array of shape (size,size) for elevation z
      - terrain: same shape, e.g. 'grass','water','cliff','slope' etc.
      - G: a NetworkX graph with node attributes (x, y, height)

    The function:
      1. Creates a 3D surface plot of the DEM,
      2. Colors the surface according to the terrain array,
      3. Overlays the nodes & edges from graph G (in red).
    """

    size = dem.shape[0]
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

    # Convert terrain labels to colors
    # This is just an example; adapt to your terrain types:
    terrain_colors = np.empty(dem.shape, dtype=object)
    terrain_colors[terrain == 'water'] = 'blue'
    terrain_colors[terrain == 'grass'] = 'green'
    # If you have other terrain types like 'cliff' or 'slope', color them:
    terrain_colors[(terrain != 'water') & (terrain != 'grass')] = 'gray'

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the DEM surface
    surf = ax.plot_surface(
        X, Y, dem,
        facecolors=terrain_colors,
        linewidth=0.5,
        edgecolor='gray',
        antialiased=False,
        shade=False,
        alpha=0.6
    )

    # Build a small legend for water vs grass
    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', label='Water'),
        Patch(facecolor='green', edgecolor='green', label='Grass'),
        Patch(facecolor='gray', edgecolor='gray', label='Other')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Overlay the graph G
    # We assume each node has attributes 'x','y','height'
    # which should align with the DEM scale. 
    # If your DEM is NxN => coords in [0,N), 
    # ensure the graph coords match that range (or are scaled).
    for node in G.nodes():
        gx = G.nodes[node]['x']
        gy = G.nodes[node]['y']
        gz = G.nodes[node]['height']
        ax.scatter(gx, gy, gz, color='red', s=40, zorder=10)

    for (u, v) in G.edges():
        x_u = G.nodes[u]['x']
        y_u = G.nodes[u]['y']
        z_u = G.nodes[u]['height']
        x_v = G.nodes[v]['x']
        y_v = G.nodes[v]['y']
        z_v = G.nodes[v]['height']
        ax.plot([x_u, x_v], [y_u, y_v], [z_u, z_v], color='red', linewidth=1.5, zorder=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Elevation')
    ax.set_title('World DEM + Scenario Graph')
    ax.view_init(elev=15, azim=-110)  # optional viewpoint angle

    plt.show()




def visualize_pareto_fronts(all_results, selected_scenario):
    rows = [
        ("battery_capacity", "Battery Capacity Variation", ["battery_capacity"]),
        ("drive", "Drive Power and Speed Variation", ["drive", "drive_power", "drive_speed"]),
        ("fly", "Fly Power and Speed Variation", ["fly", "fly_power", "fly_speed"]),
        ("swim", "Swim Power and Speed Variation", ["swim", "swim_power", "swim_speed"]),
        ("roll", "Roll Power and Speed Variation", ["roll", "roll_power", "roll_speed"]),
        ("switch", "Switch Energy and Time Variation", ["switch", "switch_time", "switch_energy"])
    ]
    category_color_map = {
        "battery_capacity": "lightgrey",
        "drive": mode_colors["drive"],
        "fly": mode_colors["fly"],
        "swim": mode_colors["swim"],
        "roll": mode_colors["roll"],
        "switch": mode_colors["switching"]
    }
    if selected_scenario not in all_results[0]["results"]:
        print(f"Baseline for scenario {selected_scenario} not found.")
        return
    baseline_data = all_results[0]["results"][selected_scenario]
    baseline_pareto = baseline_data.get("pareto_front", [])
    baseline_points = [(mp.total_time, mp.total_energy) for mp in baseline_pareto if mp]
    groups = {row_key: [] for row_key, _, _ in rows}
    for var_id, var_data in all_results.items():
        if var_id == 0:
            continue
        results = var_data.get("results", {})
        if selected_scenario not in results:
            continue
        variant_pareto = results[selected_scenario].get("pareto_front", [])
        points = [(mp.total_time, mp.total_energy) for mp in variant_pareto if mp]
        if not points:
            continue
        variant_constants = var_data.get("constants", {})
        diff = None
        for row_key, _, rel_keys in rows:
            if row_key == "battery_capacity":
                b_val = all_results[0]["constants"].get("BATTERY_CAPACITY")
                v_val = variant_constants.get("BATTERY_CAPACITY")
                try:
                    diff = (v_val - b_val) / b_val
                except Exception:
                    diff = 0.0
            elif row_key in ["drive", "fly", "swim", "roll"]:
                mode = row_key
                b_power = all_results[0]["constants"]["MODES"][mode]["power"]
                v_power = variant_constants["MODES"][mode]["power"]
                b_speed = all_results[0]["constants"]["MODES"][mode]["speed"]
                v_speed = variant_constants["MODES"][mode]["speed"]
                diff = (v_power - b_power) / b_power if abs((v_power - b_power) / b_power) >= abs((v_speed - b_speed) / b_speed) else (v_speed - b_speed) / b_speed
            elif row_key == "switch":
                b_time = all_results[0]["constants"].get("SWITCH_TIME")
                v_time = variant_constants.get("SWITCH_TIME")
                b_energy = all_results[0]["constants"].get("SWITCH_ENERGY")
                v_energy = variant_constants.get("SWITCH_ENERGY")
                diff = (v_time - b_time) / b_time if abs((v_time - b_time) / b_time) >= abs((v_energy - b_energy) / b_energy) else (v_energy - b_energy) / b_energy
            if diff is not None and abs(diff) > 1e-6:
                label = f"{diff:+.0%}"
                groups[row_key].append((label, points))
                break
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=(6, 4 * n_rows), sharex=True, sharey=True)
    if n_rows == 1:
        axes = [axes]
    for i, (row_key, title_text, _) in enumerate(rows):
        ax = axes[i]
        if baseline_points:
            xs, ys = zip(*sorted(baseline_points, key=lambda p: p[0]))
            ax.plot(xs, ys, linestyle="--", color="black")
            ax.scatter(xs, ys, color="black", marker="x", label="baseline")
        if groups.get(row_key):
            for label, pts in groups[row_key]:
                xs, ys = zip(*sorted(pts, key=lambda p: p[0]))
                col = category_color_map.get(row_key, "gray")
                ax.plot(xs, ys, linestyle="--", color=col)
                ax.scatter(xs, ys, color=col, marker="o", label=f"{row_key} {label}")
        ax.set_title(title_text, fontsize=10)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Travel Time [s]")
    axes[0].set_ylabel("Energy [Wh]")
    plt.tight_layout()
    plt.show(block=False)







# =============================================================================
# Basic Metrics Plot: Histograms and Scatter Plot.
# =============================================================================
def plot_time_histogram(times, ax):
    ax.hist(times, bins=20, color='skyblue', edgecolor='black')
    ax.set_title("Histogram of Travel Times")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Count")

def plot_energy_histogram(energies, ax):
    ax.hist(energies, bins=20, color='salmon', edgecolor='black')
    ax.set_title("Histogram of Energy Consumption")
    ax.set_xlabel("Energy [Wh]")
    ax.set_ylabel("Count")

def plot_distance_histogram(distances, ax):
    ax.hist(distances, bins=20, color='lightblue', edgecolor='black')
    ax.set_title("Histogram of Distances")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")

def plot_scatter_paths(times, energies, colors, pareto_mask, ax, mode_colors):
    times = np.array(times)
    energies = np.array(energies)
    colors = np.array(colors)
    
    pareto_mask = np.asarray(pareto_mask, dtype=bool)
    non_pareto_idx = ~pareto_mask
    if non_pareto_idx.sum() > 0:
        ax.scatter(times[non_pareto_idx], energies[non_pareto_idx],
                   color=colors[non_pareto_idx], alpha=0.7, edgecolors='none')
    pareto_idx = pareto_mask
    if pareto_idx.sum() > 0:
        ax.scatter(times[pareto_idx], energies[pareto_idx],
                   color=colors[pareto_idx], alpha=0.7, edgecolors='black', linewidths=1.5)
    
    ax.set_title("Travel Time vs Energy Consumption\n(colored by dominant mode)")
    ax.set_xlabel("Travel Time [s]")
    ax.set_ylabel("Energy Consumption [Wh]")
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color, markersize=8, label=mode.capitalize())
        for mode, color in mode_colors.items()
    ]
    ax.legend(handles=legend_elements,
              title="Dominant Mode (by distance)",
              loc="best", ncol=3,
              prop={'size': 8}, title_fontsize=8)


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


def format_constants(constants):
    battery_str = "Battery:({}s,{}Wh)".format(
        int(round(constants["RECHARGE_TIME"])),
        int(round(constants["BATTERY_CAPACITY"]))
    )
    switch_str = "Switch:({}s,{}Wh)".format(
        int(round(constants["SWITCH_TIME"])),
        int(round(constants["SWITCH_ENERGY"]))
    )
    mode_strs = []
    for mode, vals in constants["MODES"].items():
        mode_str = "{}:({}m/s,{}W)".format(
            mode.capitalize(),
            int(round(vals["speed"])),
            int(round(vals["power"]))
        )
        mode_strs.append(mode_str)
    return ", ".join([battery_str, switch_str] + mode_strs)



def plot_basic_metrics(meta_paths, pareto_front, optimal_path):
    times = [m.total_time for m in meta_paths]
    energies = [m.total_energy for m in meta_paths]
    distances = [sum(m.mode_distances.values()) for m in meta_paths]
    
    scatter_colors = []
    for m in meta_paths:
        if m.mode_distances:
            dominant_mode = max(m.mode_distances, key=m.mode_distances.get)
            scatter_colors.append(mode_colors.get(dominant_mode, 'blue'))
        else:
            scatter_colors.append('blue')
    
    pareto_mask = np.array([m in pareto_front for m in meta_paths])
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 7))
    
    plot_time_histogram(times, axs[0, 0])
    plot_energy_histogram(energies, axs[0, 1])
    plot_distance_histogram(distances, axs[1, 0])
    plot_scatter_paths(times, energies, scatter_colors, pareto_mask, axs[1, 1], mode_colors)
    
    # Plot the optimal path with a black cross marker if provided.
    if optimal_path is not None:
        axs[1, 1].scatter( optimal_path.total_time, optimal_path.total_energy, marker='X', s=100, facecolors='none', edgecolors='black', zorder=10)
    
    plt.tight_layout()
    plt.show(block=False)



def visualize_param_variations(all_results, selected_scenario, n_cols=5):
    variation_keys = sorted([k for k, v in all_results.items() if selected_scenario in v["results"]])
    n_variations = len(variation_keys)
    if n_variations == 0:
        print(f"Scenario {selected_scenario} not found in any variation.")
        return
    n_rows = int(np.ceil(n_variations / n_cols))
    
    all_times = []
    all_energies = []
    for var in variation_keys:
        data = all_results[var]["results"][selected_scenario]
        meta_paths = data["meta_paths"]
        times = [m.total_time for m in meta_paths]
        energies = [m.total_energy for m in meta_paths]
        all_times.extend(times)
        all_energies.extend(energies)

    if not all_times or not all_energies:
        print(f"No valid paths found for scenario '{selected_scenario}' in any variation. Plotting empty plots.")
        global_xlim = (0, 1)
        global_ylim = (0, 1)
    else:
        x_min, x_max = min(all_times), max(all_times)
        y_min, y_max = min(all_energies), max(all_energies)
        x_margin = 0.05 * (x_max - x_min) if x_max > x_min else 1
        y_margin = 0.05 * (y_max - y_min) if y_max > y_min else 1
        global_xlim = (x_min - x_margin, x_max + x_margin)
        global_ylim = (y_min - y_margin, y_max + y_margin)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5), sharex=True, sharey=True)
    axs = np.array(axs).flatten()
    
    # Main title
    fig.suptitle(f"Scenario: {selected_scenario}", fontsize=8)
    
    # Integrate constants info into the legend title.
    baseline_constants = all_results[0]["constants"]
    legend_title = format_constants(baseline_constants) + "\nDominant Mode (by distance)"
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
               markersize=8, label=mode.capitalize())
        for mode, color in mode_colors.items()
    ]
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=len(legend_elements), prop={'size': 8}, title=legend_title, title_fontsize=8)
    
    for idx, var in enumerate(variation_keys):
        ax = axs[idx]
        data = all_results[var]["results"][selected_scenario]
        var_constants = all_results[var]["constants"]
        meta_paths = data["meta_paths"]
        pareto_front = data["pareto_front"]
        optimal_path = data["optimal_path"]
        
        times = [m.total_time for m in meta_paths]
        energies = [m.total_energy for m in meta_paths]
        
        # Determine dominant mode color based on mode_distances.
        colors = []
        for meta in meta_paths:
            if meta.mode_distances:
                dominant_mode = max(meta.mode_distances, key=meta.mode_distances.get)
                colors.append(mode_colors.get(dominant_mode, 'blue'))
            else:
                colors.append('blue')
        
        pareto_mask = np.array([m in pareto_front for m in meta_paths]) if meta_paths else np.array([])

        plot_scatter_paths(times, energies, colors, pareto_mask, ax, mode_colors)
        
        # Plot the optimal path with a red X marker.
        if optimal_path is not None and hasattr(optimal_path, 'total_time') and hasattr(optimal_path, 'total_energy'):
            ax.scatter(optimal_path.total_time, optimal_path.total_energy, marker='X', s=100, 
                       facecolors='none', edgecolors='black', zorder=10)
        
        # Remove any subplot legend.
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        
        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)
        
        variation_description = describe_variation(baseline_constants, var_constants)
        if optimal_path is not None and hasattr(optimal_path, 'total_time') and hasattr(optimal_path, 'total_energy'):
            title = (f"Var {var} ({variation_description}) - Optimal: "
                     f"{optimal_path.total_time:.2f}s, {optimal_path.total_energy:.2f}Wh")
        else:
            title = f"Var {var} ({variation_description}) - No optimal path"              
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_xlabel(ax.get_xlabel(), fontsize=8)
        ax.set_ylabel(ax.get_ylabel(), fontsize=8)
    
    for j in range(idx + 1, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout(rect=[0, 0, 1, 0.87])
    plt.show(block=False)





# =============================================================================
# Stacked Bar Charts: Breakdown per Mode.
# =============================================================================
def plot_stacked_bars(meta_paths, sort_xticks_interval=10):
    # Store total and breakdown info; ensure mode_distances is kept as a dict.
    combined_data = [
        (
            m.total_time,
            m.total_energy,
            m.mode_times,
            m.mode_energies,
            sum(m.mode_distances.values()) if hasattr(m, "mode_distances") else 0,
            m.mode_distances if hasattr(m, "mode_distances") else {}
        )
        for m in meta_paths
    ]
    num_paths = len(meta_paths)
    path_indices = np.arange(1, num_paths + 1)
    
    # Sorted by travel time
    sorted_by_time = sorted(combined_data, key=lambda x: x[0])
    (_, _, sorted_mode_times_time, sorted_mode_energies_time, _, sorted_mode_distances_time) = zip(*sorted_by_time)
    
    # Sorted by total energy
    sorted_by_energy = sorted(combined_data, key=lambda x: x[1])
    (_, _, sorted_mode_times_energy, sorted_mode_energies_energy, _, sorted_mode_distances_energy) = zip(*sorted_by_energy)
    
    # Sorted by total distance (computed from mode_distances)
    sorted_by_distance = sorted(combined_data, key=lambda x: x[4])
    (_, _, sorted_mode_times_distance, sorted_mode_energies_distance, _, sorted_mode_distances_distance) = zip(*sorted_by_distance)
    
    # Create a 3x3 grid: rows correspond to sort method (time, energy, distance);
    # columns: Energy, Time, Distance breakdown.
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
    
    def plot_bar_subplot(ax, sorted_dicts, ylabel, title):
        bottom = np.zeros(num_paths)
        for mode in mode_colors.keys():
            # sorted_dicts is a tuple of dictionaries.
            mode_vals = [d.get(mode, 0) for d in sorted_dicts]
            ax.bar(path_indices, mode_vals, bottom=bottom,
                   label=mode.capitalize(), color=mode_colors.get(mode, 'black'))
            bottom += np.array(mode_vals)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.legend(title="Modes", fontsize=8, title_fontsize=8)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.set_xlabel("Path #", fontsize=8)
        ax.set_xticks(path_indices[::sort_xticks_interval])
        ax.set_xticklabels(path_indices[::sort_xticks_interval], rotation=45, ha='right', fontsize=8)
    
    # Row 0: Sorted by travel time
    plot_bar_subplot(axes[0, 0], sorted_mode_energies_time, "Energy [Wh]", "Energy (Time-sorted)")
    plot_bar_subplot(axes[0, 1], sorted_mode_times_time, "Time [s]", "Time (Time-sorted)")
    plot_bar_subplot(axes[0, 2], sorted_mode_distances_time, "Distance [m]", "Distance (Time-sorted)")
    
    # Row 1: Sorted by total energy
    plot_bar_subplot(axes[1, 0], sorted_mode_energies_energy, "Energy [Wh]", "Energy (Energy-sorted)")
    plot_bar_subplot(axes[1, 1], sorted_mode_times_energy, "Time [s]", "Time (Energy-sorted)")
    plot_bar_subplot(axes[1, 2], sorted_mode_distances_energy, "Distance [m]", "Distance (Energy-sorted)")
    
    # Row 2: Sorted by total distance
    plot_bar_subplot(axes[2, 0], sorted_mode_energies_distance, "Energy [Wh]", "Energy (Distance-sorted)")
    plot_bar_subplot(axes[2, 1], sorted_mode_times_distance, "Time [s]", "Time (Distance-sorted)")
    plot_bar_subplot(axes[2, 2], sorted_mode_distances_distance, "Distance [m]", "Distance (Distance-sorted)")
    
    plt.tight_layout()
    plt.show(block=False)
