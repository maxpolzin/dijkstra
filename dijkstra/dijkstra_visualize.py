#!/usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
import numpy as np
from matplotlib.patches import Patch


def short_mode_name(mode):
    return {
        'drive': 'D',
        'fly': 'F',
        'swim': 'S',
        'roll': 'R'
    }.get(mode, '?')


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
                    energy_cost = L[(u, mode)][(v, mode)].get('energy_Wh', 0.0)
                    forward_options.append(f"{short_mode_name(mode)}({travel_time:.0f}s,{energy_cost:.1f}Wh)")
                
                # Look for backward options: from v to u.
                if node == v and L.has_edge((v, mode), (u, mode)):
                    travel_time = L[(v, mode)][(u, mode)]['time']
                    energy_cost = L[(v, mode)][(u, mode)].get('energy_Wh', 0.0)
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
            energy_cost = L[(u, mode_u)][(v, mode_u)].get('energy_Wh', 0.0)
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
    """
    Visualize the world graph in 3D with node and edge labels.
    
    Parameters:
      - G_world: The world graph (nodes have x, y, height).
      - L: The layered graph.
      - path_result: A PathResult object.
      - constants: Dictionary of constants.
      - title: Plot title.
      - label_option: "all_edges" to label all traversable edges, "traveled_only" to label only the edges
                      and direction the robot has actually traveled.
    """
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
    
    plt.show()
















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


