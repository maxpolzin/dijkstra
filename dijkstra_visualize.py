import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection

def short_mode_name(mode):
    return {
        'drive':'D',
        'fly':'F',
        'swim':'S',
        'roll':'R'
    }.get(mode,'?')


def build_edge_labels_for_world(G_world, L):
    """
    Returns a dict edge_labels for G_world, where each edge (u,v)
    has a multiline label:

      --> M1(...), M2(...)
      dist(terrain)
      <-- M3(...), M4(...)

    forward costs gather from L[(u,mode) -> (v,mode)],
    backward costs from L[(v,mode) -> (u,mode)].
    The time & energy are stored in 'time','energy_Wh'.
    """
    edge_labels = {}

    for (u,v) in G_world.edges():
        dist   = G_world[u][v]['distance']
        terr   = G_world[u][v]['terrain']
        hu     = G_world.nodes[u]['height']
        hv     = G_world.nodes[v]['height']

        forward_items = []
        backward_items = []

        if L is not None:
            for (node, mode) in L.nodes():
                if node == u:
                    if L.has_edge((u,mode), (v,mode)):
                        t  = L[(u,mode)][(v,mode)]['time']
                        eW = L[(u,mode)][(v,mode)].get('energy_Wh', 0.0)
                        short_m = short_mode_name(mode)
                        forward_items.append(f"{short_m}({t:.0f}s,{eW:.1f}Wh)")
                if node == v:
                    if L.has_edge((v,mode), (u,mode)):
                        t  = L[(v,mode)][(u,mode)]['time']
                        eW = L[(v,mode)][(u,mode)].get('energy_Wh', 0.0)
                        short_m = short_mode_name(mode)
                        backward_items.append(f"{short_m}({t:.0f}s,{eW:.1f}Wh)")

        top_line = f"--> {', '.join(forward_items)}" if forward_items else ""
        mid_line = f"{dist:.0f}m\n({terr})" if top_line == "" else f"{dist:.0f}m ({terr})"
        bot_line = f"<-- {', '.join(backward_items)}" if backward_items else ""

        label_str = "\n".join(line for line in [top_line, mid_line, bot_line] if line)
        edge_labels[(u,v)] = label_str

    return edge_labels


def get_recharge_status(path_states, recharge_set, switch_nodes):
    """
    Determines the recharge status for each node in the path.
    """
    if path_states is None:
        return {}

    status_dict = {node: set() for node, _ in path_states}
    assigned_recharges = set()

    for i in range(len(path_states) - 1):
        current_node, current_mode = path_states[i]
        next_node, next_mode = path_states[i + 1]

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
            if 'yes' in statuses:
                final_status[node] = 'yes'
            else:
                final_status[node] = 'no'

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
    """
    Return a list of ((u->v), mode) for actual traveled edges,
    ignoring mode-switch edges.
    """
    if path is None:
        return []

    edges_modes=[]
    for i in range(len(path)-1):
        (u_node,u_mode) = path[i]
        (v_node,v_mode) = path[i+1]
        if (u_node!=v_node) and (u_mode==v_mode):
            edges_modes.append(((u_node,v_node),u_mode))
    return edges_modes



def visualize_world_with_multiline_3D(
    G_world,
    path_states=None,
    switch_nodes=None,
    recharge_nodes=None,
    L=None,
    constants=None,
    title="World Graph with Costs (3D)"
):
    edges_modes = layered_path_to_mode_edges(path_states)
    edges_by_mode = {
        'drive':[],
        'swim':[],
        'roll':[],
        'fly': []
    }
    for ((u,v), mode) in edges_modes:
        edges_by_mode[mode].append((u,v))

    if switch_nodes is None:
        switch_nodes = set()
    if recharge_nodes is None:
        recharge_nodes = set()

    edge_labels = build_edge_labels_for_world(G_world, L)
    node_labels = build_node_labels(G_world, path_states, switch_nodes, recharge_nodes)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    plt.title(title)

    # Plot nodes
    for n in G_world.nodes():
        x = G_world.nodes[n].get('x', 0.0)
        y = G_world.nodes[n].get('y', 0.0)
        z = G_world.nodes[n].get('height', 0.0)
        c = 'lightgreen' if z == 100 else 'darkgreen'
        ax.scatter(x, y, z, color=c, s=60, depthshade=True)
        ax.text(x, y, z, node_labels[n], fontsize=8, zorder=1)

    # Plot edges (gray by default)
    for (u,v) in G_world.edges():
        x_u = G_world.nodes[u].get('x', 0.0)
        y_u = G_world.nodes[u].get('y', 0.0)
        z_u = G_world.nodes[u].get('height', 0.0)
        x_v = G_world.nodes[v].get('x', 0.0)
        y_v = G_world.nodes[v].get('y', 0.0)
        z_v = G_world.nodes[v].get('height', 0.0)
        ax.plot([x_u, x_v], [y_u, y_v], [z_u, z_v], color='gray', alpha=0.5)

    # Plot traveled edges by mode
    color_map={'fly':'red','roll':'yellow','drive':'lightgreen','swim':'blue'}
    for mode, edgelist in edges_by_mode.items():
        c = color_map.get(mode,'black')
        for (u,v) in edgelist:
            x_u = G_world.nodes[u].get('x', 0.0)
            y_u = G_world.nodes[u].get('y', 0.0)
            z_u = G_world.nodes[u].get('height', 0.0)
            x_v = G_world.nodes[v].get('x', 0.0)
            y_v = G_world.nodes[v].get('y', 0.0)
            z_v = G_world.nodes[v].get('height', 0.0)
            ax.plot([x_u, x_v], [y_u, y_v], [z_u, z_v], color=c, linewidth=2.5)

    # Place edge labels at midpoint in 3D
    for (u,v), lbl in edge_labels.items():
        x_u = G_world.nodes[u].get('x', 0.0)
        y_u = G_world.nodes[u].get('y', 0.0)
        z_u = G_world.nodes[u].get('height', 0.0)
        x_v = G_world.nodes[v].get('x', 0.0)
        y_v = G_world.nodes[v].get('y', 0.0)
        z_v = G_world.nodes[v].get('height', 0.0)
        mx = 0.5*(x_u + x_v)
        my = 0.5*(y_u + y_v)
        mz = 0.5*(z_u + z_v)
        ax.text(mx, my, mz, lbl, fontsize=7, zorder=1,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Legend text (in 2D coordinates)
    if constants is not None:
        legend_text = (
            "Nodes:\n"
            "<ID>, <height>\n (recharge=?)\n\n"
            "Modes:\n"
            "  D(riving): green\n  R(olling): yellow\n  F(lying): red\n  S(wimming): blue\n\n"
            f"Mode switch: ({constants['SWITCH_TIME']:.0f}s, {constants['SWITCH_ENERGY']:.1f}Wh)\n"
            f"Battery: ({constants['RECHARGE_TIME']:.0f}s, {constants['BATTERY_CAPACITY']:.0f}Wh)\n"
        )
    else:
        legend_text = "Nodes:\n<ID>, <height>"

    ax.text2D(
        0.0, 0.0,
        legend_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=8,
        color='black',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    plt.show()


# %%