import networkx as nx
import matplotlib.pyplot as plt


def short_mode_name(mode):
    # e.g. drive->D, fly->F, etc.
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

        # gather forward
        forward_items = []
        backward_items = []

        if L is not None:
            # We can loop over each mode in L, but typically we'll check L.has_edge((u,mode),(v,mode)).
            # We'll gather all feasible modes from L's node set
            for (node, mode) in L.nodes():
                # node can be 1..8, mode can be 'drive','fly','swim','roll', etc.
                if node==u:
                    # check if there's an edge ((u,mode)->(v,mode)) in L
                    if L.has_edge((u,mode),(v,mode)):
                        t  = L[(u,mode)][(v,mode)]['time']
                        eW = L[(u,mode)][(v,mode)].get('energy_Wh',0.0)
                        short_m = short_mode_name(mode)
                        forward_items.append(f"{short_m}({t:.0f}s,{eW:.1f}Wh)")

                if node==v:
                    # check (v,mode)->(u,mode) for backward
                    if L.has_edge((v,mode),(u,mode)):
                        t  = L[(v,mode)][(u,mode)]['time']
                        eW = L[(v,mode)][(u,mode)].get('energy_Wh',0.0)
                        short_m = short_mode_name(mode)
                        backward_items.append(f"{short_m}({t:.0f}s,{eW:.1f}Wh)")

        # Build multiline label
        # top line: --> forward modes
        # mid line: dist(terrain)
        # bot line: <-- backward modes
        top_line = f"--> {', '.join(forward_items)}" if forward_items else ""
        mid_line = f"{dist}m ({terr})"
        bot_line = f"<-- {', '.join(backward_items)}" if backward_items else ""

        label_str = "\n".join(line for line in [top_line,mid_line,bot_line] if line)

        edge_labels[(u,v)] = label_str


    return edge_labels



def get_recharge_status(path_states, recharge_set, switch_nodes):
    """
    Determines the recharge status for each node in the path.
    
    Parameters:
        path_states (list of tuples): List of (node, mode) representing the path.
        recharge_set (set of tuples): Set of (node, mode) where recharging occurred.
        switch_nodes (set): Set of node IDs where mode switches occur.
        
    Returns:
        status_dict (dict): Dictionary mapping node to recharge status 
                            ('before', 'after', 'both', 'yes', 'no').
    """
    # Initialize a dictionary to track recharge events per node

    if path_states is None:
        return {}

    status_dict = {node: set() for node, _ in path_states}

    # Keep track of recharge events that have been assigned
    assigned_recharges = set()
   
    # Iterate through consecutive pairs in the path to identify recharge events
    for i in range(len(path_states) - 1):
        current_node, current_mode = path_states[i]
        next_node, next_mode = path_states[i + 1]

        if current_node in switch_nodes:
            # Assign 'before' if recharge occurred before switching modes
            if (current_node, current_mode) in recharge_set and (current_node, current_mode) not in assigned_recharges:
                status_dict[current_node].add('before')
                assigned_recharges.add((current_node, current_mode))

            # Assign 'after' if recharge occurred after switching modes
            if (current_node, next_mode) in recharge_set and (current_node, next_mode) not in assigned_recharges:
                status_dict[current_node].add('after')
                assigned_recharges.add((current_node, next_mode))

    # Assign 'yes' for recharges at nodes that are not switch nodes
    for (node, mode) in recharge_set:
        if node not in switch_nodes and (node, mode) not in assigned_recharges:
            status_dict[node].add('yes')

    # Finalize the recharge status labels
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
        (u_node,u_mode)=path[i]
        (v_node,v_mode)=path[i+1]
        if (u_node!=v_node) and (u_mode==v_mode):
            edges_modes.append(((u_node,v_node),u_mode))
    return edges_modes


def visualize_world_with_multiline(
    G_world,
    path_states=None,
    switch_nodes=None,
    recharge_nodes=None,
    L=None,
    constants=None,
    title="World Graph with Costs"
):
    """
    We build edge_labels using build_edge_labels_for_world(...),
    then plot G_world with those multiline labels.
    highlight_edges => color them in red (or by mode if you want).
    switch_nodes => color in lightblue
    recharge_nodes => color in orange
    """

    edges_modes = layered_path_to_mode_edges(path_states)

    # Build a dictionary of edges per mode so we can color them
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

    # We gather the multiline labels from the layered graph
    edge_labels = build_edge_labels_for_world(G_world, L)
    node_labels = build_node_labels(G_world, path_states, switch_nodes, recharge_nodes)


    pos = nx.spring_layout(G_world, seed=42)
    plt.figure(figsize=(8,8))
    plt.title(title)

    nx.draw_networkx_nodes(G_world, pos,
                           nodelist=G_world.nodes(),
                           node_color='lightgreen',
                           node_size=600)

    # Draw the node labels (multiline is recognized by networkx if we use "\n")
    nx.draw_networkx_labels(
        G_world, pos,
        labels=node_labels,
        font_size=8,
        font_color='black'
    )



    # Edges in gray by default
    nx.draw_networkx_edges(G_world, pos, edge_color='gray')

    # color traveled edges by mode
    color_map={'fly':'red','roll':'yellow','drive':'lightgreen','swim':'blue'}
    if edges_by_mode is not None:
        for mode, edgelist in edges_by_mode.items():
            c = color_map.get(mode,'black')
            nx.draw_networkx_edges(G_world, pos,
                                edgelist=edgelist,
                                edge_color=c,
                                width=2.5)




    # Now add the multiline label
    nx.draw_networkx_edge_labels(G_world, pos,
        edge_labels=edge_labels,
        rotate=False,
        font_color='black',
        font_size=7,
        label_pos=0.5,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
    )


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


    ax = plt.gca()
    ax.text(
        0.0, 0.0,
        legend_text,
        transform=ax.transAxes,
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=8,
        color='black',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
    )


    plt.axis('off')
    plt.show()


