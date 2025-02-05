import math
import random
import heapq
import networkx as nx

def get_edge_parameters(mode, terrain, h1, h2, dist, power, speed, constants):
    travel_time = dist / speed
    energy_Wh = (power * travel_time) / 3600.0
    is_feasible = False

    if mode == 'fly':
        is_feasible = True
    if terrain == 'water' and mode == 'swim':
        is_feasible = True
    if terrain == 'slope':
        if mode == 'drive':
            is_feasible = True
        elif mode == 'roll':
            is_feasible = (h1 == 100 and h2 == 0)
    if terrain == 'grass' and mode == 'drive':
        is_feasible = True

    is_feasible = is_feasible and not exceeds_battery_capacity(energy_Wh, constants['BATTERY_CAPACITY'])
    
    if is_feasible:
        return (travel_time, energy_Wh)

    return None


def exceeds_battery_capacity(energy_wh, battery_capacity):
    return energy_wh > battery_capacity


def build_layered_graph(G_world, modes, constants):
    L = nx.DiGraph()
    modes_list = list(modes.keys())
    
    for v in G_world.nodes():
        for m in modes_list:
            L.add_node((v, m), height=G_world.nodes[v]['height'])
    
    for (u, v) in G_world.edges():
        dist = G_world[u][v]['distance']
        terr = G_world[u][v]['terrain']
        height_u = G_world.nodes[u]['height']
        height_v = G_world.nodes[v]['height']
        
        for mode in modes_list:
            forward_edge = get_edge_parameters(mode, terr, height_u, height_v, dist, modes[mode]['power'], modes[mode]['speed'], constants)
            if forward_edge is not None: 
                L.add_edge((u, mode), (v, mode), time=forward_edge[0], energy_Wh=forward_edge[1], terrain=terr, distance=dist)
            backward_edge = get_edge_parameters(mode, terr, height_v, height_u, dist, modes[mode]['power'], modes[mode]['speed'], constants)
            if backward_edge is not None:
                L.add_edge((v, mode), (u, mode), time=backward_edge[0], energy_Wh=backward_edge[1], terrain=terr, distance=dist)

    for node in G_world.nodes():
        for m1 in modes_list:
            for m2 in modes_list:
                if m1 != m2:
                    switch_energy_wh = constants['SWITCH_ENERGY']
                    switch_time = constants['SWITCH_TIME']
                    if not exceeds_battery_capacity(switch_energy_wh, constants['BATTERY_CAPACITY']):
                        L.add_edge((node, m1), (node, m2),
                                   time=switch_time,
                                   energy_Wh=switch_energy_wh,
                                   terrain='switch',
                                   distance=0)

    return L


def combined_cost(time, energy):
    return (1 - energy_vs_time) * time + energy_vs_time * energy


def layered_dijkstra_with_battery(G_world, start, goal, modes, constants, energy_vs_time=0.0, dbg=False):

    L = build_layered_graph(G_world, modes, constants)

    dist = {}
    pred = {}
    recharged = {}
    best_cost = {}

    def combined_cost(time, energy):
        return (1 - energy_vs_time) * time + energy_vs_time * energy

    # Helper: update the cumulative energy and time given an edge.
    def update_state(cur_energy, cur_time, edge_energy, edge_time):
        if cur_energy + edge_energy <= constants['BATTERY_CAPACITY']:
            return cur_energy + edge_energy, cur_time + edge_time, False
        else:
            recharge_time_adjusted = (cur_energy / constants['BATTERY_CAPACITY']) * constants['RECHARGE_TIME']
            return edge_energy, cur_time + recharge_time_adjusted + edge_time, True

    # Helper: reconstruct path and collect recharge events.
    def reconstruct_path(state):
        path = []
        recharge_set = set()
        c = state
        while c is not None:
            path.append((c[0], c[1]))
            p = pred.get(c)
            if p is not None and recharged.get(c, False):
                recharge_set.add((p[0], p[1]))
            c = p
        path.reverse()
        return path, recharge_set

    # Helper: find nodes where mode-switch occurred.
    def find_mode_switch_nodes(path):
        s = set()
        for i in range(len(path) - 1):
            (u_node, u_mode) = path[i]
            (v_node, v_mode) = path[i + 1]
            if (u_node == v_node) and (u_mode != v_mode):
                s.add(u_node)
        return s

    source = (*start, 0.0, 0.0)  # state: (node, mode, cum_energy, cum_time)
    initial_cost = combined_cost(0.0, 0.0)

    dist[source] = initial_cost
    pred[source] = None
    recharged[source] = False
    best_cost[(start[0], start[1])] = initial_cost

    pq = [(initial_cost, source)]
    if dbg:
        print(f"Initialized pq with {source} cost {initial_cost:.2f}")

    while pq:
        cur_cost, current_state = heapq.heappop(pq)
        cur_node, cur_mode, cur_energy, cur_time = current_state

        if dbg:
            print(f"Popped state: {current_state} cost {cur_cost:.2f}")
        if cur_cost > dist.get(current_state, float('inf')):
            if dbg:
                print(f"Skipping {current_state} cost {cur_cost:.2f}")
            continue

        if (cur_node == goal[0]) and (cur_mode == goal[1]):
            if dbg:
                print(f"Reached goal {goal} cost {cur_cost:.2f}")
            path, recharge_set = reconstruct_path(current_state)
            sorted_recharge_nodes = [nm for nm in path if nm in recharge_set]
            switch_nodes = find_mode_switch_nodes(path)
            return (L, cur_time, cur_energy, path, sorted_recharge_nodes, switch_nodes)

        for nbr in L.successors((cur_node, cur_mode)):
            edge_data = L[(cur_node, cur_mode)][nbr]
            edge_time = edge_data['time']
            edge_energy = edge_data.get('energy_Wh', 0.0)
            nbr_node, nbr_mode = nbr

            if dbg:
                print(f"Exploring {nbr} time {edge_time:.2f} energy {edge_energy:.2f}")

            new_used, new_time, did_recharge = update_state(cur_energy, cur_time, edge_energy, edge_time)
            next_state = (nbr_node, nbr_mode, new_used, new_time)
            new_cost = combined_cost(new_time, new_used)

            if (nbr_node, nbr_mode) in best_cost and new_cost >= best_cost[(nbr_node, nbr_mode)]:
                if dbg:
                    print(f"Skipping {next_state} cost {new_cost:.2f} >= {best_cost[(nbr_node, nbr_mode)]:.2f}")
                continue
            best_cost[(nbr_node, nbr_mode)] = new_cost

            if new_cost < dist.get(next_state, float('inf')):
                dist[next_state] = new_cost
                pred[next_state] = current_state
                recharged[next_state] = did_recharge
                heapq.heappush(pq, (new_cost, next_state))
                if dbg:
                    print(f"Updated {next_state} cost {new_cost:.2f}")
            else:
                if dbg:
                    print(f"Skipping update for {next_state}")

    return (L, float('inf'), float('inf'), [], set(), set())