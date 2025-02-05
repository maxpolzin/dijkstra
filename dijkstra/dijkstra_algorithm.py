
import math
import random
import heapq
import networkx as nx



def is_edge_allowed(mode, terrain, h1, h2, dist, power):
    if mode == 'fly':
        return True

    if terrain == 'water' and mode == 'swim':
        return True

    if terrain == 'slope':
        if mode == 'drive':
            return True
        elif mode == 'roll':
            return h1 == 100 and h2 == 0

    if terrain == 'grass' and mode == 'drive':
        return True

    return False


def exceeds_battery_capacity(energy_wh, battery_capacity):
    return energy_wh > battery_capacity


def build_layered_graph(G_world, modes, constants):
    L = nx.DiGraph()
    modes_list = list(modes.keys())

    # 1) Create layered nodes
    for v in G_world.nodes():
        for m in modes_list:
            L.add_node((v, m), height=G_world.nodes[v]['height'])

    # 2) Add travel edges based on mode, terrain, height, distance, and energy constraints
    for (u, v) in G_world.edges():
        
        dist = G_world[u][v]['distance']
        terr = G_world[u][v]['terrain']
        hu = G_world.nodes[u]['height']
        hv = G_world.nodes[v]['height']

        for mode in modes_list:
            speed = modes[mode]['speed']
            power = modes[mode]['power']

            # Forward direction (u -> v)
            if is_edge_allowed(mode, terr, hu, hv, dist, power):
                travel_time = dist / speed  # in seconds
                energy_Wh = (power * travel_time) / 3600.0  # Convert to Wh

                if not exceeds_battery_capacity(energy_Wh, constants['BATTERY_CAPACITY']):
                    L.add_edge(
                        (u, mode),
                        (v, mode),
                        time=travel_time,
                        energy_Wh=energy_Wh,
                        terrain=terr,
                        distance=dist
                    )
                # else:
                #     print(f"Excluded edge {(u, v)} in mode '{mode}' due to high energy requirement: {energy_Wh:.3f} Wh")

            # Backward direction (v -> u)
            if is_edge_allowed(mode, terr, hv, hu, dist, power):
                travel_time = dist / speed
                energy_Wh = (power * travel_time) / 3600.0

                if not exceeds_battery_capacity(energy_Wh, constants['BATTERY_CAPACITY']):
                    L.add_edge(
                        (v, mode),
                        (u, mode),
                        time=travel_time,
                        energy_Wh=energy_Wh,
                        terrain=terr,
                        distance=dist
                    )
                # else:
                #     print(f"Excluded edge {(v, u)} in mode '{mode}' due to high energy requirement: {energy_Wh:.3f} Wh")

    # 3) Add mode-switch edges with energy and time constraints
    for node in G_world.nodes():
        for m1 in modes_list:
            for m2 in modes_list:
                if m1 != m2:
                    switch_energy_wh = constants['SWITCH_ENERGY'] 
                    switch_time = constants['SWITCH_TIME']  

                    if not exceeds_battery_capacity(switch_energy_wh, constants['BATTERY_CAPACITY']):
                        L.add_edge(
                            (node, m1),
                            (node, m2),
                            time=switch_time,
                            energy_Wh=switch_energy_wh,
                            terrain='switch',
                            distance=0
                        )
                    # else:
                    #     print(f"Excluded mode-switch at node {node} from '{m1}' to '{m2}' due to high energy requirement.")

    return L



###############################################################################
# 3) LAYERED DIJKSTRA WITH RECHARGING
###############################################################################
def layered_dijkstra_with_battery(G_world, start, goal, modes, constants, dbg = False):
    L=build_layered_graph(G_world, modes, constants)


    dist = {}
    pred = {}
    recharged = {}

    # Auxiliary dictionary to track the best (minimum) time and used energy for each (node, mode)
    best_time_energy = {}
    best_time_energy[start] = 0.0, 0.0

    source = (*start, 0.0)
    dist[source] = 0.0
    pred[source] = None
    recharged[source] = False

    pq = [(0.0, source)]
    print(f"Initialized priority queue with source: {source} at time 0.0s") if dbg else None

    while pq:
        cur_time, current_state = heapq.heappop(pq)
        cur_node, cur_mode, cur_used = current_state
        print(f"\nPopped state from queue: Node {cur_node}, Mode '{cur_mode}', Used Energy {cur_used:.2f} Wh, Current Time {cur_time:.2f}s") if dbg else None        

        # Skip if we have already found a better path
        if cur_time > dist.get((cur_node, cur_mode, cur_used), math.inf):
            print(f" - Skipping state {current_state} as a better path was already found (dist={dist.get(current_state, math.inf):.2f}s)") if dbg else None
            continue

        # Check if we've reached the goal
        if (cur_node == goal[0]) and (cur_mode == goal[1]):
            print(f" - Reached goal: Node {goal[0]}, Mode '{goal[1]}' at time {cur_time:.2f}s") if dbg else None

            # Reconstruct the path
            final_time = cur_time
            path = []
            recharge_set = set()
            c = (cur_node, cur_mode, cur_used)
            while c is not None:
                path.append((c[0], c[1]))
                p = pred.get(c, None)
                if p is not None and recharged.get(c, False):
                    recharge_set.add((p[0], p[1]))
                    print(f"   - Recharge occurred at Node {p[0]}, Mode '{p[1]}'") if dbg else None
                c = p
            path.reverse()
            print(f" - Reconstructed path: {path}") if dbg else None
            print(f" - Recharge events: {recharge_set}") if dbg else None
            

            def get_final_energy(path):

                final_energy = 0.0
                for i in range(len(path) - 1):
                    (u_node, u_mode) = path[i]
                    (v_node, v_mode) = path[i + 1]
                    
                    # Include all edges, including mode-switch edges
                    if L.has_edge((u_node, u_mode), (v_node, v_mode)):
                        edge_en = L[(u_node, u_mode)][(v_node, v_mode)].get('energy_Wh', 0.0)
                        final_energy += edge_en

                return final_energy
            

            def find_mode_switch_nodes(path):
                s=set()
                for i in range(len(path)-1):
                    (u_node,u_mode)=path[i]
                    (v_node,v_mode)=path[i+1]
                    if (u_node==v_node) and (u_mode!=v_mode):
                        s.add(u_node)
                return s


            sorted_recharge_nodes = []
            for node_mode in path:
                if node_mode in recharge_set and node_mode not in sorted_recharge_nodes:
                    sorted_recharge_nodes.append(node_mode)


            final_energy = get_final_energy(path)
            switch_nodes = find_mode_switch_nodes(path)

            return (L, final_time, final_energy, path, sorted_recharge_nodes, switch_nodes)

        # Explore neighbors
        for nbr in L.successors((cur_node, cur_mode)):
            edge_data = L[(cur_node, cur_mode)][nbr]
            edge_time = edge_data['time']
            edge_energy = edge_data.get('energy_Wh', 0.0)
            (nbr_node, nbr_mode) = nbr

            print(f"   - Exploring neighbor: Node {nbr_node}, Mode '{nbr_mode}', Edge Time {edge_time:.2f}s, Edge Energy {edge_energy:.2f} Wh") if dbg else None

            if cur_used + edge_energy <= constants['BATTERY_CAPACITY']:
                new_used = cur_used + edge_energy
                new_time = cur_time + edge_time
                did_recharge = False
                print(f"     - No recharge needed. New Used Energy: {new_used:.2f} Wh, New Time: {new_time:.2f}s") if dbg else None

            else:
                recharge_time_adjusted = (cur_used / constants['BATTERY_CAPACITY']) * constants['RECHARGE_TIME']

                new_time = cur_time + recharge_time_adjusted + edge_time
                new_used = edge_energy  # After recharge, used_energy is set to edge_energy
                did_recharge = True  # Recharge occurred at current node

                print(f"     - Recharge needed. Energy Deficit: {cur_used:.2f} Wh") if dbg else None
                print(f"       - Recharge Time Adjusted: {recharge_time_adjusted:.2f}s") if dbg else None
                print(f"       - New Used Energy: {new_used:.2f} Wh, New Time: {new_time:.2f}s") if dbg else None

            next_state = (nbr_node, nbr_mode, new_used)

            ###
            # Add logic here to check what states exist. 
            # If a state with nbr_node and nbr_mode exists with less time and less used_energy, then skip adding the state to the heap
            if (nbr_node, nbr_mode) in best_time_energy:
                existing_time, existing_energy = best_time_energy[(nbr_node, nbr_mode)]
                if new_time >= existing_time and new_used >= existing_energy:
                    print(f"     - Existing state for Node {nbr_node}, Mode '{nbr_mode}' has less or equal time ({existing_time:.2f}s) and energy ({existing_energy:.2f} Wh). Skipping adding this state.") if dbg else None
                    continue
            # Update the best_time_energy dictionary with the new state
            best_time_energy[(nbr_node, nbr_mode)] = (new_time, new_used)
            ###

            # Update distance and predecessors if a better path is found
            if new_time < dist.get(next_state, math.inf):                
                dist[next_state] = new_time
                pred[next_state] = (cur_node, cur_mode, cur_used)
                recharged[next_state] = did_recharge
                heapq.heappush(pq, (new_time, next_state))
                print(f"     - Updated state: {next_state} with time {new_time:.2f}s and {'recharged' if did_recharge else 'no recharge'}") if dbg else None

            else:
                print(f"     - Existing state {next_state} has better or equal time. Skipping update.") if dbg else None


    return (L, math.inf, math.inf, [], set(), set())
