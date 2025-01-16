# %%
import networkx as nx
import matplotlib.pyplot as plt
import random
import heapq

# -----------------------------
# Build the directed graph
# -----------------------------
DG = nx.DiGraph()
nodes = range(1, 9)
DG.add_nodes_from(nodes)

directed_edges = [
    (1, 2, 5),
    (2, 1, 6),
    (1, 3, random.randint(0, 10)),
    (3, 1, random.randint(0, 10)),
    (2, 3, random.randint(0, 10)),
    (3, 2, random.randint(0, 10)),
    (2, 4, random.randint(0, 10)),
    (4, 2, random.randint(0, 10)),
    (3, 5, random.randint(0, 10)),
    (5, 3, random.randint(0, 10)),
    (4, 5, random.randint(0, 10)),
    (5, 4, random.randint(0, 10)),
    (5, 6, random.randint(0, 10)),
    (6, 5, random.randint(0, 10)),
    (5, 7, random.randint(0, 10)),
    (7, 5, random.randint(0, 10)),
    (6, 7, random.randint(0, 10)),
    (7, 6, random.randint(0, 10)),
    (4, 8, random.randint(0, 10)),
    (8, 4, random.randint(0, 10)),
    (6, 8, random.randint(0, 10)),
    (8, 6, random.randint(0, 10)),
    (7, 8, random.randint(0, 10)),
    (8, 7, random.randint(0, 10))
]

DG.add_weighted_edges_from(directed_edges)

# -----------------------------
# Original Dijkstra's algorithm (no recharge)
# -----------------------------
def dijkstra(graph, start, target):
    pq = [(0, start)]
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    predecessors = {node: None for node in graph.nodes}

    while pq:
        current_distance, current_node = heapq.heappop(pq)
        # If we pop the target, we can reconstruct the path
        if current_node == target:
            path = []
            while current_node:
                path.append(current_node)
                current_node = predecessors[current_node]
            return current_distance, path[::-1]

        # If we pulled a distance that's not current, skip
        if current_distance > distances[current_node]:
            continue

        for neighbor in graph.neighbors(current_node):
            edge_weight = graph[current_node][neighbor]['weight']
            distance = current_distance + edge_weight

            # Relaxation
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    return float('inf'), []

# -----------------------------
# Modified Dijkstra with recharge
# -----------------------------
def dijkstra_with_recharge(graph, start, target, threshold=5, recharge_time=5):
    """
    If the partial distance to a neighbor exceeds 'threshold',
    add 'recharge_time' to the cost.
    """
    pq = [(0, start)]
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    predecessors = {node: None for node in graph.nodes}

    while pq:
        current_distance, current_node = heapq.heappop(pq)
        # If we pop the target, we can reconstruct the path
        if current_node == target:
            path = []
            while current_node:
                path.append(current_node)
                current_node = predecessors[current_node]
            return current_distance, path[::-1]

        # If we pulled a distance that's not current, skip
        if current_distance > distances[current_node]:
            continue

        for neighbor in graph.neighbors(current_node):
            edge_weight = graph[current_node][neighbor]['weight']
            # Compute new distance
            distance = current_distance + edge_weight

            # If exceeding threshold, add recharge penalty
            if distance > threshold:
                distance += recharge_time

            # Relaxation
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    return float('inf'), []

# -----------------------------
# Compare the two algorithms
# -----------------------------
shortest_distance, shortest_path = dijkstra(DG, 1, 8)
print("=== Original Dijkstra ===")
print("Shortest distance:", shortest_distance)
print("Shortest path:", shortest_path)

shortest_distance_recharge, shortest_path_recharge = dijkstra_with_recharge(
    DG, 1, 8, threshold=5, recharge_time=5
)
print("\n=== Modified Dijkstra (Recharge) ===")
print("Shortest distance:", shortest_distance_recharge)
print("Shortest path:", shortest_path_recharge)

# -----------------------------
# Plot the graph with the original Dijkstra result
# -----------------------------
pos = nx.spring_layout(DG, seed=42)  # Layout for consistent visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
nx.draw(
    DG, pos,
    with_labels=True,
    node_color='lightgreen',
    node_size=500,
    font_size=12,
    font_weight='bold',
    arrowsize=15
)
plt.title("All Edges (Original Distances)")

# Build edge labels
all_edges = nx.get_edge_attributes(DG, 'weight')
nx.draw_networkx_edge_labels(DG, pos, edge_labels=all_edges, font_size=10, label_pos=0.25)

# Highlight the original shortest path
if len(shortest_path) > 1:
    path_edges = list(zip(shortest_path, shortest_path[1:]))
    nx.draw_networkx_edges(
        DG,
        pos,
        edgelist=path_edges,
        edge_color='red',
        arrows=True,
        arrowsize=15,
        width=2
    )

plt.subplot(1, 2, 2)
nx.draw(
    DG, pos,
    with_labels=True,
    node_color='lightgreen',
    node_size=500,
    font_size=12,
    font_weight='bold',
    arrowsize=15
)
plt.title("All Edges (Recharge Algorithm)")

# Same edges/labels for consistency
nx.draw_networkx_edge_labels(DG, pos, edge_labels=all_edges, font_size=10, label_pos=0.25)

# Highlight the recharge shortest path
if len(shortest_path_recharge) > 1:
    path_edges_recharge = list(zip(shortest_path_recharge, shortest_path_recharge[1:]))
    nx.draw_networkx_edges(
        DG,
        pos,
        edgelist=path_edges_recharge,
        edge_color='blue',
        arrows=True,
        arrowsize=15,
        width=2
    )

plt.tight_layout()
plt.show()

# %% 