#%%
%reload_ext autoreload
%autoreload 2

# Uncomment the following line if you want interactive matplotlib widgets
%matplotlib widget

import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time

###############################################################################
# World Building & Visualization (2D)
###############################################################################

def build_world():
    size = 1000
    dem = np.zeros((size, size))
    terrain = np.full((size, size), 'grass', dtype=object)

    river_start_x = 200
    river_end_x   = 350
    river_start_y = 0
    river_end_y   = 700
    dem[river_start_y:river_end_y, river_start_x:river_end_x] = 0
    terrain[river_start_y:river_end_y, river_start_x:river_end_x] = 'water'

    plateau_start_y = 800
    plateau_end_y   = 1000
    plateau_start_x = 0
    plateau_end_x   = 850
    dem[plateau_start_y:plateau_end_y, plateau_start_x:plateau_end_x] = 100

    slope_start_x = 400
    slope_end_x   = 700
    slope_start_y = 0
    slope_end_y   = 800
    for col in range(slope_start_x, slope_end_x):
        slope_height = ((col - slope_start_x + 1) / (slope_end_x - slope_start_x)) * 100
        dem[slope_start_y:slope_end_y, col] = slope_height

    flat_start_x = 700
    flat_end_x   = 850
    last_slope_height = dem[:, slope_end_x - 1].copy()
    dem[:, flat_start_x:flat_end_x] = last_slope_height[:, np.newaxis]

    return dem, terrain

def visualize_world_and_path(dem, terrain, rrt_star, path=None):
    """
    2D visualization of the DEM and the RRT* tree/path.
    """

    plt.figure(figsize=(10, 8))
    # Display DEM using imshow in 2D
    plt.imshow(dem, origin='lower', cmap='terrain', alpha=0.6)

    # Plot the RRT* tree edges
    for node in rrt_star.node_list:
        if node.parent:
            plt.plot([node.x, node.parent.x],
                     [node.y, node.parent.y],
                     color="blue", linewidth=0.5)

    # Plot the final path if it exists
    if path:
        path_arr = np.array(path)
        plt.plot(path_arr[:, 0], path_arr[:, 1], color="red", linewidth=2)

    # Mark start and goal
    plt.scatter(rrt_star.start.x, rrt_star.start.y,
                color="green", s=100, label="Start")
    plt.scatter(rrt_star.end.x, rrt_star.end.y,
                color="magenta", s=100, label="Goal")

    plt.title("2D RRT* on DEM")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


###############################################################################
# RRT* Implementation (2D)
###############################################################################

def calculate_distance_2d(node1, node2):
    return math.hypot(node1.x - node2.x, node1.y - node2.y)

def calculate_cost(from_node, to_node):
    return calculate_distance_2d(from_node, to_node)

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, dem, terrain, start, goal, rand_area, expand_dis):
        self.MAX_ITER = 5000
        self.CONNECT_RADIUS = 500.0
        self.start = Node(*start)
        self.end   = Node(*goal)
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.node_list = [self.start]
        self.dem = dem
        self.terrain = terrain

    def planning(self):
        for _ in range(self.MAX_ITER):
            rnd_node = self.get_random_node()

            nearest_node = self.get_nearest_node(rnd_node)
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # Basic feasibility check: skip if not feasible. (Optional to refine)
            if not self.is_feasible_path(nearest_node, new_node):
                continue

            near_nodes = self.find_near_nodes(new_node)
            new_node = self.choose_parent(new_node, nearest_node, near_nodes)
            self.node_list.append(new_node)
            self.rewire(new_node, near_nodes)

            if self.reached_goal(new_node):
                return self.generate_final_course(new_node)

        return None

    def get_random_node(self):
        GOAL_SAMPLE_RATE = 0.0
        if random.random() <= GOAL_SAMPLE_RATE:
            return Node(self.end.x, self.end.y)
        else:
            x = random.uniform(self.min_rand, self.max_rand)
            y = random.uniform(self.min_rand, self.max_rand)
            return Node(x, y)

    def get_nearest_node(self, rnd_node):
        return min(self.node_list, key=lambda node: calculate_distance_2d(node, rnd_node))

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Move from 'from_node' toward 'to_node' by 'extend_length' in 2D.
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dist = math.hypot(dx, dy)
        if dist == 0:
            return from_node

        actual_dist = min(extend_length, dist)
        ratio = actual_dist / dist

        new_x = from_node.x + ratio * dx
        new_y = from_node.y + ratio * dy

        new_node = Node(new_x, new_y)
        new_node.parent = from_node
        new_node.cost = from_node.cost + calculate_cost(from_node, new_node)
        return new_node

    def is_feasible_path(self, node1, node2):
        """
        Placeholder feasibility check (always True).
        You could add collision checks, terrain checks, etc.
        """
        return True

    def find_near_nodes(self, new_node):
        n_nodes = len(self.node_list) + 1
        r = self.CONNECT_RADIUS * (math.log(n_nodes) / n_nodes)**(1.0 / 3.0)
        return [node for node in self.node_list if calculate_distance_2d(node, new_node) <= r]

    def choose_parent(self, new_node, nearest_node, near_nodes):
        min_cost = nearest_node.cost + calculate_cost(nearest_node, new_node)
        best_node = nearest_node
        for near_node in near_nodes:
            if self.is_feasible_path(near_node, new_node):
                cost = near_node.cost + calculate_cost(near_node, new_node)
                if cost < min_cost:
                    min_cost = cost
                    best_node = near_node
        new_node.cost = min_cost
        new_node.parent = best_node
        return new_node

    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            if self.is_feasible_path(new_node, near_node):
                new_cost = new_node.cost + calculate_cost(new_node, near_node)
                if new_cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = new_cost

    def reached_goal(self, node):
        return calculate_distance_2d(node, self.end) <= self.expand_dis

    def generate_final_course(self, last_node):
        path = [[self.end.x, self.end.y]]
        n = last_node
        while n is not None:
            path.append([n.x, n.y])
            n = n.parent
        path.reverse()
        return path

###############################################################################
# Main
###############################################################################

def main():
    dem, terrain = build_world()

    size = dem.shape[0] - 1

    # 2D Start & Goal: (x, y)
    start = [0, 0]
    goal  = [1000, 800]

    # 2D sampling range
    rand_area = [-50, size + 50]

    print("Start 2D RRT* path planning")

    rrt_star = RRTStar(dem, terrain, start, goal, rand_area, expand_dis=1.0)

    start_time = time.perf_counter()
    path = rrt_star.planning()
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Planning method runtime: {elapsed_time:.6f} seconds")

    if path:
        print("Path found!")
    else:
        print("No path found")

    visualize_world_and_path(dem, terrain, rrt_star, path)

if __name__ == "__main__":
    main()
