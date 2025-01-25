#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random
import time

###############################################################################
# World Building & Combined Visualization
###############################################################################

def build_world():
    size = 1000
    dem = np.zeros((size, size))
    terrain = np.full((size, size), 'grass', dtype=object)

    river_start_x = 300
    river_end_x   = 500
    river_start_y = 0
    river_end_y   = 700
    dem[river_start_y:river_end_y, river_start_x:river_end_x] = 0
    terrain[river_start_y:river_end_y, river_start_x:river_end_x] = 'water'

    plateau_start_y = 800
    plateau_end_y   = 1000
    plateau_start_x = 0
    plateau_end_x   = 1000
    dem[plateau_start_y:plateau_end_y, plateau_start_x:plateau_end_x] = 100

    slope_start_x = 600
    slope_end_x   = 800
    slope_start_y = 0
    slope_end_y   = 800
    for col in range(slope_start_x, slope_end_x):
        slope_height = ((col - slope_start_x + 1) / (slope_end_x - slope_start_x)) * 100
        dem[slope_start_y:slope_end_y, col] = slope_height

    flat_start_x = 800
    flat_end_x   = 1000
    last_slope_height = dem[:, slope_end_x - 1].copy()
    dem[:, flat_start_x:flat_end_x] = last_slope_height[:, np.newaxis]

    return dem, terrain

def visualize_world_and_path(dem, terrain, rrt_star, path=None):
    size = dem.shape[0]
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

    terrain_colors = np.empty(dem.shape, dtype=object)
    terrain_colors[terrain == 'water'] = 'blue'
    terrain_colors[terrain == 'grass'] = 'green'

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set partial transparency on the terrain so path is always visible
    surf = ax.plot_surface(
        X, Y, dem,
        facecolors=terrain_colors,
        linewidth=0.5,
        edgecolor='gray',
        antialiased=False,
        shade=False,
        alpha=0.6  # <--- Transparency so we can see objects under surface
    )

    # Plot edges of the RRT* tree
    for node in rrt_star.node_list:
        if node.parent:
            xs = [node.x, node.parent.x]
            ys = [node.y, node.parent.y]
            zs = [node.z, node.parent.z]
            ax.plot(xs, ys, zs, color="blue", linewidth=0.5, zorder=5)

    # Plot the path in red with higher zorder
    if path:
        path_arr = np.array(path)
        ax.plot(path_arr[:, 0],
                path_arr[:, 1],
                path_arr[:, 2],
                color="red",
                linewidth=2,
                zorder=10)

    # Mark start and goal
    ax.scatter(rrt_star.start.x, rrt_star.start.y, rrt_star.start.z,
               color="green", s=100, label="Start", zorder=20)
    ax.scatter(rrt_star.end.x, rrt_star.end.y, rrt_star.end.z,
               color="magenta", s=100, label="Goal", zorder=20)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', label='Water'),
        Patch(facecolor='green', edgecolor='green', label='Grass')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('World + RRT* Path (Surface Partially Transparent)')
    ax.view_init(elev=45, azim=-110)

    # Optionally adjust how Matplotlib orders 3D elements
    # ax.set_proj_type('ortho')        # Orthographic projection
    # ax.set_zsort('min')             # or 'max' or 'none'

    plt.show()


###############################################################################
# RRT* Implementation with Ground Checks
###############################################################################

def calculate_distance(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2 + (node1.z - node2.z)**2)

def calculate_distance_and_angle(node1, node2):
    dx = node2.x - node1.x
    dy = node2.y - node1.y
    dz = node2.z - node1.z
    d = calculate_distance(node1, node2)
    theta = math.atan2(dy, dx)
    phi = math.atan2(math.sqrt(dx**2 + dy**2), dz)
    return d, theta, phi

def calculate_cost(from_node, to_node):
    return calculate_distance(from_node, to_node)

class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, rand_area, expand_dis, dem):
        self.MAX_ITER = 5000
        self.CONNECT_RADIUS = 2.0
        self.start = Node(*start)
        self.end   = Node(*goal)
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.node_list = [self.start]
        self.dem = dem

    def planning(self):
        for _ in range(self.MAX_ITER):
            rnd_node = self.get_random_node()
            # If the random node itself is infeasible, skip
            if not self.is_feasible_node(rnd_node):
                continue

            nearest_node = self.get_nearest_node(rnd_node)
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # If new_node is below ground or out of map => infeasible
            if not self.is_feasible_node(new_node):
                continue

            # Check if the path from nearest_node -> new_node is collision-free (above ground)
            if self.is_feasible_path(nearest_node, new_node):
                near_nodes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, nearest_node, near_nodes)
                self.node_list.append(new_node)
                self.rewire(new_node, near_nodes)

            if self.reached_goal(new_node):
                return self.generate_final_course(new_node)

        return None

    def get_random_node(self):
        GOAL_SAMPLE_RATE = 5
        if random.randint(0, 100) > GOAL_SAMPLE_RATE:
            x = random.uniform(self.min_rand, self.max_rand)
            y = random.uniform(self.min_rand, self.max_rand)
            z = random.uniform(self.min_rand, self.max_rand)/8
            return Node(x, y, z)
        else:
            return Node(self.end.x, self.end.y, self.end.z)

    def get_nearest_node(self, rnd_node):
        return min(self.node_list, key=lambda node: calculate_distance(node, rnd_node))

    def steer(self, from_node, to_node, extend_length=float("inf")):
        d, theta, phi = calculate_distance_and_angle(from_node, to_node)
        dist = min(extend_length, d)
        new_node = Node(from_node.x, from_node.y, from_node.z)
        new_node.x += dist * math.cos(theta) * math.sin(phi)
        new_node.y += dist * math.sin(theta) * math.sin(phi)
        new_node.z += dist * math.cos(phi)
        new_node.parent = from_node
        new_node.cost = from_node.cost + calculate_cost(from_node, new_node)
        return new_node

    def is_feasible_node(self, node):
        # Round or floor to nearest integer indices to access dem
        ix = int(round(node.x))
        iy = int(round(node.y))

        # Check boundary
        if ix < 0 or iy < 0 or ix >= self.dem.shape[1] or iy >= self.dem.shape[0]:
            return False

        ground_height = self.dem[iy, ix]
        if node.z < ground_height:
            return False
        return True

    def is_feasible_path(self, node1, node2):
        # Sample along the line in small steps to ensure above ground
        steps = 10
        for i in range(1, steps + 1):
            t = i / steps
            x = node1.x + t * (node2.x - node1.x)
            y = node1.y + t * (node2.y - node1.y)
            z = node1.z + t * (node2.z - node1.z)

            ix = int(round(x))
            iy = int(round(y))
            if ix < 0 or iy < 0 or ix >= self.dem.shape[1] or iy >= self.dem.shape[0]:
                return False
            ground_height = self.dem[iy, ix]
            if z < ground_height:
                return False
        return True

    def find_near_nodes(self, new_node):
        n_nodes = len(self.node_list) + 1
        r = self.CONNECT_RADIUS * (math.log(n_nodes) / n_nodes)**(1.0 / 3.0)
        return [node for node in self.node_list if calculate_distance(node, new_node) <= r]

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
            # Check feasibility again in case of new edge
            if self.is_feasible_path(new_node, near_node):
                new_cost = new_node.cost + calculate_cost(new_node, near_node)
                if new_cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = new_cost

    def reached_goal(self, node):
        return calculate_distance(node, self.end) <= self.expand_dis

    def generate_final_course(self, last_node):
        path = [[self.end.x, self.end.y, self.end.z]]
        n = last_node
        while n is not None:
            path.append([n.x, n.y, n.z])
            n = n.parent
        path.reverse()
        return path

###############################################################################
# Main
###############################################################################

def main():
    dem, terrain = build_world()

    size = dem.shape[0] - 1
    end_height   = dem[size, size]
    start_height = dem[0, 0]

    # Start at top-left corner, on ground
    start = [0, 0, start_height]
    # Goal at bottom-right corner, on ground
    goal  = [size, size, end_height]

    # We allow random sampling from -50.. (size+50) for x,y,z
    rand_area = [-50, size + 50]

    print("Start RRT* path planning in 3D, ensuring each node/edge is above ground")

    rrt_star = RRTStar(start, goal, rand_area, expand_dis=5.0, dem=dem)

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
