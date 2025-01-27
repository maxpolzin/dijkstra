# %% rrt_star_3d.py

%reload_ext autoreload
%autoreload 2

# Uncomment the following line if you want interactive matplotlib widgets
%matplotlib widget


###############################################################################
# 1) Imports
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

###############################################################################
# 2) Global Constants
###############################################################################
MAX_ITER = 5000           # Maximum iterations for RRT*

GOAL_SAMPLE_RATE = 0.01    # Probability of directly sampling the goal
EXPAND_DIS = 20.0          # Extension distance in steer
CONNECT_RADIUS = 200.0     # Radius used in find_near_nodes

# Start and goal in (x, y, z)
SIZE = 1000               # Dimensions for DEM creation
RAND_MIN_XY = -50
RAND_MAX_XY = SIZE + 50

RAND_MIN_Z = 0
RAND_MAX_Z = 200


START = (0, 0, 0)
GOAL  = (1000, 200, 0)


###############################################################################
# 3) RRT* Classes and Methods
###############################################################################

def calculate_distance(node1, node2):
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2 + (node1.z - node2.z)**2)

def calculate_distance_and_angle(node1, node2):
    dx = node2.x - node1.x
    dy = node2.y - node1.y
    dz = node2.z - node1.z
    d = calculate_distance(node1, node2)
    if d == 0:
        return 0.0, 0.0, 0.0

    # theta is angle in the xy-plane, phi is angle down from the z-axis
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
    def __init__(self, dem, terrain):
        # Use global constants
        self.start = Node(*START)
        self.end   = Node(*GOAL)
        self.node_list = [self.start]
        self.dem = dem
        self.terrain = terrain

    def planning(self):
        for _ in range(MAX_ITER):
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node(rnd_node)
            new_node = self.steer(nearest_node, rnd_node)

            if self.is_feasible_path(nearest_node, new_node):
                near_nodes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, nearest_node, near_nodes)
                self.node_list.append(new_node)
                self.rewire(new_node, near_nodes)

            if self.reached_goal(new_node):
                return self.generate_final_course(new_node)

        return None

    def get_random_node(self):
        if random.random() <= GOAL_SAMPLE_RATE:
            return Node(self.end.x, self.end.y, self.end.z)
        else:
            x = random.uniform(RAND_MIN_XY, RAND_MAX_XY)
            y = random.uniform(RAND_MIN_XY, RAND_MAX_XY)
            z = random.uniform(RAND_MIN_Z, RAND_MAX_Z)
            return Node(x, y, z)

    def get_elevation(self, x, y):
        size_x = self.dem.shape[1]
        size_y = self.dem.shape[0]

        # Clamp coordinates to DEM range
        x_clamped = max(0, min(size_x - 1, int(round(x))))
        y_clamped = max(0, min(size_y - 1, int(round(y))))

        return self.dem[y_clamped][x_clamped]

    def get_nearest_node(self, rnd_node):
        return min(self.node_list, key=lambda node: calculate_distance(node, rnd_node))

    def steer(self, from_node, to_node):
        d, theta, phi = calculate_distance_and_angle(from_node, to_node)
        dist = min(EXPAND_DIS, d)
        new_node = Node(from_node.x, from_node.y, from_node.z)
        if d > 0:
            new_node.x += dist * math.cos(theta) * math.sin(phi)
            new_node.y += dist * math.sin(theta) * math.sin(phi)
            new_node.z += dist * math.cos(phi)
        new_node.parent = from_node
        new_node.cost = from_node.cost + calculate_cost(from_node, new_node)
        return new_node

    def is_feasible_path(self, node1, node2):
        """
        Placeholder feasibility check (always returns True).
        You could check for collisions, terrain constraints, etc.
        """
        return True

    def find_near_nodes(self, new_node):
        n_nodes = len(self.node_list) + 1
        r = CONNECT_RADIUS * (math.log(n_nodes) / n_nodes)**(1.0 / 3.0)
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
            if self.is_feasible_path(new_node, near_node):
                new_cost = new_node.cost + calculate_cost(new_node, near_node)
                if new_cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = new_cost

    def reached_goal(self, node):
        return calculate_distance(node, self.end) <= EXPAND_DIS

    def generate_final_course(self, last_node):
        path = [[self.end.x, self.end.y, self.end.z]]
        n = last_node
        while n is not None:
            path.append([n.x, n.y, n.z])
            n = n.parent
        path.reverse()
        return path

###############################################################################
# 4) World-Building Function
###############################################################################
def build_world():
    dem = np.zeros((SIZE, SIZE))
    terrain = np.full((SIZE, SIZE), 'grass', dtype=object)

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

###############################################################################
# 5) Visualization Functions
###############################################################################
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
        alpha=0.6
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
        ax.plot(path_arr[:, 0], path_arr[:, 1], path_arr[:, 2],
                color="red", linewidth=2, zorder=10)

    # Mark start and goal
    ax.scatter(rrt_star.start.x, rrt_star.start.y, rrt_star.start.z,
               color="green", s=100, label="Start", zorder=20)
    ax.scatter(rrt_star.end.x, rrt_star.end.y, rrt_star.end.z,
               color="magenta", s=100, label="Goal", zorder=20)

    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', label='Water'),
        Patch(facecolor='green', edgecolor='green', label='Grass')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('3D RRT* Path on DEM')
    ax.view_init(elev=45, azim=-90)
    plt.show()

###############################################################################
# 6) Main Block
###############################################################################
def main():
    dem, terrain = build_world()
    rrt_star = RRTStar(dem, terrain)

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
