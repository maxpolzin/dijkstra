# %%
# %reload_ext autoreload
# %autoreload 2

# Uncomment the following line if you want interactive matplotlib widgets
# %matplotlib widget

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

###############################################################################
# Constants Definitions
###############################################################################

CONSTANTS = {
    'SWITCH_TIME': 100.0,        # seconds time penalty for mode switch
    'SWITCH_ENERGY': 1.0,        # Wh penalty for switching
    'BATTERY_CAPACITY': 15.0,    # Wh
    'RECHARGE_TIME': 1000.0,     # seconds
    'MODES': {
        'fly':   {'speed': 5.0,  'power': 1000.0},  # m/s, W
        'swim':  {'speed': 0.5,  'power':   10.0},  # m/s, W
        'roll':  {'speed': 3.0,  'power':    1.0},   # m/s, W
        'drive': {'speed': 1.0,  'power':   30.0},   # m/s, W
    }
}


###############################################################################
# Terrain and DEM Generation Functions
###############################################################################

def build_world():
    """
    Generates a deterministic synthetic world with DEM and terrain types.

    Returns:
    - dem (2D numpy array): Digital Elevation Model (in meters).
    - terrain (2D numpy array): Terrain types ('water' or 'grass').
    """
    # Initialize a 10x10 grid with zeros (DEM in meters)
    dem = np.zeros((10, 10))    
    terrain = np.full((10, 10), 'grass', dtype=object)

    # 1. Add River: x=300m to x=500m (columns 3,4), y=0m to y=700m (rows 0-6)
    river_start_x = 300  # meters
    river_end_x = 500    # meters
    river_start_y = 0     # meters
    river_end_y = 700    # meters
    # Convert meters to grid indices (100m per grid cell)
    river_start_col = int(river_start_x / 100)
    river_end_col = int(river_end_x / 100)
    river_start_row = int(river_start_y / 100)
    river_end_row = int(river_end_y / 100)
    dem[river_start_row:river_end_row, river_start_col:river_end_col] = 0  # River height
    terrain[river_start_row:river_end_row, river_start_col:river_end_col] = 'water'  # River terrain

    # 2. Add High Plateau in Top-Left with Surrounding Cliffs
    plateau_start_x = 0       # meters
    plateau_end_x = 1000      # meters
    plateau_start_y = 800      # meters
    plateau_end_y = 1000      # meters
    plateau_start_col = int(plateau_start_x / 100)
    plateau_end_col = int(plateau_end_x / 100)
    plateau_start_row = int(plateau_start_y / 100)
    plateau_end_row = int(plateau_end_y / 100)
    dem[plateau_start_row:plateau_end_row, plateau_start_col:plateau_end_col] = 100  # Plateau height

    # 3. Add Sloped Terrain: x=600m to x=800m (columns 6,7), y=0m to y=800m (rows 0-7)
    slope_start_x = 600  # meters
    slope_end_x = 800    # meters
    slope_start_y = 0     # meters
    slope_end_y = 800     # meters
    slope_start_col = int(slope_start_x / 100)
    slope_end_col = int(slope_end_x / 100)
    slope_start_row = int(slope_start_y / 100)
    slope_end_row = int(slope_end_y / 100)
    for j in range(slope_start_col, slope_end_col):
        slope_height = ((j + 1 - slope_start_col) / (slope_end_col - slope_start_col)) * 100  # Gradually increase to 100m
        dem[slope_start_row:slope_end_row, j] = slope_height

    # 4. Make the remaining right quarter flat again: x=800m to x=1000m (columns 8,9)
    flat_start_x = 800  # meters
    flat_end_x = 1000    # meters
    flat_start_col = int(flat_start_x / 100)
    flat_end_col = int(flat_end_x / 100)
    last_slope_height = dem[:, slope_end_col -1].copy()
    dem[:, flat_start_col:flat_end_col] = last_slope_height[:, np.newaxis]

    return dem, terrain


###############################################################################
# Visualization Function
###############################################################################

def visualize_world(dem, terrain, resolution_m=100):
    """
    Plots the DEM and terrain types as a 3D landscape with grid lines.

    Parameters:
    - dem (2D numpy array): Digital Elevation Model (in meters).
    - terrain (2D numpy array): Terrain types ('water' or 'grass').
    - resolution_m (int): Grid resolution in meters.
    """
    grid_size = dem.shape[0]
    size_m = grid_size * resolution_m
    x = np.linspace(0, size_m, grid_size)
    y = np.linspace(0, size_m, grid_size)
    X, Y = np.meshgrid(x, y)

    # Create color map based on terrain
    terrain_colors = np.empty(dem.shape, dtype=object)
    terrain_colors[terrain == 'water'] = 'blue'
    terrain_colors[terrain == 'grass'] = 'green'

    # Adjust figure size to be more manageable
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface with terrain colors and grid lines
    surf = ax.plot_surface(X, Y, dem, facecolors=terrain_colors, linewidth=0.5, edgecolor='gray', antialiased=False, shade=False)

    # Create a proxy mappable for the legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', edgecolor='blue', label='Water'),
                       Patch(facecolor='green', edgecolor='green', label='Grass')]
    ax.legend(handles=legend_elements, loc='upper right')

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Elevation (m)')
    ax.set_title('3D Visualization of the Synthetic World')

    # Improve contour visibility by adjusting view angle
    ax.view_init(elev=45, azim=-110)

    plt.show()

###############################################################################
# Simplified RRT* Algorithm Implementation
###############################################################################

@dataclass(order=True)
class State:
    used_time: float
    position: Tuple[float, float, float] = field(compare=False)  # (x, y, z) in meters
    parent: Optional['State'] = field(compare=False, default=None)

class RRTStar:
    def __init__(self, dem: np.ndarray, terrain: np.ndarray, constants: Dict):
        self.dem = dem
        self.terrain = terrain
        self.constants = constants
        self.start: Optional[State] = None
        self.goal: Optional[State] = None
        self.tree: List[State] = []
        self.step_size = 10.0  # meters
        self.max_iterations = 1000
        self.goal_threshold = 10.0  # meters

    def set_start(self, position: Tuple[float, float, float]):
        self.start = State(used_time=0.0, position=position, parent=None)
        self.tree.append(self.start)

    def set_goal(self, position: Tuple[float, float, float]):
        self.goal = State(used_time=0.0, position=position, parent=None)

    def is_goal_reached(self, state: State) -> bool:
        if not self.goal:
            return False
        distance = np.linalg.norm(np.array(state.position) - np.array(self.goal.position))
        return distance <= self.goal_threshold  # meters

    def sample_state(self) -> Tuple[float, float, float]:
        """
        Samples a random point within the world boundaries.

        Returns:
        - (x, y, z): Tuple of coordinates in meters.
        """
        x = np.random.uniform(0, 1000)
        y = np.random.uniform(0, 1000)
        z = np.random.uniform(0, 200)  # Assuming z up to 200 meters
        return (x, y, z)

    def nearest_neighbor(self, sampled_position: Tuple[float, float, float]) -> Optional[State]:
        """
        Finds the nearest neighbor in the tree to the sampled position.

        Parameters:
        - sampled_position (Tuple[float, float, float]): The (x, y, z) position to find the nearest neighbor for.

        Returns:
        - Optional[State]: The nearest neighbor state if found, else None.
        """
        min_dist = float('inf')
        nearest = None
        for node in self.tree:
            dist = np.linalg.norm(np.array(node.position) - np.array(sampled_position))
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest

    def steer(self, from_state: State, to_position: Tuple[float, float, float]) -> Optional[State]:
        """
        Attempts to move from `from_state` towards `to_position` by step_size.

        Parameters:
        - from_state (State): The current state.
        - to_position (Tuple[float, float, float]): The target (x, y, z) position.

        Returns:
        - Optional[State]: The new state if the move is valid, else None.
        """
        direction = np.array(to_position) - np.array(from_state.position)
        distance = np.linalg.norm(direction)
        if distance == 0:
            return None
        direction = direction / distance  # Normalize

        # Limit the step to step_size
        step = min(self.step_size, distance)
        new_position = np.array(from_state.position) + direction * step
        new_position = tuple(new_position)

        # Check boundaries
        if not self.is_within_bounds(new_position):
            return None

        # Create new state
        new_used_time = from_state.used_time + step  # Assuming used_time is the distance
        new_state = State(
            used_time=new_used_time,
            position=new_position,
            parent=from_state
        )
        return new_state

    def is_within_bounds(self, position: Tuple[float, float, float]) -> bool:
        x, y, z = position
        return (0 <= x <= 1000) and (0 <= y <= 1000) and (0 <= z <= 200)

    def build_tree(self) -> Optional[State]:
        """
        Builds the RRT* tree up to `max_iterations`.
        Returns the goal state if reached, else None.
        """
        for i in range(self.max_iterations):
            sampled_pos = self.sample_state()
            nearest = self.nearest_neighbor(sampled_pos)
            if nearest is None:
                continue

            new_state = self.steer(nearest, sampled_pos)
            if new_state is not None:
                self.tree.append(new_state)

                # Check if goal is reached
                if self.is_goal_reached(new_state):
                    self.goal.parent = new_state
                    self.goal.used_time = new_state.used_time + np.linalg.norm(np.array(new_state.position) - np.array(self.goal.position))
                    self.tree.append(self.goal)
                    return self.goal

        return None  # Goal not reached within max_iterations

    def extract_path(self, end_state: State) -> List[Tuple[float, float, float]]:
        """
        Extracts the path from start to end state.

        Parameters:
        - end_state (State): The goal state.

        Returns:
        - List of positions as tuples (x, y, z).
        """
        path = []
        current = end_state
        while current is not None:
            path.append(current.position)
            current = current.parent
        path.reverse()
        return path

###############################################################################
# Path Visualization Function (Removed as per user request)
###############################################################################
# The visualize_path function has been removed as per your instructions.

###############################################################################
# Main Execution: Building, Visualizing the World and Running RRT*
###############################################################################

def main():
    # 1. Build the world
    dem, terrain = build_world()
    visualize_world(dem, terrain, resolution_m=100)  # Each grid cell is 100m

    # 2. Initialize RRT*
    rrt_star = RRTStar(dem, terrain, CONSTANTS)

    # 3. Define start and goal positions
    # Start at (0, 0, 0) meters
    start_position = (0.0, 0.0, 0.0)  # (x, y, z)
    # Goal at (1000, 1000, 0) meters
    goal_position = (1000.0, 1000.0, 0.0)  # (x, y, z)

    rrt_star.set_start(start_position)
    rrt_star.set_goal(goal_position)

    # 4. Build the RRT* tree
    goal_state = rrt_star.build_tree()
    print("Tree size:", len(rrt_star.tree))
    print("Goal state:", goal_state)

    if goal_state:
        print("Goal reached!")
        path = rrt_star.extract_path(goal_state)
        print("Path:")
        for step in path:
            print(f"Position: {step}")
        
        # Note: visualize_path has been removed as per your request.
    else:
        print("Goal not reached within the maximum number of iterations.")

if __name__ == "__main__":
    main()

# %%
