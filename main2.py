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
# Main Execution: Building, Visualizing the World and Running RRT*
###############################################################################

def main():
    # 1. Build the world
    dem, terrain = build_world()
    visualize_world(dem, terrain, resolution_m=100)  # Each grid cell is 100m

if __name__ == "__main__":
    main()

# %%
