# %%
%reload_ext autoreload
%autoreload 2

# Uncomment the following line if you want interactive matplotlib widgets
%matplotlib widget

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



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


def build_world():

    dem = np.zeros((10, 10))    
    terrain = np.full((10, 10), 'grass', dtype=object)

    river_start_x = 3
    river_end_x = 5
    river_start_y = 0
    river_end_y = 7
    dem[river_start_y:river_end_y, river_start_x:river_end_x] = 0  # River height
    terrain[river_start_y:river_end_y, river_start_x:river_end_x] = 'water'  # River terrain

    plateau_start_x = 0
    plateau_end_x = 9
    plateau_start_y = 8
    plateau_end_y = 10
    dem[plateau_start_y:plateau_end_y, plateau_start_x:plateau_end_x] = 10  # Plateau height

    slope_start_x = 6
    slope_end_x = 8
    slope_start_y = 0
    slope_end_y = 8
    for j in range(slope_start_x, slope_end_x):
        slope_height = (j+1 - slope_start_x) / (slope_end_x - slope_start_x) * 10  # Gradually increase to 10
        dem[slope_start_y:slope_end_y, j] = slope_height

    flat_start_x = 8
    flat_end_x = 10
    last_slope_height = dem[:, slope_end_x -1].copy()
    dem[:, flat_start_x:flat_end_x] = last_slope_height[:, np.newaxis]

    return dem, terrain


def visualize_world(dem, terrain, resolution_m=1):
    grid_size = dem.shape[0]
    size_m = grid_size * resolution_m
    x = np.linspace(0, size_m, grid_size)
    y = np.linspace(0, size_m, grid_size)
    X, Y = np.meshgrid(x, y)

    terrain_colors = np.empty(dem.shape, dtype=object)
    terrain_colors[terrain == 'water'] = 'blue'
    terrain_colors[terrain == 'grass'] = 'green'

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, dem, facecolors=terrain_colors, linewidth=0.5, edgecolor='gray', antialiased=False, shade=False)

    ax.view_init(elev=45, azim=-110)

    plt.show()



def main():

    dem, terrain = build_world()
    visualize_world(dem, terrain)




if __name__ == "__main__":
    main()

# %%
