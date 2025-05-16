# %%

import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2
from scipy.ndimage import gaussian_filter

# Parameters
width = 30       # Width of the landscape
depth = 30       # Depth of the landscape
scale = 100.0    # Increased scale for larger, smoother features
octaves = 4      # Reduced octaves for less roughness
persistence = 0.5
lacunarity = 2.0
smoothing_sigma = 1.0  # Sigma for Gaussian smoothing

# Generate grid coordinates
x = np.linspace(0, width, width)
y = np.linspace(0, depth, depth)
x, y = np.meshgrid(x, y)

# Generate Perlin noise-based height map
def generate_height_map(width, depth, scale, octaves, persistence, lacunarity):
    height_map = np.zeros((depth, width))
    for i in range(depth):
        for j in range(width):
            height_map[i][j] = pnoise2(
                j / scale,
                i / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=42
            )
    # Normalize the height map to range [0, 1]
    min_val = np.min(height_map)
    max_val = np.max(height_map)
    height_map = (height_map - min_val) / (max_val - min_val)
    return height_map

height_map = generate_height_map(width, depth, scale, octaves, persistence, lacunarity)

# Apply Gaussian smoothing to the height map
def smooth_height_map(height_map, sigma=1.0):
    return gaussian_filter(height_map, sigma=sigma)

height_map = smooth_height_map(height_map, sigma=smoothing_sigma)

# Adjust height values: reduce by 0.3 and clamp to 0
def adjust_heights(height_map, reduction=0.3):
    height_map = height_map - reduction
    height_map = np.maximum(height_map, 0.0)  # Set negative values to 0
    return height_map

height_map = adjust_heights(height_map, reduction=0.3)

# Introduce a specific sharp cliff from x=0 to center y=middle
def add_diagonal_cliff(height_map, drop_magnitude=0.3):
    depth, width = height_map.shape
    center_x = width // 2
    center_y = depth // 2
    
    for i in range(min(center_x, center_y) + 1):
        height_map[i, i] -= drop_magnitude  # Apply drop along the diagonal
    height_map = np.maximum(height_map, 0.0)  # Clamp to 0
    return height_map

height_map = add_diagonal_cliff(height_map, drop_magnitude=0.3)

# Introduce an isolated elevated area with smooth blending
def add_elevated_area(height_map, center_x, center_y, radius=5, elevation=0.6):
    depth, width = height_map.shape
    y_indices, x_indices = np.indices((depth, width))
    distance = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    
    # Create a smooth mask using a Gaussian function for blending
    mask = np.exp(-(distance**2) / (2 * (radius / 2)**2))
    
    # Apply the elevation with smooth blending
    height_map += elevation * mask
    return height_map

# Define the center of the elevated area
center_x = width // 2
center_y = depth // 2

height_map = add_elevated_area(height_map, center_x, center_y, radius=5, elevation=0.6)

# Introduce sharp edges on the central elevated area
def add_sharp_edges_to_elevated_area(height_map, center_x, center_y, edge_length=3, drop_magnitude=0.4):
    depth, width = height_map.shape
    
    # Define two perpendicular edges: top and right
    # Top Edge
    start_row = center_y - 1
    end_row = start_row + 1  # Single row
    start_col = center_x - edge_length // 2
    end_col = start_col + edge_length
    height_map[start_row:end_row, start_col:end_col] -= drop_magnitude
    
    # Right Edge
    start_col = center_x + 1
    end_col = start_col + 1  # Single column
    start_row = center_y - edge_length // 2
    end_row = start_row + edge_length
    height_map[start_row:end_row, start_col:end_col] -= drop_magnitude
    
    # Clamp heights to ensure no negative values
    height_map = np.maximum(height_map, 0.0)
    
    return height_map

height_map = add_sharp_edges_to_elevated_area(height_map, center_x, center_y, edge_length=3, drop_magnitude=0.4)

# Normalize again after adjustments to ensure values are within [0, 1]
def normalize_height_map(height_map):
    min_val = np.min(height_map)
    max_val = np.max(height_map)
    if max_val - min_val != 0:
        height_map = (height_map - min_val) / (max_val - min_val)
    else:
        height_map = np.zeros_like(height_map)
    return height_map

height_map = normalize_height_map(height_map)

# Apply Gaussian smoothing again to ensure overall smoothness without affecting cliffs
height_map = smooth_height_map(height_map, sigma=smoothing_sigma)

# Create 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with antialiased edges for better visualization
surf = ax.plot_surface(
    x, y, height_map,
    cmap='terrain',
    linewidth=0,
    antialiased=True
)

# Customize the z axis
ax.set_zlim(0, 1)
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Height')

# Add a color bar which maps values to colors
fig.colorbar(surf, shrink=0.5, aspect=10, label='Height')

# Set the view angle for better visualization
ax.view_init(elev=60, azim=-45)

plt.title('Smoother 3D Landscape with Specific Sharp Cliff and Elevated Area')
plt.show()
