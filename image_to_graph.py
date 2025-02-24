import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
import rasterio
from rasterio.transform import xy
from rasterio import Affine
from skimage.transform import resize
from scipy.spatial import Delaunay

def downsample_dem(dem, transform, new_shape=(500, 500)):
    """
    Downsample the DEM array to new_shape using bilinear interpolation
    and update the affine transform accordingly.
    """
    original_rows, original_cols = dem.shape
    new_rows, new_cols = new_shape

    # Downsample using bilinear interpolation.
    dem_downsampled = resize(dem, (new_rows, new_cols), order=1,
                             preserve_range=True, anti_aliasing=True)
    dem_downsampled = dem_downsampled.astype(dem.dtype)

    # Compute scale factors.
    scale_x = original_cols / new_cols
    scale_y = original_rows / new_rows

    # Update the transform.
    new_transform = transform * Affine.scale(scale_x, scale_y)
    return dem_downsampled, new_transform

def quadtree_decompose(dem, r0, r1, c0, c1, std_threshold):
    """
    Recursively subdivide the DEM region [r0:r1, c0:c1] if the elevation standard deviation
    exceeds std_threshold. Returns a list of cells, where each cell is a tuple:
      (r0, r1, c0, c1, mean_elevation)
    """
    cell = dem[r0:r1, c0:c1]
    if cell.size == 0:
        return []
    
    cell_std = np.std(cell)
    # Stop subdividing if variation is low or if the cell is very small.
    if cell_std <= std_threshold or (r1 - r0 <= 2 and c1 - c0 <= 2):
        return [(r0, r1, c0, c1, np.mean(cell))]
    else:
        rm = (r0 + r1) // 2
        cm = (c0 + c1) // 2
        cells = []
        cells.extend(quadtree_decompose(dem, r0, rm, c0, cm, std_threshold))
        cells.extend(quadtree_decompose(dem, r0, rm, cm, c1, std_threshold))
        cells.extend(quadtree_decompose(dem, rm, r1, c0, cm, std_threshold))
        cells.extend(quadtree_decompose(dem, rm, r1, cm, c1, std_threshold))
        return cells

def create_quadtree_nodes(tif_path, target_shape=(500, 500), std_threshold=5.0):
    """
    Reads the DEM from a TIFF file, downsamples it, and applies quadtree decomposition.
    Returns a list of nodes, where each node is a dictionary containing:
      - 'centroid': (x, y) coordinate computed from the cell center,
      - 'elevation': average elevation in the cell,
      - 'bounds': the cell bounds as (r0, r1, c0, c1)
    """
    # Open the DEM.
    with rasterio.open(tif_path) as src:
        dem = src.read(1)
        transform = src.transform

    # Downsample the DEM.
    dem_ds, new_transform = downsample_dem(dem, transform, new_shape=target_shape)
    rows, cols = dem_ds.shape

    # Perform quadtree decomposition.
    cells = quadtree_decompose(dem_ds, 0, rows, 0, cols, std_threshold)

    # Create nodes from each quadtree leaf cell.
    nodes = []
    for cell in cells:
        r0, r1, c0, c1, mean_elev = cell
        # Compute the centroid in pixel coordinates.
        center_r = (r0 + r1) / 2
        center_c = (c0 + c1) / 2
        # Convert the pixel centroid to spatial (XY) coordinates.
        x, y = xy(new_transform, center_r, center_c)
        node = {
            'centroid': (x, y),
            'elevation': mean_elev,
            'bounds': (r0, r1, c0, c1)
        }
        nodes.append(node)
    return nodes, dem_ds, new_transform

def create_quadtree_3d_mesh(tif_path, target_shape=(500,500), std_threshold=5.0):
    """
    Uses quadtree segmentation to generate nodes from the DEM, then computes a Delaunay 
    triangulation on the nodes' centroids to form a 3D height mesh.
    
    Returns:
      - points: an array of XY coordinates from the cell centroids.
      - heights: an array of average elevations for each node.
      - tri: the Delaunay triangulation object computed on the XY points.
      - nodes: the list of node dictionaries.
      - dem_ds: the downsampled DEM array.
      - new_transform: the updated affine transform.
    """
    nodes, dem_ds, new_transform = create_quadtree_nodes(tif_path, target_shape, std_threshold)
    
    # Extract centroid coordinates and elevations.
    points = []
    heights = []
    for node in nodes:
        x, y = node['centroid']
        points.append([x, y])
        heights.append(node['elevation'])
    points = np.array(points)
    heights = np.array(heights)
    
    # Compute Delaunay triangulation on the XY coordinates.
    if len(points) >= 3:
        tri = Delaunay(points)
    else:
        tri = None
    return points, heights, tri, nodes, dem_ds, new_transform

def plot_quadtree_3d_mesh(points, heights, tri, nodes, dem_ds, transform):
    """
    Visualizes the quadtree segmentation result as a 3D height mesh.
    
    - The Delaunay triangulation of the node centroids is used to create a surface.
    - Additionally, the boundaries of each quadtree cell are overlayed.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the Delaunay triangulated surface.
    if tri is not None:
        surf = ax.plot_trisurf(points[:, 0], points[:, 1], heights,
                               triangles=tri.simplices, cmap='terrain', edgecolor='none', alpha=0.8)
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Elevation')
    else:
        ax.scatter(points[:, 0], points[:, 1], heights, c='r', marker='o')
    
    # Plot the boundaries of each quadtree cell.
    # for node in nodes:
    #     r0, r1, c0, c1 = node['bounds']
    #     # Define the corners of the cell in pixel coordinates.
    #     corners_pixels = [(r0, c0), (r0, c1), (r1, c1), (r1, c0), (r0, c0)]
    #     # Convert corners to spatial (XY) coordinates.
    #     corners_xy = [xy(transform, r, c) for r, c in corners_pixels]
    #     xs, ys = zip(*corners_xy)
    #     # Use the node's average elevation for the Z value.
    #     zs = [node['elevation']] * len(corners_xy)
    #     ax.plot(xs, ys, zs, color='k', lw=1, alpha=0.6)
    
    ax.set_title("3D Height Mesh from Quadtree Segmentation")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Elevation")
    plt.show()

def main():
    tif_path = 'swissalti3d_2021_2568-1140_0.5_2056_5728.tif'
    
    # Parameters: adjust target_shape and std_threshold as needed.
    target_shape = (500, 500)
    std_threshold = 25.0
    
    # Create the quadtree segmentation and compute a Delaunay triangulation of the cell centroids.
    points, heights, tri, nodes, dem_ds, new_transform = create_quadtree_3d_mesh(
        tif_path, target_shape, std_threshold
    )
    
    # Visualize the result in 3D.
    plot_quadtree_3d_mesh(points, heights, tri, nodes, dem_ds, new_transform)

if __name__ == '__main__':
    main()
