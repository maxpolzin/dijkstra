from ursina import *
from ursina.shaders import lit_with_shadows_shader
from noise import pnoise2
import numpy as np
import random

app = Ursina()

# ------------------------
# Initialize Shaders
# ------------------------
random.seed(0)
Entity.default_shader = lit_with_shadows_shader

# ------------------------
# Terrain Parameters
# ------------------------
terrain_width = 100     # Number of vertices along the X-axis
terrain_length = 100    # Number of vertices along the Z-axis
terrain_scale = 20      # Controls the zoom level of the noise
terrain_height = 15     # Maximum height of the terrain
octaves = 6             # Number of noise layers
persistence = 0.5       # Amplitude of each octave
lacunarity = 2.0        # Frequency of each octave
seed = 42               # Seed for randomness
water_level = 5         # Height at which water is placed

# ------------------------
# Generate Terrain Mesh
# ------------------------
def generate_terrain(width, length, scale, octaves, persistence, lacunarity, seed, max_height, water_level):
    vertices = []
    colors = []
    uvs = []
    indices = []

    height_map = np.zeros((width, length))

    # Generate height map using Perlin noise
    for z in range(length):
        for x in range(width):
            nx = x / scale
            nz = z / scale
            noise_val = pnoise2(nx, nz,
                                octaves=octaves,
                                persistence=persistence,
                                lacunarity=lacunarity,
                                repeatx=1024,
                                repeaty=1024,
                                base=seed)
            noise_normalized = (noise_val + 0.5)  # Normalize to [0,1]
            height = noise_normalized * max_height
            height_map[x][z] = height

            # Define vertex position
            vertices.append(Vec3(x, height, z))

            # Assign colors based on height
            if height < water_level:
                vertex_color = color.blue  # Water
            elif height < water_level + (max_height - water_level) * 0.3:
                vertex_color = color.gray  # Dirt (Replaced Beige with Gray)
            elif height < water_level + (max_height - water_level) * 0.6:
                vertex_color = color.green  # Grass
            else:
                vertex_color = color.white  # Snow/Rock
            colors.append(vertex_color)

            # Simple UVs
            uvs.append((x / width, z / length))

    # Define triangles with corrected indexing (Row-Major Order)
    for z in range(length - 1):
        for x in range(width - 1):
            top_left = z * width + x
            top_right = z * width + (x + 1)
            bottom_left = (z + 1) * width + x
            bottom_right = (z + 1) * width + (x + 1)

            # First triangle (top_left, bottom_left, top_right)
            indices.append(top_left)
            indices.append(bottom_left)
            indices.append(top_right)

            # Second triangle (top_right, bottom_left, bottom_right)
            indices.append(top_right)
            indices.append(bottom_left)
            indices.append(bottom_right)

    # Create mesh
    terrain_mesh = Mesh(vertices=vertices, triangles=indices, uvs=uvs, colors=colors, mode='triangle')
    return terrain_mesh

# Generate terrain mesh
terrain_mesh = generate_terrain(
    width=terrain_width,
    length=terrain_length,
    scale=terrain_scale,
    octaves=octaves,
    persistence=persistence,
    lacunarity=lacunarity,
    seed=seed,
    max_height=terrain_height,
    water_level=water_level
)

# Create terrain entity
terrain = Entity(model=terrain_mesh, shader=lit_with_shadows_shader, collider='mesh')

# ------------------------
# Free-Fly First-Person Controller
# ------------------------
class FreeFlyController(Entity):
    def __init__(self, **kwargs):
        super().__init__()
        self.camera_pivot = Entity(parent=self, y=1.8)
        camera.parent = self.camera_pivot
        camera.position = (0, 0, 0)
        camera.rotation = (0, 0, 0)
        self.speed = 20  # Increased speed for flying
        self.mouse_sensitivity = Vec2(40, 40)
        self.velocity = Vec3(0, 0, 0)
        self.gravity = 0  # Disable gravity

    def update(self):
        # Mouse look
        self.rotation_y += mouse.velocity[0] * self.mouse_sensitivity.x
        self.camera_pivot.rotation_x -= mouse.velocity[1] * self.mouse_sensitivity.y
        self.camera_pivot.rotation_x = clamp(self.camera_pivot.rotation_x, -90, 90)

        # Movement
        direction = Vec3(
            self.forward * (held_keys['w'] - held_keys['s']) +
            self.right * (held_keys['d'] - held_keys['a']) +
            self.up * (held_keys['space'] - held_keys['left shift'])
        ).normalized()

        self.position += direction * self.speed * time.dt

# Instantiate Free-Fly Controller
player = FreeFlyController()
player.position = (terrain_width / 2, terrain_height + 10, terrain_length / 2)


# ------------------------
# Pause Functionality
# ------------------------
def pause_input(key):
    if key == 'tab':    # Press Tab to toggle edit/play mode
        editor_camera.enabled = not editor_camera.enabled

        player.visible_self = editor_camera.enabled
        player.cursor.enabled = not editor_camera.enabled
        # gun.enabled = not editor_camera.enabled  # Removed gun, so this line is no longer needed
        mouse.locked = not editor_camera.enabled
        editor_camera.position = player.position

        application.paused = editor_camera.enabled

pause_handler = Entity(ignore_paused=True, input=pause_input)

# ------------------------
# Lighting and Sky
# ------------------------
sun = DirectionalLight()
sun.look_at(Vec3(1, -1, -1))
Sky()

# ------------------------
# Run the Application
# ------------------------
app.run()
