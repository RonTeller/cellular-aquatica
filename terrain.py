"""Procedural terrain generation using Perlin noise."""

import numpy as np
from noise import pnoise1, pnoise2
from materials import Material

def generate_terrain(width: int, height: int, seed: int = None) -> np.ndarray:
    """
    Generate procedural terrain with stone, dirt, caves, and drainage holes.

    Args:
        width: Grid width in cells
        height: Grid height in cells
        seed: Random seed for reproducible generation

    Returns:
        2D numpy array of material types (uint8)
    """
    if seed is not None:
        np.random.seed(seed)
        base_offset = seed
    else:
        base_offset = np.random.randint(0, 10000)

    grid = np.full((height, width), Material.AIR, dtype=np.uint8)

    # Generate height map using Perlin noise
    terrain_heights = np.zeros(width)
    for x in range(width):
        # Multiple octaves for more natural terrain
        noise_val = pnoise1(x * 0.005 + base_offset, octaves=4, persistence=0.5)
        # Map noise (-1 to 1) to height range (65% to 85% from top)
        terrain_heights[x] = int(height * (0.75 + noise_val * 0.1))

    # Stone depth offset (stone starts a bit below dirt surface)
    stone_depth = 15

    # Fill terrain
    for x in range(width):
        surface_y = int(terrain_heights[x])

        for y in range(surface_y, height):
            depth = y - surface_y

            # Determine material based on depth
            if depth < stone_depth:
                grid[y, x] = Material.DIRT
            else:
                grid[y, x] = Material.STONE

    # Generate caves using 2D Perlin noise
    cave_threshold = 0.3
    for y in range(height):
        for x in range(width):
            if grid[y, x] != Material.AIR:
                # Only create caves in stone layer
                surface_y = int(terrain_heights[x])
                if y > surface_y + stone_depth // 2:
                    cave_noise = pnoise2(
                        x * 0.03 + base_offset,
                        y * 0.03 + base_offset,
                        octaves=2
                    )
                    if cave_noise > cave_threshold:
                        grid[y, x] = Material.AIR

    # Add vertical shafts/pockets from surface
    num_pockets = np.random.randint(3, 7)
    for _ in range(num_pockets):
        # Random x position
        pocket_x = np.random.randint(width // 8, width * 7 // 8)
        surface_y = int(terrain_heights[pocket_x])

        # Pocket width and depth
        pocket_width = np.random.randint(3, 8)
        pocket_depth = np.random.randint(height // 6, height // 3)

        # Carve pocket from surface downward
        for dy in range(pocket_depth):
            y = surface_y + dy
            if y >= height:
                break

            # Vary width as we go down (can widen or narrow)
            width_var = int(np.sin(dy * 0.2) * 2)
            current_width = max(2, pocket_width + width_var)

            for dx in range(-current_width // 2, current_width // 2 + 1):
                x = pocket_x + dx
                if 0 <= x < width and y < height:
                    grid[y, x] = Material.AIR

    # Add drainage holes at bottom (allow water/sand to fall off screen)
    num_drains = np.random.randint(2, 5)
    for _ in range(num_drains):
        drain_x = np.random.randint(width // 6, width * 5 // 6)
        drain_width = np.random.randint(4, 10)

        # Carve from bottom upward
        drain_height = np.random.randint(10, 25)
        for y in range(height - 1, max(0, height - drain_height), -1):
            for dx in range(-drain_width // 2, drain_width // 2 + 1):
                x = drain_x + dx
                if 0 <= x < width:
                    grid[y, x] = Material.AIR

    return grid
