"""Material definitions and properties for the terrain simulator."""

from enum import IntEnum
import numpy as np

class Material(IntEnum):
    """Material types stored as uint8 in the simulation grid."""
    AIR = 0
    STONE = 1
    DIRT = 2
    WATER = 3
    METAL = 4
    FISH = 5  # Fish body (rendered on top of water)
    SKELETON = 6  # Dead fish skeleton (sinks and decays)
    STEAM = 7  # Evaporated water (rises and dissipates)

# Base colors for each material (RGB tuples) - Noita-inspired darker palette
MATERIAL_COLORS = {
    Material.AIR: (20, 24, 42),        # Dark night sky
    Material.STONE: (60, 55, 65),       # Dark purple-gray stone
    Material.DIRT: (90, 60, 35),        # Rich brown earth
    Material.WATER: (30, 90, 180),      # Deep blue water
    Material.METAL: (160, 165, 180),    # Silvery metal
    Material.FISH: (255, 140, 0),       # Orange fish
    Material.SKELETON: (140, 140, 130), # Grey bone color
    Material.STEAM: (180, 185, 195),    # Light grey steam
}

# Color variation ranges for each material (adds visual noise)
MATERIAL_VARIATION = {
    Material.AIR: 5,
    Material.STONE: 15,
    Material.DIRT: 20,
    Material.WATER: 25,
    Material.METAL: 20,
    Material.FISH: 30,
    Material.SKELETON: 15,
    Material.STEAM: 10,
}

def create_color_lookup() -> np.ndarray:
    """Create a lookup table for fast material-to-color conversion."""
    lookup = np.zeros((256, 3), dtype=np.uint8)
    for material, color in MATERIAL_COLORS.items():
        lookup[material] = color
    return lookup

COLOR_LOOKUP = create_color_lookup()

def get_varied_colors(grid: np.ndarray, variation_map: np.ndarray = None) -> np.ndarray:
    """
    Get colors for grid with per-pixel variation for a more organic look.

    Args:
        grid: 2D material grid
        variation_map: Pre-computed noise map for consistent variation

    Returns:
        3D array of RGB colors (height, width, 3)
    """
    height, width = grid.shape

    # Base colors from lookup
    colors = COLOR_LOOKUP[grid].astype(np.int16)

    # Generate variation if not provided
    if variation_map is None:
        variation_map = np.random.randint(-128, 128, (height, width), dtype=np.int16)

    # Apply material-specific variation
    for material in Material:
        mask = grid == material
        var_amount = MATERIAL_VARIATION.get(material, 0)
        if var_amount > 0:
            # Scale the variation map to material's range
            scaled_var = (variation_map * var_amount // 128)
            for c in range(3):
                colors[:, :, c] = np.where(
                    mask,
                    np.clip(colors[:, :, c] + scaled_var, 0, 255),
                    colors[:, :, c]
                )

    return colors.astype(np.uint8)
