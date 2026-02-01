"""Pygame rendering for the terrain simulation."""

import numpy as np
import pygame
from materials import COLOR_LOOKUP, MATERIAL_VARIATION, Material

class Renderer:
    """Handles rendering the simulation grid to a Pygame surface with scaling."""

    def __init__(self, screen: pygame.Surface, cell_size: int = 1):
        """
        Initialize the renderer.

        Args:
            screen: Pygame surface to render to
            cell_size: Pixels per simulation cell (for scaling)
        """
        self.screen = screen
        self.screen_width = screen.get_width()
        self.screen_height = screen.get_height()
        self.cell_size = cell_size

        # Simulation dimensions
        self.sim_width = self.screen_width // cell_size
        self.sim_height = self.screen_height // cell_size

        # Pre-generate static noise map for consistent variation
        self.noise_map = np.random.randint(-128, 128, (self.sim_height, self.sim_width), dtype=np.int16)

        # Water shimmer animation
        self.water_time = 0

        # Create a surface at simulation resolution for fast rendering
        self.sim_surface = pygame.Surface((self.sim_width, self.sim_height))

    def render(self, grid: np.ndarray) -> None:
        """
        Render the simulation grid to the screen with Noita-style visuals.

        Args:
            grid: 2D numpy array of material types
        """
        # Base colors from lookup
        colors = COLOR_LOOKUP[grid].astype(np.int16)

        # Apply static variation to solid materials (stone, dirt, metal)
        for material in [Material.STONE, Material.DIRT, Material.METAL]:
            mask = grid == material
            var_amount = MATERIAL_VARIATION[material]
            scaled_var = (self.noise_map * var_amount // 128)
            for c in range(3):
                colors[:, :, c] = np.where(
                    mask,
                    np.clip(colors[:, :, c] + scaled_var, 0, 255),
                    colors[:, :, c]
                )

        # Animated water shimmer
        self.water_time += 0.15
        water_mask = grid == Material.WATER
        if np.any(water_mask):
            # Create shimmer using sin wave + noise
            y_coords, x_coords = np.where(water_mask)
            if len(y_coords) > 0:
                shimmer = np.sin(x_coords * 0.1 + y_coords * 0.05 + self.water_time) * 20
                shimmer += self.noise_map[water_mask] * 0.1

                # Apply shimmer to blue and green channels
                water_colors = colors[water_mask]
                water_colors[:, 1] = np.clip(water_colors[:, 1] + shimmer * 0.5, 0, 255)
                water_colors[:, 2] = np.clip(water_colors[:, 2] + shimmer, 0, 255)
                colors[water_mask] = water_colors

        # Add slight ambient variation to air
        air_mask = grid == Material.AIR
        air_var = self.noise_map * 3 // 128
        for c in range(3):
            colors[:, :, c] = np.where(
                air_mask,
                np.clip(colors[:, :, c] + air_var, 0, 255),
                colors[:, :, c]
            )

        # Convert to uint8 and transpose for pygame (expects width, height, 3)
        color_array = np.transpose(colors.astype(np.uint8), (1, 0, 2))

        # Blit to simulation surface
        pygame.surfarray.blit_array(self.sim_surface, color_array)

        # Scale up to screen size
        if self.cell_size > 1:
            pygame.transform.scale(self.sim_surface, (self.screen_width, self.screen_height), self.screen)
        else:
            self.screen.blit(self.sim_surface, (0, 0))

    def render_ui(self, fps: float, rain_intensity: int, paused: bool, sim_steps: int) -> None:
        """
        Render UI overlay with stats.

        Args:
            fps: Current frames per second
            rain_intensity: Current rain intensity setting
            paused: Whether simulation is paused
            sim_steps: Simulation steps per frame
        """
        font = pygame.font.Font(None, 24)

        # Semi-transparent background for UI
        ui_surface = pygame.Surface((140, 95), pygame.SRCALPHA)
        ui_surface.fill((0, 0, 0, 180))
        self.screen.blit(ui_surface, (5, 5))

        # Stats text
        texts = [
            f"FPS: {fps:.0f}",
            f"Rain: {rain_intensity}",
            f"Speed: {sim_steps}x",
        ]

        if paused:
            texts.append("PAUSED")

        for i, text in enumerate(texts):
            color = (255, 255, 0) if text == "PAUSED" else (255, 255, 255)
            text_surface = font.render(text, True, color)
            self.screen.blit(text_surface, (10, 10 + i * 20))

        # Controls help (bottom left)
        controls = [
            "SPACE: Pause",
            "R: Regenerate",
            "UP/DOWN: Rain",
            "[/]: Sim speed",
            "M: Drop metal",
            "Left click: Water",
            "Right click: Dig",
            "ESC: Quit"
        ]

        ctrl_surface = pygame.Surface((130, len(controls) * 18 + 10), pygame.SRCALPHA)
        ctrl_surface.fill((0, 0, 0, 150))
        y_offset = self.screen_height - len(controls) * 18 - 15
        self.screen.blit(ctrl_surface, (5, y_offset))

        small_font = pygame.font.Font(None, 20)
        for i, text in enumerate(controls):
            ctrl_text = small_font.render(text, True, (180, 180, 180))
            self.screen.blit(ctrl_text, (10, y_offset + 5 + i * 18))
