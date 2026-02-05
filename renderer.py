"""Pygame rendering for the terrain simulation."""

import numpy as np
import pygame
from materials import COLOR_LOOKUP, MATERIAL_VARIATION, Material


class SettingsPanel:
    """A panel for displaying and modifying simulation settings."""

    def __init__(self, x: int, y: int, width: int):
        self.x = x
        self.y = y
        self.width = width
        self.font = None
        self.small_font = None
        self.settings = []  # List of (name, key, value, min, max, step, format)
        self.button_rects = {}  # Maps (setting_key, 'plus'/'minus') to rect

    def init_fonts(self):
        """Initialize fonts (must be called after pygame.init)."""
        self.font = pygame.font.Font(None, 22)
        self.small_font = pygame.font.Font(None, 18)

    def add_setting(self, name: str, key: str, value: float, min_val: float,
                    max_val: float, step: float, fmt: str = ".3f"):
        """Add a setting to the panel."""
        self.settings.append({
            'name': name,
            'key': key,
            'value': value,
            'min': min_val,
            'max': max_val,
            'step': step,
            'format': fmt
        })

    def update_value(self, key: str, value: float):
        """Update a setting's value."""
        for setting in self.settings:
            if setting['key'] == key:
                setting['value'] = value
                break

    def get_value(self, key: str) -> float:
        """Get a setting's value."""
        for setting in self.settings:
            if setting['key'] == key:
                return setting['value']
        return 0

    def handle_click(self, pos: tuple) -> tuple:
        """
        Handle mouse click. Returns (key, delta) if a button was clicked,
        or (None, 0) otherwise.
        """
        for (key, btn_type), rect in self.button_rects.items():
            if rect.collidepoint(pos):
                for setting in self.settings:
                    if setting['key'] == key:
                        delta = setting['step'] if btn_type == 'plus' else -setting['step']
                        return (key, delta)
        return (None, 0)

    def render(self, screen: pygame.Surface, fish_count: int = 0):
        """Render the settings panel."""
        if self.font is None:
            self.init_fonts()

        # Panel background
        panel_height = len(self.settings) * 50 + 60
        panel_surface = pygame.Surface((self.width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((20, 20, 30, 220))
        screen.blit(panel_surface, (self.x, self.y))

        # Title
        title = self.font.render("Settings", True, (255, 255, 255))
        screen.blit(title, (self.x + 10, self.y + 8))

        # Fish count
        fish_text = self.small_font.render(f"Fish: {fish_count}", True, (255, 200, 100))
        screen.blit(fish_text, (self.x + 100, self.y + 10))

        # Settings
        self.button_rects.clear()
        y_pos = self.y + 35

        for setting in self.settings:
            # Setting name
            name_text = self.small_font.render(setting['name'], True, (200, 200, 200))
            screen.blit(name_text, (self.x + 10, y_pos))

            # Value
            fmt = setting['format']
            val_str = f"{setting['value']:{fmt}}"
            val_text = self.font.render(val_str, True, (255, 255, 255))
            screen.blit(val_text, (self.x + 10, y_pos + 16))

            # Minus button
            minus_rect = pygame.Rect(self.x + self.width - 70, y_pos + 12, 28, 22)
            pygame.draw.rect(screen, (80, 80, 100), minus_rect)
            pygame.draw.rect(screen, (120, 120, 140), minus_rect, 1)
            minus_text = self.font.render("-", True, (255, 255, 255))
            screen.blit(minus_text, (minus_rect.x + 10, minus_rect.y + 2))
            self.button_rects[(setting['key'], 'minus')] = minus_rect

            # Plus button
            plus_rect = pygame.Rect(self.x + self.width - 38, y_pos + 12, 28, 22)
            pygame.draw.rect(screen, (80, 80, 100), plus_rect)
            pygame.draw.rect(screen, (120, 120, 140), plus_rect, 1)
            plus_text = self.font.render("+", True, (255, 255, 255))
            screen.blit(plus_text, (plus_rect.x + 8, plus_rect.y + 2))
            self.button_rects[(setting['key'], 'plus')] = plus_rect

            y_pos += 50


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

        # Settings panel (right side)
        self.settings_panel = SettingsPanel(
            self.screen_width - 180, 10, 170
        )
        self._init_settings_panel()

    def _init_settings_panel(self):
        """Initialize settings panel with default values."""
        self.settings_panel.add_setting("Fish Spawn Rate", "spawn", 0.020, 0.0, 0.1, 0.005, ".3f")
        self.settings_panel.add_setting("Fish Death Rate", "death", 0.0005, 0.0, 0.01, 0.0001, ".4f")
        self.settings_panel.add_setting("Rain Intensity", "rain", 1, 0, 50, 1, ".0f")
        self.settings_panel.add_setting("Evaporation Rate", "evap", 0.01, 0.0, 0.1, 0.005, ".3f")
        self.settings_panel.add_setting("Sim Speed", "speed", 1, 1, 20, 1, ".0f")
        self.settings_panel.add_setting("Wind Speed", "wind", 0.0, -1.0, 1.0, 0.1, ".1f")
        self.settings_panel.add_setting("Dry Season %", "dry_thresh", 50, 10, 90, 5, ".0f")
        self.settings_panel.add_setting("Rain Season %", "rain_thresh", 20, 5, 50, 5, ".0f")

    def handle_click(self, pos: tuple) -> tuple:
        """Handle mouse click for settings panel."""
        return self.settings_panel.handle_click(pos)

    def update_setting(self, key: str, value: float):
        """Update a setting value in the panel."""
        self.settings_panel.update_value(key, value)

    def get_setting(self, key: str) -> float:
        """Get a setting value from the panel."""
        return self.settings_panel.get_value(key)

    def render(self, grid: np.ndarray) -> None:
        """
        Render the simulation grid to the screen with Noita-style visuals.

        Args:
            grid: 2D numpy array of material types
        """
        # Base colors from lookup
        colors = COLOR_LOOKUP[grid].astype(np.int16)

        # Pre-compute variation lookup (material index -> variation amount)
        var_lookup = np.zeros(256, dtype=np.int16)
        for mat in [Material.STONE, Material.DIRT, Material.METAL, Material.FISH, Material.SKELETON]:
            var_lookup[mat] = MATERIAL_VARIATION[mat]
        var_lookup[Material.AIR] = 3  # Slight air variation

        # Get variation amount for each cell based on material type
        var_amounts = var_lookup[grid]

        # Apply variation in one vectorized operation
        scaled_var = (self.noise_map * var_amounts // 128).astype(np.int16)
        colors[:, :, 0] = np.clip(colors[:, :, 0] + scaled_var, 0, 255)
        colors[:, :, 1] = np.clip(colors[:, :, 1] + scaled_var, 0, 255)
        colors[:, :, 2] = np.clip(colors[:, :, 2] + scaled_var, 0, 255)

        # Animated water shimmer (optimized - only compute for water cells)
        self.water_time += 0.15
        water_mask = grid == Material.WATER
        water_count = np.sum(water_mask)
        if water_count > 0:
            # Create shimmer using sin wave + noise
            y_coords, x_coords = np.where(water_mask)
            shimmer = np.sin(x_coords * 0.1 + y_coords * 0.05 + self.water_time) * 20
            shimmer += self.noise_map[water_mask] * 0.1

            # Apply shimmer to blue and green channels
            water_colors = colors[water_mask]
            water_colors[:, 1] = np.clip(water_colors[:, 1] + shimmer * 0.5, 0, 255)
            water_colors[:, 2] = np.clip(water_colors[:, 2] + shimmer, 0, 255)
            colors[water_mask] = water_colors

        # Convert to uint8 and transpose for pygame (expects width, height, 3)
        color_array = np.transpose(colors.astype(np.uint8), (1, 0, 2))

        # Blit to simulation surface
        pygame.surfarray.blit_array(self.sim_surface, color_array)

        # Scale up to screen size
        if self.cell_size > 1:
            pygame.transform.scale(self.sim_surface, (self.screen_width, self.screen_height), self.screen)
        else:
            self.screen.blit(self.sim_surface, (0, 0))

    def render_ui(self, fps: float, rain_intensity: int, paused: bool, sim_steps: int,
                  fish_count: int = 0, spawn_chance: float = 0.02, death_chance: float = 0.0005,
                  evap_rate: float = 0.01, season: str = 'rain', water_ratio: float = 0.0,
                  wind_speed: float = 0.0, dry_threshold: float = 50, rain_threshold: float = 20) -> None:
        """
        Render UI overlay with stats and settings panel.
        """
        font = pygame.font.Font(None, 24)

        # Semi-transparent background for stats (top left)
        ui_surface = pygame.Surface((150, 95), pygame.SRCALPHA)
        ui_surface.fill((0, 0, 0, 180))
        self.screen.blit(ui_surface, (5, 5))

        # Season indicator with color
        season_color = (100, 150, 255) if season == 'rain' else (255, 200, 100)
        season_text = f"Season: {season.upper()}"
        season_surface = font.render(season_text, True, season_color)
        self.screen.blit(season_surface, (10, 10))

        # Water level
        water_pct = water_ratio * 100
        water_text = font.render(f"Water: {water_pct:.1f}%", True, (150, 200, 255))
        self.screen.blit(water_text, (10, 30))

        # FPS
        fps_text = font.render(f"FPS: {fps:.0f}", True, (255, 255, 255))
        self.screen.blit(fps_text, (10, 50))

        # Paused indicator
        if paused:
            pause_text = font.render("PAUSED", True, (255, 255, 0))
            self.screen.blit(pause_text, (10, 70))

        # Controls help (bottom left)
        controls = [
            "SPACE: Pause",
            "R: Regenerate",
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

        # Update settings panel values
        self.settings_panel.update_value("spawn", spawn_chance)
        self.settings_panel.update_value("death", death_chance)
        self.settings_panel.update_value("rain", rain_intensity)
        self.settings_panel.update_value("evap", evap_rate)
        self.settings_panel.update_value("speed", sim_steps)
        self.settings_panel.update_value("wind", wind_speed)
        self.settings_panel.update_value("dry_thresh", dry_threshold)
        self.settings_panel.update_value("rain_thresh", rain_threshold)

        # Render settings panel
        self.settings_panel.render(self.screen, fish_count)
