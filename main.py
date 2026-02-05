"""
2D Terrain Simulator - Main Entry Point

A cellular automata-based terrain simulator with water physics,
procedural terrain generation, and rain simulation.
Inspired by Noita's visual style.
"""

import sys
import numpy as np
import pygame

from materials import Material
from terrain import generate_terrain
from simulation import Simulation
from renderer import Renderer

# Window configuration
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800
WINDOW_TITLE = "Cellular Aquatica"

# Simulation resolution (pixels per cell - higher = fewer cells = faster)
CELL_SIZE = 4  # Each cell is 4x4 pixels
SIM_WIDTH = WINDOW_WIDTH // CELL_SIZE   # 320 cells
SIM_HEIGHT = WINDOW_HEIGHT // CELL_SIZE  # 200 cells

# Simulation settings
TARGET_FPS = 60
DEFAULT_RAIN_INTENSITY = 1
MAX_RAIN_INTENSITY = 50
DEFAULT_SIM_STEPS = 1  # Simulation steps per frame
MAX_SIM_STEPS = 20
MIN_SIM_STEPS = 1

def main():
    """Main entry point for the terrain simulator."""
    # Initialize Pygame
    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    # Generate initial terrain at simulation resolution
    print(f"Generating terrain ({SIM_WIDTH}x{SIM_HEIGHT} cells, {CELL_SIZE}x scale)...")
    grid = generate_terrain(SIM_WIDTH, SIM_HEIGHT)

    # Initialize simulation and renderer
    simulation = Simulation(grid)
    renderer = Renderer(screen, CELL_SIZE)

    # State variables
    running = True
    paused = False
    rain_intensity = DEFAULT_RAIN_INTENSITY
    sim_steps = DEFAULT_SIM_STEPS
    mouse_held = False
    mouse_button = None

    print("Simulation started. Press SPACE to pause, R to regenerate terrain.")
    print(f"Simulation speed: {sim_steps}x (use [ and ] to adjust)")

    # Main game loop
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:
                    # Regenerate terrain with new seed
                    seed = np.random.randint(0, 100000)
                    grid = generate_terrain(SIM_WIDTH, SIM_HEIGHT, seed)
                    simulation = Simulation(grid)
                    print(f"Terrain regenerated (seed: {seed})")

                elif event.key == pygame.K_UP:
                    rain_intensity = min(rain_intensity + 1, MAX_RAIN_INTENSITY)

                elif event.key == pygame.K_DOWN:
                    rain_intensity = max(rain_intensity - 1, 0)

                elif event.key == pygame.K_RIGHTBRACKET:
                    sim_steps = min(sim_steps + 1, MAX_SIM_STEPS)
                    print(f"Simulation speed: {sim_steps}x")

                elif event.key == pygame.K_LEFTBRACKET:
                    sim_steps = max(sim_steps - 1, MIN_SIM_STEPS)
                    print(f"Simulation speed: {sim_steps}x")

                elif event.key == pygame.K_m:
                    simulation.drop_metal_object()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Check if click is on settings panel
                mx, my = pygame.mouse.get_pos()
                key, delta = renderer.handle_click((mx, my))

                if key is not None:
                    # Handle settings change
                    if key == 'spawn':
                        simulation.fish_spawn_chance = max(0, min(0.1,
                            simulation.fish_spawn_chance + delta))
                    elif key == 'death':
                        simulation.fish_death_chance = max(0, min(0.01,
                            simulation.fish_death_chance + delta))
                    elif key == 'rain':
                        rain_intensity = max(0, min(MAX_RAIN_INTENSITY,
                            int(rain_intensity + delta)))
                    elif key == 'evap':
                        simulation.evaporation_rate = max(0, min(0.1,
                            simulation.evaporation_rate + delta))
                    elif key == 'speed':
                        sim_steps = max(MIN_SIM_STEPS, min(MAX_SIM_STEPS,
                            int(sim_steps + delta)))
                    elif key == 'wind':
                        simulation.wind_speed = max(-1.0, min(1.0,
                            simulation.wind_speed + delta))
                    elif key == 'dry_thresh':
                        # Convert percentage to decimal (UI shows %, simulation uses 0-1)
                        new_pct = max(10, min(90, simulation.rain_threshold * 100 + delta))
                        simulation.rain_threshold = new_pct / 100
                    elif key == 'rain_thresh':
                        # Convert percentage to decimal
                        new_pct = max(5, min(50, simulation.dry_threshold * 100 + delta))
                        simulation.dry_threshold = new_pct / 100
                else:
                    # Normal mouse interaction
                    mouse_held = True
                    mouse_button = event.button

            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_held = False
                mouse_button = None

        # Handle mouse interaction (convert screen coords to sim coords)
        if mouse_held:
            mx, my = pygame.mouse.get_pos()
            # Don't interact with simulation if clicking on settings panel area
            if mx < WINDOW_WIDTH - 180:
                sim_x, sim_y = mx // CELL_SIZE, my // CELL_SIZE
                if 0 <= sim_x < SIM_WIDTH and 0 <= sim_y < SIM_HEIGHT:
                    if mouse_button == 1:  # Left click - add water
                        simulation.add_material(sim_x, sim_y, Material.WATER, radius=2)
                    elif mouse_button == 3:  # Right click - remove material
                        simulation.remove_material(sim_x, sim_y, radius=3)

        # Update simulation (multiple steps per frame for faster physics)
        if not paused:
            # Add rain only during rain season
            if rain_intensity > 0 and simulation.season == 'rain':
                simulation.add_rain(rain_intensity)

            # Run multiple simulation steps for faster gravity
            for _ in range(sim_steps):
                simulation.update()

        # Render
        renderer.render(simulation.grid)

        # Calculate FPS
        fps = clock.get_fps()
        renderer.render_ui(
            fps, rain_intensity, paused, sim_steps,
            fish_count=len(simulation.fish),
            spawn_chance=simulation.fish_spawn_chance,
            death_chance=simulation.fish_death_chance,
            evap_rate=simulation.evaporation_rate,
            season=simulation.season,
            water_ratio=simulation.get_water_ratio(),
            wind_speed=simulation.wind_speed,
            dry_threshold=simulation.rain_threshold * 100,
            rain_threshold=simulation.dry_threshold * 100
        )

        # Update display
        pygame.display.flip()

        # Cap frame rate
        clock.tick(TARGET_FPS)

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
