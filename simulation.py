"""Cellular automata simulation engine for terrain and fluid dynamics."""

import numpy as np
from materials import Material


class Fish:
    """A fish that swims in water with natural movement."""

    BASE_SPEED = 0.6  # Base swimming speed
    _next_id = 0  # Class-level ID counter

    def __init__(self, x: float, y: float):
        # Unique ID for tracking structural integrity
        self.id = Fish._next_id
        Fish._next_id += 1

        self.x = float(x)
        self.y = float(y)

        # Swimming parameters
        self.direction = np.random.choice([-1, 1])  # -1 = left, 1 = right
        self.randomize_speed()  # Set initial random speed
        self.vx = self.direction * self.swim_speed
        self.vy = 0.0

        # Oscillation for natural swimming motion
        self.swim_phase = np.random.uniform(0, 2 * np.pi)
        self.swim_frequency = np.random.uniform(0.08, 0.15)
        self.vertical_tendency = np.random.uniform(-0.02, 0.02)  # Slight up/down preference

        # Size as numeric value (affects shape and collision outcomes)
        # All fish start small (size_value = 1.0)
        self.size_value = 1.0

        self.alive = True
        self.age = 0
        self.max_age = np.random.randint(800, 2000)  # Lifespan in frames
        self.turn_cooldown = 0  # Prevent rapid direction changes

        # Random color for this fish (vibrant, saturated colors)
        self.color = self._generate_random_color()
        # Eye color (black or white)
        self.eye_color = (0, 0, 0) if np.random.random() < 0.7 else (255, 255, 255)

        # Cached cells for performance (invalidated when position/direction changes)
        self._cached_cells = None
        self._cached_pos = None
        self._cached_dir = None
        self._cached_shape = None

    def _generate_random_color(self) -> tuple:
        """Generate a random vibrant fish color."""
        # Use HSV-like approach for vibrant colors
        # Hue: full range, Saturation: high, Value: high
        hue = np.random.random()

        # Convert hue to RGB (simplified HSV to RGB with S=0.8, V=1.0)
        h = hue * 6.0
        c = 0.8  # Chroma (saturation * value)
        x = c * (1 - abs(h % 2 - 1))
        m = 0.2  # To brighten

        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        # Add some random variation
        r = min(255, int((r + m) * 255 + np.random.randint(-20, 20)))
        g = min(255, int((g + m) * 255 + np.random.randint(-20, 20)))
        b = min(255, int((b + m) * 255 + np.random.randint(-20, 20)))

        return (max(0, r), max(0, g), max(0, b))

    @property
    def size(self) -> str:
        """Get size category based on size_value."""
        if self.size_value < 1.5:
            return 'small'
        elif self.size_value < 2.5:
            return 'medium'
        else:
            return 'large'

    def grow_from_eating(self, eaten_pixel_count: int) -> None:
        """Grow the fish by half the eaten fish's pixels (rounded down)."""
        # Each 3 pixels ≈ 0.5 size_value unit (slower growth for bigger fish)
        pixels_gained = eaten_pixel_count // 3
        self.size_value += pixels_gained * 0.5
        # Cap max size
        self.size_value = min(self.size_value, 5.0)
        # Clear cached shape
        self._cached_shape = None

    def randomize_speed(self) -> None:
        """Randomize swim speed between 0.5x and 2x base speed."""
        multiplier = np.random.uniform(0.5, 2.0)
        self.swim_speed = Fish.BASE_SPEED * multiplier

    def get_shape(self) -> list:
        """Get fish shape as list of (dx, dy, is_eye) tuples from head position.

        Returns list of (dx, dy, is_eye) where is_eye=True for eye cells.
        """
        if self._cached_shape is not None and self._cached_dir == self.direction:
            return self._cached_shape

        d = self.direction

        if self.size == 'small':
            # Small fish: 8 pixels with eye
            #   ●o      <- eye and top of head
            #  oooo<    <- body with pointed tail
            shape = [
                (0, -1, True),     # Eye
                (-d, -1, False),   # Top of head
                (0, 0, False),     # Nose/head
                (-d, 0, False),    # Body 1
                (-d * 2, 0, False),  # Body 2
                (-d * 3, 0, False),  # Body 3
                (-d * 4, 0, False),  # Tail base
                (-d * 5, -1, False), # Tail tip (angled up)
            ]
        elif self.size == 'medium':
            # Medium fish: 14 pixels with eye and fins
            #    ●oo       <- eye and dorsal area
            #   ooooo<     <- main body
            #    oo        <- ventral/belly
            shape = [
                (-d, -1, True),      # Eye
                (-d * 2, -1, False), # Dorsal 1
                (-d * 3, -1, False), # Dorsal 2 (fin)
                (0, 0, False),       # Nose
                (-d, 0, False),      # Head
                (-d * 2, 0, False),  # Body 1
                (-d * 3, 0, False),  # Body 2
                (-d * 4, 0, False),  # Body 3
                (-d * 5, 0, False),  # Tail base
                (-d * 6, -1, False), # Tail top
                (-d * 6, 1, False),  # Tail bottom
                (-d * 2, 1, False),  # Belly 1
                (-d * 3, 1, False),  # Belly 2 / ventral fin
            ]
        else:
            # Large fish: 22 pixels with eye, fins, and forked tail
            #      o         <- dorsal fin tip
            #    ●ooo        <- eye and upper body
            #   oooooo<      <- main body with tail
            #    oooo   <    <- lower body and tail fork
            #      o         <- ventral fin
            shape = [
                (-d * 3, -2, False),  # Dorsal fin tip
                (-d, -1, True),       # Eye
                (-d * 2, -1, False),  # Upper body 1
                (-d * 3, -1, False),  # Upper body 2 (dorsal base)
                (-d * 4, -1, False),  # Upper body 3
                (0, 0, False),        # Nose
                (-d, 0, False),       # Head
                (-d * 2, 0, False),   # Body 1
                (-d * 3, 0, False),   # Body 2
                (-d * 4, 0, False),   # Body 3
                (-d * 5, 0, False),   # Body 4
                (-d * 6, 0, False),   # Tail base
                (-d * 7, -1, False),  # Tail fork upper
                (-d * 7, 1, False),   # Tail fork lower
                (-d * 2, 1, False),   # Lower body 1
                (-d * 3, 1, False),   # Lower body 2
                (-d * 4, 1, False),   # Lower body 3
                (-d * 5, 1, False),   # Lower body 4
                (-d * 3, 2, False),   # Ventral fin
            ]

        self._cached_shape = shape
        return shape

    def get_shape_simple(self) -> list:
        """Get fish shape as simple (dx, dy) list for collision detection."""
        return [(dx, dy) for dx, dy, _ in self.get_shape()]

    def get_center_of_mass_offset(self) -> float:
        """Get the x-offset of center of mass from head position."""
        shape = self.get_shape()
        if not shape:
            return 0.0
        total_x = sum(dx for dx, dy, _ in shape)
        return total_x / len(shape)

    def turn_with_pivot(self) -> None:
        """Turn the fish around, pivoting around center of mass."""
        # Calculate center of mass position before turning
        old_com_offset = self.get_center_of_mass_offset()
        old_com_x = self.x + old_com_offset

        # Flip direction
        self.direction *= -1
        self.vx = -self.vx
        self._cached_cells = None
        self._cached_shape = None  # Shape changes with direction

        # Calculate new center of mass offset after turning
        new_com_offset = self.get_center_of_mass_offset()

        # Adjust position so center of mass stays in same world position
        self.x = old_com_x - new_com_offset

    def get_cells(self) -> list:
        """Get world coordinates of all fish cells (cached for performance)."""
        ix, iy = int(round(self.x)), int(round(self.y))

        # Cache key includes position, direction, AND size (so cache invalidates on growth)
        if (self._cached_cells is None or
            self._cached_pos != (ix, iy) or
            self._cached_dir != self.direction or
            len(self._cached_cells) != len(self.get_shape())):
            self._cached_cells = [(ix + dx, iy + dy) for dx, dy, _ in self.get_shape()]
            self._cached_pos = (ix, iy)
            self._cached_dir = self.direction

        return self._cached_cells

    def get_cells_with_type(self) -> list:
        """Get world coordinates with cell type (for rendering with eye color)."""
        ix, iy = int(round(self.x)), int(round(self.y))
        return [(ix + dx, iy + dy, is_eye) for dx, dy, is_eye in self.get_shape()]

    def update_swimming(self) -> None:
        """Update swimming motion - natural side-to-side movement."""
        # Update swim phase for oscillation
        self.swim_phase += self.swim_frequency

        # Horizontal movement - maintain swim direction with slight speed variation
        speed_wobble = np.sin(self.swim_phase * 2) * 0.1
        self.vx = self.direction * (self.swim_speed + speed_wobble)

        # Vertical oscillation - gentle up/down wave motion
        self.vy = np.sin(self.swim_phase) * 0.15 + self.vertical_tendency

        # Occasionally change vertical tendency
        if np.random.random() < 0.01:
            self.vertical_tendency = np.random.uniform(-0.03, 0.03)

        # Decrease turn cooldown
        if self.turn_cooldown > 0:
            self.turn_cooldown -= 1

    def turn_around(self) -> None:
        """Make the fish turn and swim the other direction, pivoting around center of mass."""
        if self.turn_cooldown <= 0:
            self.turn_with_pivot()
            self.vx = self.direction * self.swim_speed
            self.turn_cooldown = 30  # Prevent turning again for 30 frames


class Skeleton:
    """A dead fish skeleton that sinks and decays."""

    def __init__(self, x: float, y: float, shape: list, direction: int):
        """
        Create a skeleton from a dead fish.

        Args:
            x, y: Position
            shape: List of (dx, dy) offsets from the fish
            direction: Fish direction (-1 or 1)
        """
        self.x = float(x)
        self.y = float(y)
        self.shape = shape
        self.direction = direction
        self.sink_speed = 0.8  # Fast sinking
        self.at_bottom = False
        self.decay_timer = 0
        self.decay_time = 180  # ~3 seconds at 60 FPS

    def get_cells(self) -> list:
        """Get world coordinates of all skeleton cells."""
        ix, iy = int(round(self.x)), int(round(self.y))
        return [(ix + dx, iy + dy) for dx, dy in self.shape]

    def update(self) -> bool:
        """
        Update skeleton position and decay.
        Returns True if skeleton should be removed.
        """
        if self.at_bottom:
            self.decay_timer += 1
            return self.decay_timer >= self.decay_time
        return False


class MetalObject:
    """A rigid metal object with physics properties."""

    def __init__(self, x: float, y: float, shape: list):
        """
        Initialize a metal object.

        Args:
            x, y: Position (float for sub-pixel precision)
            shape: List of (dx, dy) cell offsets relative to origin
        """
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0  # Velocity
        self.vy = 0.0
        self.angle = 0.0  # Rotation in radians
        self.angular_vel = 0.0  # Angular velocity
        self.base_shape = shape  # Original shape
        self.shape = list(shape)  # Current rotated shape

        # Calculate center of mass (relative to origin)
        self.com_x = sum(dx for dx, dy in shape) / len(shape)
        self.com_y = sum(dy for dx, dy in shape) / len(shape)

    def get_rotated_shape(self) -> list:
        """Get shape cells rotated around center of mass."""
        if abs(self.angle) < 0.01:
            return list(self.base_shape)

        cos_a = np.cos(self.angle)
        sin_a = np.sin(self.angle)
        rotated = []

        for dx, dy in self.base_shape:
            # Translate to COM, rotate, translate back
            rx = dx - self.com_x
            ry = dy - self.com_y
            new_x = rx * cos_a - ry * sin_a + self.com_x
            new_y = rx * sin_a + ry * cos_a + self.com_y
            rotated.append((int(round(new_x)), int(round(new_y))))

        return rotated

    def get_world_cells(self) -> list:
        """Get list of (world_x, world_y) for all cells."""
        ix, iy = int(round(self.x)), int(round(self.y))
        return [(ix + dx, iy + dy) for dx, dy in self.shape]


class Simulation:
    """
    Cellular automata simulation for water and terrain physics.

    Uses a checkerboard update pattern to avoid directional bias.
    """

    def __init__(self, grid: np.ndarray):
        """
        Initialize simulation with a material grid.

        Args:
            grid: 2D numpy array of material types (uint8)
        """
        self.grid = grid
        self.height, self.width = grid.shape
        self.frame = 0

        # Rigid metal objects with physics
        self.metal_objects = []

        # Fish (life simulation)
        self.fish = []
        self.skeletons = []  # Dead fish skeletons
        self.fish_spawn_chance = 0.02  # Base chance to spawn (multiplied by water ratio)
        self.fish_death_chance = 0.0001  # Base chance of death per frame (increases with size)
        self.fish_eat_chance = 0.2  # Chance that collision results in eating (vs just bouncing)
        self.min_water_for_life = 50  # Minimum water cells needed for fish to spawn

        # Fish ID grid - tracks which fish owns each cell (-1 = no fish)
        self.fish_grid = np.full((self.height, self.width), -1, dtype=np.int32)

        # Fish color grid - stores RGB color for each fish cell (for per-fish coloring)
        self.fish_color_grid = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Seasons system
        self.season = 'rain'  # 'rain' or 'dry'
        self.rain_threshold = 0.50  # Switch to dry season when water reaches 50%
        self.dry_threshold = 0.20  # Switch to rain season when water drops to 20%
        self.evaporation_rate = 0.01  # Chance per surface water cell to evaporate per frame
        self.total_cells = self.width * self.height

        # Cached water count (updated incrementally for performance)
        self._water_count = np.sum(grid == Material.WATER)
        self._water_count_dirty = False  # Flag for when full recount needed

        # Pre-allocated random direction array (reused each frame)
        self._rand_dirs = (np.random.randint(0, 2, (self.height, self.width)) * 2 - 1).astype(np.int8)

        # Physics constants
        self.gravity = 0.5
        self.bounce_damping = 0.5  # Energy retained after bounce
        self.friction = 0.85
        self.rotation_damping = 0.92

        # Wind system (affects steam movement)
        # Positive = wind blowing right, negative = wind blowing left
        self.wind_speed = 0.0  # Range: -1.0 to 1.0

    def update(self) -> None:
        """Perform one simulation step using checkerboard pattern."""
        self.frame += 1
        offset = self.frame % 2

        # Update rigid metal objects first
        self._update_metal_objects()

        # Update fish
        self._update_fish()

        # Update skeletons (sinking and decay)
        self._update_skeletons()

        # Try to spawn new fish
        self._try_spawn_fish()

        # Update seasons and evaporation
        self._update_seasons()

        # Gradually vary wind direction and speed
        if self.frame % 60 == 0:  # Change wind every ~1 second
            # Wind changes slowly, with occasional gusts
            wind_change = np.random.uniform(-0.15, 0.15)
            self.wind_speed = np.clip(self.wind_speed + wind_change, -1.0, 1.0)
            # Small chance of calm or gust
            if np.random.random() < 0.1:
                self.wind_speed *= np.random.choice([0.0, 1.5])
                self.wind_speed = np.clip(self.wind_speed, -1.0, 1.0)

        # Update steam (rises and drifts with wind)
        self._update_steam()

        # Shuffle random directions occasionally (every 4 frames) instead of regenerating
        if self.frame % 4 == 0:
            np.random.shuffle(self._rand_dirs.ravel())
            self._rand_dirs = self._rand_dirs.reshape((self.height, self.width))

        # Alternate x-iteration direction each frame to reduce left/right bias
        reverse_x = (self.frame // 2) % 2 == 1

        # Cache enum values as integers to avoid enum lookup overhead in hot loop
        MAT_WATER = int(Material.WATER)
        MAT_DIRT = int(Material.DIRT)
        grid = self.grid  # Local reference for speed
        rand_dirs = self._rand_dirs
        width = self.width
        height = self.height

        # Process from bottom to top for falling materials
        for y in range(height - 2, -1, -1):
            start_x = (y + offset) % 2
            if reverse_x:
                x_range = range(width - 1 - ((width - 1 - start_x) % 2), -1, -2)
            else:
                x_range = range(start_x, width, 2)

            for x in x_range:
                material = grid[y, x]
                if material == MAT_WATER:
                    self._update_water_fast(grid, x, y, int(rand_dirs[y, x]), width, height)
                elif material == MAT_DIRT:
                    self._update_dirt_fast(grid, x, y, int(rand_dirs[y, x]), width, height)

    def _update_water(self, x: int, y: int, rand_dir: int) -> None:
        """Update a water cell with 2-cell lookahead for better flow."""
        self._update_water_fast(self.grid, x, y, rand_dir, self.width, self.height)

    def _update_water_fast(self, g, x: int, y: int, rand_dir: int, w: int, h: int) -> None:
        """Optimized water update with pre-cached values."""
        # Use integer constants to avoid enum lookup overhead
        AIR = 0
        WATER = 3

        if g[y, x] != WATER:
            return

        # Fall off bottom of screen
        if y == h - 1:
            g[y, x] = AIR
            self._water_count -= 1
            return

        # 1. Try to move straight down
        if g[y + 1, x] == AIR:
            g[y + 1, x] = WATER
            g[y, x] = AIR
            return

        # 2. Try diagonal down
        for dx in (rand_dir, -rand_dir):
            nx = x + dx
            if 0 <= nx < w and g[y + 1, nx] == AIR:
                g[y + 1, nx] = WATER
                g[y, x] = AIR
                return

        # 3. Look 2 cells ahead
        for dx in (rand_dir, -rand_dir):
            nx = x + dx
            if 0 <= nx < w and g[y, nx] == AIR:
                nx2 = x + dx * 2
                can_flow_down = g[y + 1, nx] == AIR
                can_flow_down2 = (0 <= nx2 < w and g[y + 1, nx2] == AIR)
                if can_flow_down or can_flow_down2:
                    g[y, nx] = WATER
                    g[y, x] = AIR
                    return

        # 4. Simple horizontal spread
        for dx in (rand_dir, -rand_dir):
            nx = x + dx
            if 0 <= nx < w and g[y, nx] == AIR:
                g[y, nx] = WATER
                g[y, x] = AIR
                return

    def _update_dirt(self, x: int, y: int, rand_dir: int) -> None:
        """Update a dirt cell - falls down if possible."""
        self._update_dirt_fast(self.grid, x, y, rand_dir, self.width, self.height)

    def _update_dirt_fast(self, g, x: int, y: int, rand_dir: int, w: int, h: int) -> None:
        """Optimized dirt update with pre-cached values."""
        # Use integer constants to avoid enum lookup overhead
        AIR = 0
        DIRT = 2

        if g[y, x] != DIRT:
            return

        # Fall off bottom of screen
        if y == h - 1:
            g[y, x] = AIR
            return

        # 1. Try to fall straight down
        if g[y + 1, x] == AIR:
            g[y + 1, x] = DIRT
            g[y, x] = AIR
            return

        # 2. Try diagonal fall
        for dx in (rand_dir, -rand_dir):
            nx = x + dx
            if 0 <= nx < w:
                if g[y, nx] == AIR and g[y + 1, nx] == AIR:
                    g[y + 1, nx] = DIRT
                    g[y, x] = AIR
                    return

    def _update_metal_objects(self) -> None:
        """Update all rigid metal objects with physics."""
        objects_to_remove = []

        for i, obj in enumerate(self.metal_objects):
            # Clear current position from grid
            for wx, wy in obj.get_world_cells():
                if 0 <= wx < self.width and 0 <= wy < self.height:
                    if self.grid[wy, wx] == Material.METAL:
                        self.grid[wy, wx] = Material.AIR

            # Apply gravity
            obj.vy += self.gravity

            # Apply rotation damping
            obj.angular_vel *= self.rotation_damping

            # Update rotation and recalculate shape
            obj.angle += obj.angular_vel
            obj.shape = obj.get_rotated_shape()

            # Calculate new position
            new_x = obj.x + obj.vx
            new_y = obj.y + obj.vy

            # Store old position
            old_x, old_y = obj.x, obj.y
            old_cells = set(obj.get_world_cells())

            # Temporarily move object to check collisions
            obj.x, obj.y = new_x, new_y

            # Check for collisions
            collision_points = []
            fell_off = False

            for wx, wy in obj.get_world_cells():
                # Check if fell off bottom
                if wy >= self.height:
                    fell_off = True
                    break

                # Wall collisions
                if wx < 0 or wx >= self.width:
                    collision_points.append((wx, wy, 'wall'))
                    continue

                if wy < 0:
                    continue

                # Check grid collision (not with own previous position)
                if (wx, wy) not in old_cells:
                    cell = self.grid[wy, wx]
                    if cell == Material.WATER:
                        # Displace water upward
                        for check_y in range(wy - 1, -1, -1):
                            if self.grid[check_y, wx] == Material.AIR:
                                self.grid[check_y, wx] = Material.WATER
                                break
                    elif cell not in (Material.AIR, Material.WATER):
                        collision_points.append((wx, wy, 'solid'))

            if fell_off:
                objects_to_remove.append(i)
                obj.x, obj.y = old_x, old_y
                continue

            # Handle collisions
            if collision_points:
                obj.x, obj.y = old_x, old_y

                # Calculate center of mass in world coords
                world_com_x = obj.x + obj.com_x
                world_com_y = obj.y + obj.com_y

                # Average collision point
                avg_col_x = sum(p[0] for p in collision_points) / len(collision_points)
                avg_col_y = sum(p[1] for p in collision_points) / len(collision_points)

                # Vector from COM to collision
                rel_x = avg_col_x - world_com_x
                rel_y = avg_col_y - world_com_y

                # Determine collision normal
                if abs(obj.vy) > abs(obj.vx) * 0.5:
                    # Mostly vertical collision
                    normal_x, normal_y = 0, -1 if obj.vy > 0 else 1
                else:
                    # Mostly horizontal collision
                    normal_x, normal_y = -1 if obj.vx > 0 else 1, 0

                # Calculate torque from off-center collision
                torque = (rel_x * normal_y - rel_y * normal_x) * 0.008
                obj.angular_vel += torque * (abs(obj.vx) + abs(obj.vy))

                # Bounce with damping
                speed = np.sqrt(obj.vx ** 2 + obj.vy ** 2)

                if abs(normal_y) > 0:  # Vertical collision
                    obj.vy = -obj.vy * self.bounce_damping
                    obj.vx *= self.friction

                    # Add horizontal velocity from rotation/torque
                    if abs(rel_x) > 2:
                        obj.vx += np.sign(rel_x) * speed * 0.2

                    if normal_y < 0:
                        obj.y -= 1
                else:  # Horizontal collision
                    obj.vx = -obj.vx * self.bounce_damping
                    obj.vy *= self.friction
                    if normal_x < 0:
                        obj.x -= 1
                    else:
                        obj.x += 1

                # Stop if very slow
                if abs(obj.vx) < 0.15 and abs(obj.vy) < 0.4 and abs(obj.angular_vel) < 0.02:
                    obj.vx = 0
                    obj.vy = 0
                    obj.angular_vel = 0

            # Draw object at current position
            for wx, wy in obj.get_world_cells():
                if 0 <= wx < self.width and 0 <= wy < self.height:
                    self.grid[wy, wx] = Material.METAL

        # Remove objects that fell off screen
        for i in reversed(objects_to_remove):
            self.metal_objects.pop(i)

    def _update_fish(self) -> None:
        """Update all fish - movement, lifecycle, collisions, and death."""
        # Cache material values as integers to avoid enum lookup overhead
        MAT_FISH = int(Material.FISH)
        MAT_WATER = int(Material.WATER)
        grid = self.grid
        width = self.width
        height = self.height

        fish_to_remove = set()

        # First pass: clear all fish from grid and fish_grid
        self.fish_grid.fill(-1)
        for fish in self.fish:
            for fx, fy in fish.get_cells():
                if 0 <= fx < width and 0 <= fy < height:
                    if grid[fy, fx] == MAT_FISH:
                        grid[fy, fx] = MAT_WATER

        # Second pass: update each fish
        for i, fish in enumerate(self.fish):
            if i in fish_to_remove:
                continue

            # Age the fish
            fish.age += 1
            if fish.age > fish.max_age:
                fish.alive = False

            # Size-based death chance (bigger fish more likely to die)
            death_chance = self.fish_death_chance * fish.size_value
            if np.random.random() < death_chance:
                fish.alive = False

            if not fish.alive:
                fish_to_remove.add(i)
                continue

            # Update swimming motion (no gravity - fish swim freely)
            fish.update_swimming()

            # Pre-check: if fish is near screen edge, force turn away from edge
            # Use center-of-mass pivot for natural turning animation
            com_offset = fish.get_center_of_mass_offset()
            com_x = fish.x + com_offset

            # Check if center of mass is too close to edges
            margin = 5
            if fish.direction == -1 and com_x < margin:
                # Near left edge, swimming left - turn right with pivot
                fish.turn_with_pivot()
                fish.vx = abs(fish.vx)
            elif fish.direction == 1 and com_x > self.width - margin:
                # Near right edge, swimming right - turn left with pivot
                fish.turn_with_pivot()
                fish.vx = -abs(fish.vx)

            # Calculate new position
            new_x = fish.x + fish.vx
            new_y = fish.y + fish.vy

            # Check if new position is valid
            can_move = True
            old_x, old_y = fish.x, fish.y
            fish.x, fish.y = new_x, new_y

            for fx, fy in fish.get_cells():
                if not (0 <= fx < width and 0 <= fy < height):
                    can_move = False
                    break
                cell = grid[fy, fx]
                if cell != MAT_WATER and cell != MAT_FISH:
                    can_move = False
                    break

            if can_move:
                # Movement successful
                pass
            else:
                # Can't move - revert position and turn with center-of-mass pivot
                fish.x, fish.y = old_x, old_y
                fish.turn_with_pivot()
                fish.randomize_speed()
                fish.vertical_tendency = -fish.vertical_tendency

                # Check if new position after pivot turn is valid
                found_valid = True
                for fx, fy in fish.get_cells():
                    if not (0 <= fx < width and 0 <= fy < height):
                        found_valid = False
                        break
                    cell = grid[fy, fx]
                    if cell != MAT_WATER and cell != MAT_FISH:
                        found_valid = False
                        break

                if not found_valid:
                    # Try small nudges to find valid position
                    for nudge_y in [0, -1, 1, -2, 2, -3, 3]:
                        for nudge_x in [0, fish.direction, fish.direction * 2, -fish.direction]:
                            test_x = fish.x + nudge_x
                            test_y = fish.y + nudge_y
                            fish.x, fish.y = test_x, test_y
                            fish._cached_cells = None

                            valid = True
                            for fx, fy in fish.get_cells():
                                if not (0 <= fx < width and 0 <= fy < height):
                                    valid = False
                                    break
                                cell = grid[fy, fx]
                                if cell != MAT_WATER and cell != MAT_FISH:
                                    valid = False
                                    break

                            if valid:
                                found_valid = True
                                break
                        if found_valid:
                            break

                if not found_valid:
                    # Still stuck - try moving toward center of screen
                    center_dir = 1 if old_x < self.width / 2 else -1
                    for dist in range(1, 10):
                        for nudge_y in [0, -1, 1, -2, 2]:
                            test_x = old_x + center_dir * dist
                            test_y = old_y + nudge_y
                            fish.x, fish.y = test_x, test_y
                            fish.direction = center_dir
                            fish.vx = center_dir * fish.swim_speed
                            fish._cached_cells = None
                            fish._cached_shape = None

                            valid = True
                            for fx, fy in fish.get_cells():
                                if not (0 <= fx < width and 0 <= fy < height):
                                    valid = False
                                    break
                                cell = grid[fy, fx]
                                if cell != MAT_WATER and cell != MAT_FISH:
                                    valid = False
                                    break

                            if valid:
                                found_valid = True
                                break
                        if found_valid:
                            break

                if not found_valid:
                    # Completely stuck - kill the fish
                    fish.alive = False
                    fish_to_remove.add(i)
                    fish.x, fish.y = old_x, old_y

        # Third pass: check for collisions using spatial hashing
        # Build spatial hash grid (cell size ~10 to group nearby fish)
        cell_size = 10
        spatial_grid = {}
        for i, fish in enumerate(self.fish):
            if i in fish_to_remove or not fish.alive:
                continue
            # Hash based on fish head position
            gx, gy = int(fish.x) // cell_size, int(fish.y) // cell_size
            key = (gx, gy)
            if key not in spatial_grid:
                spatial_grid[key] = []
            spatial_grid[key].append(i)

        # Check collisions only between fish in same or adjacent cells
        checked_pairs = set()
        for (gx, gy), fish_indices in spatial_grid.items():
            # Get all fish in this cell and adjacent cells
            nearby_indices = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (gx + dx, gy + dy)
                    if key in spatial_grid:
                        nearby_indices.extend(spatial_grid[key])

            for i in fish_indices:
                if i in fish_to_remove:
                    continue
                fish1 = self.fish[i]
                if not fish1.alive:
                    continue

                cells1 = set(fish1.get_cells())

                for j in nearby_indices:
                    if j <= i or j in fish_to_remove:
                        continue
                    pair = (min(i, j), max(i, j))
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)

                    fish2 = self.fish[j]
                    if not fish2.alive:
                        continue

                    cells2 = set(fish2.get_cells())

                    # Check for collision: overlapping OR adjacent (shared edge)
                    # Adjacent means any cell from fish1 is next to any cell from fish2
                    collision = False
                    if cells1 & cells2:
                        # Overlapping cells
                        collision = True
                    else:
                        # Check for adjacent cells (shared edge, not diagonal)
                        for (x1, y1) in cells1:
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                if (x1 + dx, y1 + dy) in cells2:
                                    collision = True
                                    break
                            if collision:
                                break

                    if not collision:
                        continue

                    # Collision! Check if eating happens or just bounce
                    if np.random.random() < self.fish_eat_chance:
                        # Eating happens - bigger fish eats smaller
                        if fish1.size_value > fish2.size_value:
                            # Fish1 eats fish2 - grow by half eaten fish's pixels
                            fish1.grow_from_eating(len(fish2.get_shape()))
                            fish1._cached_cells = None  # Clear cache after growth
                            fish1.randomize_speed()
                            fish2.alive = False
                            fish_to_remove.add(j)
                        elif fish2.size_value > fish1.size_value:
                            # Fish2 eats fish1 - grow by half eaten fish's pixels
                            fish2.grow_from_eating(len(fish1.get_shape()))
                            fish2._cached_cells = None  # Clear cache after growth
                            fish2.randomize_speed()
                            fish1.alive = False
                            fish_to_remove.add(i)
                            break  # Fish1 is dead, stop checking its collisions
                        else:
                            # Same size - 50/50 chance
                            if np.random.random() < 0.5:
                                fish1.grow_from_eating(len(fish2.get_shape()))
                                fish1._cached_cells = None  # Clear cache after growth
                                fish1.randomize_speed()
                                fish2.alive = False
                                fish_to_remove.add(j)
                            else:
                                fish2.grow_from_eating(len(fish1.get_shape()))
                                fish2._cached_cells = None  # Clear cache after growth
                                fish2.randomize_speed()
                                fish1.alive = False
                                fish_to_remove.add(i)
                                break
                    else:
                        # No eating - both fish turn around with pivot and change speed
                        fish1.turn_with_pivot()
                        fish1.randomize_speed()

                        fish2.turn_with_pivot()
                        fish2.randomize_speed()

        # Fourth pass: draw surviving fish and track in fish_grid with colors
        for i, fish in enumerate(self.fish):
            if i in fish_to_remove or not fish.alive:
                continue

            cells_drawn = 0
            expected_cells = len(fish.get_shape())

            for fx, fy, is_eye in fish.get_cells_with_type():
                if 0 <= fx < width and 0 <= fy < height:
                    cell = grid[fy, fx]
                    # Draw fish if cell is water or already fish (could be overlapping)
                    if cell == MAT_WATER or cell == MAT_FISH:
                        grid[fy, fx] = MAT_FISH
                        self.fish_grid[fy, fx] = fish.id
                        # Store color (eye color or body color)
                        if is_eye:
                            self.fish_color_grid[fy, fx] = fish.eye_color
                        else:
                            self.fish_color_grid[fy, fx] = fish.color
                        cells_drawn += 1

            # Structural integrity check: fish must have all its cells drawn
            if cells_drawn < expected_cells:
                # Fish lost integrity - mark for removal
                fish.alive = False
                fish_to_remove.add(i)

        # Fifth pass: verify structural integrity - check for orphaned fish pixels
        # Any FISH cell that doesn't match a valid fish ID gets cleared
        # Use vectorized operation for speed
        orphan_mask = (grid == MAT_FISH) & (self.fish_grid == -1)
        grid[orphan_mask] = MAT_WATER

        # Convert dead fish to skeletons and remove them
        for i in sorted(fish_to_remove, reverse=True):
            if i < len(self.fish):
                dead_fish = self.fish[i]
                # Create a skeleton at the fish's position with same shape
                # Extract (dx, dy) from shape tuples (ignoring is_eye flag)
                simple_shape = [(dx, dy) for dx, dy, _ in dead_fish.get_shape()]
                skeleton = Skeleton(
                    dead_fish.x, dead_fish.y,
                    simple_shape,
                    dead_fish.direction
                )
                self.skeletons.append(skeleton)
                self.fish.pop(i)

    def _try_spawn_fish(self) -> None:
        """Occasionally spawn fish in water pools. Spawn rate scales with water volume."""
        # Use cached water count
        water_count = self._get_water_count()

        if water_count < self.min_water_for_life:
            return

        # Limit fish population
        max_fish = max(1, water_count // 80)  # 1 fish per 80 water cells max
        if len(self.fish) >= max_fish:
            return

        # Spawn chance scales with water ratio (more water = more spawning)
        water_ratio = water_count / self.total_cells
        spawn_chance = self.fish_spawn_chance * water_ratio * 10  # Scale factor
        if np.random.random() > spawn_chance:
            return

        # Try random positions to find water (faster than argwhere for sparse water)
        max_attempts = 10
        spawn_x, spawn_y = None, None

        for _ in range(max_attempts):
            rx = np.random.randint(0, self.width)
            ry = np.random.randint(0, self.height)
            if self.grid[ry, rx] == Material.WATER:
                spawn_x, spawn_y = rx, ry
                break

        if spawn_x is None:
            return

        # Quick check for enough water nearby (use sampling instead of full scan)
        water_nearby = 0
        sample_offsets = [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1), (0, 0)]
        for dx, dy in sample_offsets:
            nx, ny = spawn_x + dx, spawn_y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.grid[ny, nx] == Material.WATER:
                    water_nearby += 1

        if water_nearby >= 5:  # Need decent pool size (adjusted for sampling)
            # Create fish and verify all its cells would be in water
            new_fish = Fish(spawn_x, spawn_y)
            can_spawn = True
            for fx, fy in new_fish.get_cells():
                if not (0 <= fx < self.width and 0 <= fy < self.height):
                    can_spawn = False
                    break
                if self.grid[fy, fx] != Material.WATER:
                    can_spawn = False
                    break

            if can_spawn:
                self.fish.append(new_fish)

    def _update_skeletons(self) -> None:
        """Update skeletons - sink through water and decay at bottom."""
        skeletons_to_remove = []

        for i, skeleton in enumerate(self.skeletons):
            # Clear skeleton from grid
            for sx, sy in skeleton.get_cells():
                if 0 <= sx < self.width and 0 <= sy < self.height:
                    if self.grid[sy, sx] == Material.SKELETON:
                        self.grid[sy, sx] = Material.WATER

            # Check if should be removed (decayed)
            if skeleton.update():
                skeletons_to_remove.append(i)
                continue

            if not skeleton.at_bottom:
                # Try to sink
                new_y = skeleton.y + skeleton.sink_speed

                # Check if can sink
                can_sink = True
                skeleton.y = new_y
                hit_bottom = False
                hit_fish = []  # Track fish that need to be pushed

                for sx, sy in skeleton.get_cells():
                    if sy >= self.height:
                        # Fell off screen
                        skeletons_to_remove.append(i)
                        can_sink = False
                        break
                    if not (0 <= sx < self.width):
                        can_sink = False
                        break
                    if 0 <= sy < self.height:
                        cell = self.grid[sy, sx]
                        # Can sink through water and air
                        if cell == Material.FISH:
                            # Find which fish is at this position and mark for pushing
                            for fish in self.fish:
                                if (sx, sy) in fish.get_cells():
                                    if fish not in hit_fish:
                                        hit_fish.append(fish)
                        elif cell not in (Material.WATER, Material.SKELETON, Material.AIR):
                            # Hit solid ground
                            can_sink = False
                            hit_bottom = True
                            break

                # Push any fish that are in the way
                for fish in hit_fish:
                    # Push fish to the side (randomly left or right)
                    push_dir = np.random.choice([-1, 1])
                    fish.x += push_dir * 2
                    fish.y -= 1  # Also push up slightly
                    # Clear fish's cached cells since position changed
                    fish._cached_cells = None

                if not can_sink:
                    skeleton.y = new_y - skeleton.sink_speed  # Revert
                    if hit_bottom:
                        skeleton.at_bottom = True

            # Draw skeleton at current position
            if i not in skeletons_to_remove:
                for sx, sy in skeleton.get_cells():
                    if 0 <= sx < self.width and 0 <= sy < self.height:
                        # Draw over water, air, or fish (skeleton takes priority)
                        if self.grid[sy, sx] in (Material.WATER, Material.AIR, Material.FISH):
                            self.grid[sy, sx] = Material.SKELETON

        # Remove decayed skeletons
        for i in reversed(skeletons_to_remove):
            if i < len(self.skeletons):
                self.skeletons.pop(i)

    def _update_seasons(self) -> None:
        """Update season state and handle evaporation during dry season."""
        water_ratio = self._get_water_count() / self.total_cells

        # Check for season transitions
        if self.season == 'rain' and water_ratio >= self.rain_threshold:
            self.season = 'dry'
        elif self.season == 'dry' and water_ratio <= self.dry_threshold:
            self.season = 'rain'

        # Evaporation during dry season, but only after all water has settled
        if self.season == 'dry' and self.evaporation_rate > 0:
            # Check if any water is still falling (has air below it)
            if not self._has_falling_water():
                self._evaporate_surface_water()

    def _has_falling_water(self) -> bool:
        """Check if any water cell has air below it (still falling)."""
        water_mask = self.grid == Material.WATER
        # Check if any water has air directly below (excluding bottom row)
        air_below = np.zeros_like(water_mask)
        air_below[:-1, :] = self.grid[1:, :] == Material.AIR
        falling_water = water_mask & air_below
        return np.any(falling_water)

    def _evaporate_surface_water(self) -> None:
        """Evaporate water from the surface - creates steam that rises."""
        # Vectorized surface water detection
        water_mask = self.grid == Material.WATER

        # Surface water: water with air or steam above, or at top row
        above_clear = np.zeros_like(water_mask)
        above_clear[0, :] = True  # Top row is always "surface" if water
        above_clear[1:, :] = (self.grid[:-1, :] == Material.AIR) | (self.grid[:-1, :] == Material.STEAM)

        surface_water = water_mask & above_clear

        # Apply random evaporation - turn water into steam
        evap_mask = surface_water & (np.random.random(self.grid.shape) < self.evaporation_rate)
        evap_count = np.sum(evap_mask)
        if evap_count > 0:
            self.grid[evap_mask] = Material.STEAM
            self._water_count -= evap_count

    def _update_steam(self) -> None:
        """Update steam particles - rise up and drift with wind."""
        STEAM = int(Material.STEAM)
        AIR = int(Material.AIR)

        g = self.grid
        h, w = self.height, self.width
        wind = self.wind_speed

        # Find all steam positions (vectorized) - much faster than checking every cell
        steam_positions = np.argwhere(g == STEAM)
        if len(steam_positions) == 0:
            return

        # Sort by y (top to bottom) to avoid conflicts
        steam_positions = steam_positions[steam_positions[:, 0].argsort()]

        # Pre-calculate wind parameters
        wind_dir = 1 if wind >= 0 else -1
        wind_chance = abs(wind) * 0.7
        rise_chance = 0.8 - wind_chance * 0.3
        drift_bias = 0.5 + abs(wind) * 0.4

        # Generate random values for all steam particles at once
        n = len(steam_positions)
        rand_vals = np.random.random(n)
        drift_rand = np.random.random(n)
        dissipate_rand = np.random.random(n)

        for i, (y, x) in enumerate(steam_positions):
            if y == 0 or g[y, x] != STEAM:  # May have been moved by previous iteration
                continue

            moved = False
            rand = rand_vals[i]

            if rand < rise_chance:
                # Try to rise straight up
                if g[y - 1, x] == AIR:
                    g[y - 1, x] = STEAM
                    g[y, x] = AIR
                    moved = True
                else:
                    # Try diagonal up (wind direction preferred)
                    for dx in [wind_dir, -wind_dir]:
                        nx = x + dx
                        if 0 <= nx < w and g[y - 1, nx] == AIR:
                            g[y - 1, nx] = STEAM
                            g[y, x] = AIR
                            moved = True
                            break
            else:
                # Wind drift - move horizontally
                dx = wind_dir if drift_rand[i] < drift_bias else -wind_dir
                nx = x + dx
                if 0 <= nx < w:
                    if g[y, nx] == AIR:
                        g[y, nx] = STEAM
                        g[y, x] = AIR
                        moved = True
                    elif y > 0 and g[y - 1, nx] == AIR:
                        g[y - 1, nx] = STEAM
                        g[y, x] = AIR
                        moved = True

            # Small chance to dissipate randomly
            if not moved and dissipate_rand[i] < 0.01:
                g[y, x] = AIR

        # Steam at top row or edges disappears (vectorized)
        g[0, g[0, :] == STEAM] = AIR
        g[g[:, 0] == STEAM, 0] = AIR
        g[g[:, w - 1] == STEAM, w - 1] = AIR

    def _get_water_count(self) -> int:
        """Get cached water count, recalculating only if dirty."""
        if self._water_count_dirty:
            self._water_count = np.sum(self.grid == Material.WATER)
            self._water_count_dirty = False
        return self._water_count

    def get_water_ratio(self) -> float:
        """Get current water level as a ratio of total cells."""
        return self._get_water_count() / self.total_cells

    def drop_metal_object(self) -> None:
        """Drop a random metallic shape from the sky."""
        center_x = np.random.randint(30, self.width - 30)
        start_y = 5

        shape_type = np.random.randint(0, 5)
        shape = []

        if shape_type == 0:
            # Rectangle
            w = np.random.randint(12, 20)
            h = np.random.randint(8, 15)
            for dy in range(h):
                for dx in range(w):
                    shape.append((dx - w // 2, dy))

        elif shape_type == 1:
            # L-shape
            thickness = np.random.randint(3, 5)
            leg_h = np.random.randint(12, 20)
            leg_w = np.random.randint(10, 16)
            for dy in range(leg_h):
                for t in range(thickness):
                    shape.append((t, dy))
            for dx in range(leg_w):
                for t in range(thickness):
                    if (dx, leg_h - 1 - t) not in shape:
                        shape.append((dx, leg_h - 1 - t))

        elif shape_type == 2:
            # T-shape
            thickness = np.random.randint(3, 5)
            top_w = np.random.randint(15, 25)
            stem_h = np.random.randint(10, 18)
            for dx in range(top_w):
                for t in range(thickness):
                    shape.append((dx - top_w // 2, t))
            for dy in range(stem_h):
                for t in range(thickness):
                    if (t - thickness // 2, thickness + dy) not in shape:
                        shape.append((t - thickness // 2, thickness + dy))

        elif shape_type == 3:
            # Cross / plus shape
            thickness = np.random.randint(3, 5)
            arm_len = np.random.randint(8, 14)
            for dx in range(-arm_len, arm_len + 1):
                for t in range(-thickness // 2, thickness // 2 + 1):
                    shape.append((dx, t))
            for dy in range(-arm_len, arm_len + 1):
                for t in range(-thickness // 2, thickness // 2 + 1):
                    if (t, dy) not in shape:
                        shape.append((t, dy))

        else:
            # Irregular blob
            num_cells = np.random.randint(80, 150)
            shape = [(0, 0)]
            for _ in range(num_cells - 1):
                base = shape[np.random.randint(len(shape))]
                dx = base[0] + np.random.choice([-1, 0, 1])
                dy = base[1] + np.random.choice([-1, 0, 1])
                if (dx, dy) not in shape:
                    shape.append((dx, dy))

        # Normalize shape
        min_dy = min(dy for _, dy in shape)
        shape = [(dx, dy - min_dy) for dx, dy in shape]

        # Create object with physics
        obj = MetalObject(center_x, start_y, shape)
        obj.angular_vel = np.random.uniform(-0.08, 0.08)

        self.metal_objects.append(obj)

    def add_rain(self, intensity: int, wind: float = 0.0) -> None:
        """Spawn rain particles at the top of the simulation."""
        added = 0
        xs = np.random.randint(0, self.width, intensity)
        for x in xs:
            if wind != 0:
                x = int(x + wind * self.width * 0.1)
                x = max(0, min(self.width - 1, x))
            if self.grid[0, x] == Material.AIR:
                self.grid[0, x] = Material.WATER
                added += 1
        self._water_count += added

    def add_material(self, x: int, y: int, material: int, radius: int = 3) -> None:
        """Add material at a position with given radius."""
        # Mark water count as dirty since we may be adding/removing water
        self._water_count_dirty = True
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        if material == Material.WATER and self.grid[ny, nx] == Material.STONE:
                            continue
                        self.grid[ny, nx] = material

    def remove_material(self, x: int, y: int, radius: int = 3) -> None:
        """Remove material (set to air) at a position with given radius."""
        self.add_material(x, y, Material.AIR, radius)
