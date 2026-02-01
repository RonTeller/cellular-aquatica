"""Cellular automata simulation engine for terrain and fluid dynamics."""

import numpy as np
from materials import Material


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

        # Physics constants
        self.gravity = 0.5
        self.bounce_damping = 0.5  # Energy retained after bounce
        self.friction = 0.85
        self.rotation_damping = 0.92

    def update(self) -> None:
        """Perform one simulation step using checkerboard pattern."""
        self.frame += 1
        offset = self.frame % 2

        # Update rigid metal objects first
        self._update_metal_objects()

        # Pre-generate random directions for this frame
        rand_dirs = np.random.randint(0, 2, (self.height, self.width)) * 2 - 1

        # Alternate x-iteration direction each frame to reduce left/right bias
        reverse_x = (self.frame // 2) % 2 == 1

        # Process from bottom to top for falling materials
        for y in range(self.height - 2, -1, -1):
            start_x = (y + offset) % 2
            x_range = range(start_x, self.width, 2)
            if reverse_x:
                x_range = range(self.width - 1 - ((self.width - 1 - start_x) % 2), -1, -2)

            for x in x_range:
                material = self.grid[y, x]
                if material == Material.WATER:
                    self._update_water(x, y, rand_dirs[y, x])
                elif material == Material.DIRT:
                    self._update_dirt(x, y, rand_dirs[y, x])

    def _update_water(self, x: int, y: int, rand_dir: int) -> None:
        """Update a water cell with 2-cell lookahead for better flow."""
        if self.grid[y, x] != Material.WATER:
            return

        g = self.grid
        h, w = self.height, self.width

        # Fall off bottom of screen
        if y == h - 1:
            g[y, x] = Material.AIR
            return

        # 1. Try to move straight down
        if y + 1 < h and g[y + 1, x] == Material.AIR:
            g[y + 1, x] = Material.WATER
            g[y, x] = Material.AIR
            return

        # 2. Try diagonal down
        for dx in [rand_dir, -rand_dir]:
            nx, ny = x + dx, y + 1
            if 0 <= nx < w and ny < h and g[ny, nx] == Material.AIR:
                g[ny, nx] = Material.WATER
                g[y, x] = Material.AIR
                return

        # 3. Look 2 cells ahead
        for dx in [rand_dir, -rand_dir]:
            nx = x + dx
            if 0 <= nx < w and g[y, nx] == Material.AIR:
                nx2 = x + dx * 2
                can_flow_down = (y + 1 < h and g[y + 1, nx] == Material.AIR)
                can_flow_down2 = (0 <= nx2 < w and y + 1 < h and g[y + 1, nx2] == Material.AIR)
                if can_flow_down or can_flow_down2:
                    g[y, nx] = Material.WATER
                    g[y, x] = Material.AIR
                    return

        # 4. Simple horizontal spread
        for dx in [rand_dir, -rand_dir]:
            nx = x + dx
            if 0 <= nx < w and g[y, nx] == Material.AIR:
                g[y, nx] = Material.WATER
                g[y, x] = Material.AIR
                return

    def _update_dirt(self, x: int, y: int, rand_dir: int) -> None:
        """Update a dirt cell - falls down if possible."""
        if self.grid[y, x] != Material.DIRT:
            return

        g = self.grid
        h, w = self.height, self.width

        # Fall off bottom of screen
        if y == h - 1:
            g[y, x] = Material.AIR
            return

        # 1. Try to fall straight down
        if y + 1 < h and g[y + 1, x] == Material.AIR:
            g[y + 1, x] = Material.DIRT
            g[y, x] = Material.AIR
            return

        # 2. Try diagonal fall
        for dx in [rand_dir, -rand_dir]:
            nx, ny = x + dx, y + 1
            if 0 <= nx < w and ny < h:
                if g[y, nx] == Material.AIR and g[ny, nx] == Material.AIR:
                    g[ny, nx] = Material.DIRT
                    g[y, x] = Material.AIR
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
        xs = np.random.randint(0, self.width, intensity)
        for x in xs:
            if wind != 0:
                x = int(x + wind * self.width * 0.1)
                x = max(0, min(self.width - 1, x))
            if self.grid[0, x] == Material.AIR:
                self.grid[0, x] = Material.WATER

    def add_material(self, x: int, y: int, material: int, radius: int = 3) -> None:
        """Add material at a position with given radius."""
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
