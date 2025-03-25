import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # better interactivity
import matplotlib.pyplot as plt
import random
from matplotlib.widgets import Button

# Element types
EMPTY = 0
WALL = 1
SAND = 2
WOOD = 3
FIRE = 4
DARK_SMOKE = 5
LIGHT_SMOKE = 6
BALLOON = 7
WATER = 8 # Water values from 8 to 15

# Colors for visualization
COLORS = {
    EMPTY: (0.5, 0.5, 0.5),  # Gray
    WALL: (0.0, 0.0, 0.0),  # Black
    SAND: (0.9, 0.8, 0.2),  # Yellow
    WOOD: (0.6, 0.3, 0.1),  # Brown
    FIRE: (1.0, 0.0, 0.0),  # Red
    DARK_SMOKE: (0.3, 0.3, 0.3),  # Dark Gray
    LIGHT_SMOKE: (0.7, 0.7, 0.7),  # Light Gray
    BALLOON: (1.0, 0.0, 1.0),  # Magenta
}


def get_water_color(value):
    """Get water color based on amount (8-16)"""
    # Normalize the value between 0 (lightest) and 1 (darkest)
    normalized = (value - 8) / 8  # Map 8-16 to 0-1
    normalized = max(0, min(1, normalized))  # Clamp to 0-1 range

    # Interpolate between light blue and dark blue
    light_blue = (0.7, 0.85, 1.0)  # Light blue
    dark_blue = (0.0, 0.2, 0.8)  # Dark blue

    r = light_blue[0] + normalized * (dark_blue[0] - light_blue[0])
    g = light_blue[1] + normalized * (dark_blue[1] - light_blue[1])
    b = light_blue[2] + normalized * (dark_blue[2] - light_blue[2])

    return (r, g, b)


class CellularAutomaton2D:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=float)
        self.next_grid = np.zeros((height, width), dtype=float)
        self.smoke_lifetimes = {}  # Track smoke lifetimes

    def generate_cave(self, fill_ratio=0.45, iterations=15):
        """Generate a cave using B678/S2345678 rule"""
        # Initialize grid with random cells
        for y in range(self.height):
            for x in range(self.width):
                # Create border walls
                if (x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1):
                    self.grid[y, x] = WALL
                # Random fill based on ratio
                elif random.random() < fill_ratio:
                    self.grid[y, x] = WALL
                else:
                    self.grid[y, x] = EMPTY

        # Apply B678/S2345678 rule for specified iterations
        for _ in range(iterations):
            self.next_grid = self.grid.copy()

            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    # Count neighbors
                    neighbors = self._count_neighbors(x, y, WALL)

                    # Apply B678/S2345678 rule
                    if self.grid[y, x] == EMPTY:
                        # Birth: B678
                        if neighbors >= 6:
                            self.next_grid[y, x] = WALL
                    else:  # Cell is alive
                        # Survival: S2345678
                        if neighbors < 2:
                            self.next_grid[y, x] = EMPTY

            self.grid = self.next_grid.copy()

    def _count_neighbors(self, x, y, element_type):
        """Count neighbors of specified type"""
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and
                        int(self.grid[ny, nx]) == element_type):
                    count += 1
        return count

    def update(self):
        """Update the grid for one generation"""
        self.next_grid = self.grid.copy()

        # Process cells in random order to avoid directional bias
        coordinates = [(x, y) for y in range(self.height) for x in range(self.width)]
        random.shuffle(coordinates)

        for x, y in coordinates:
            cell_type = int(self.grid[y, x])

            # Skip empty cells and walls
            if cell_type == EMPTY or cell_type == WALL:
                continue

            # Process each element type
            if cell_type == SAND:
                self._update_sand(x, y)
            elif cell_type == WOOD:
                self._update_wood(x, y)
            elif cell_type == FIRE:
                self._update_fire(x, y)
            elif cell_type in [DARK_SMOKE, LIGHT_SMOKE]:
                self._update_smoke(x, y, cell_type)
            elif cell_type >= WATER:
                self._update_water(x, y)
            elif cell_type == BALLOON:
                self._update_balloon(x, y)

        self.grid = self.next_grid.copy()

        # Update smoke lifetimes and remove expired smoke
        expired = []
        for pos, lifetime in self.smoke_lifetimes.items():
            if lifetime <= 0:
                x, y = pos
                if (0 <= x < self.width and 0 <= y < self.height and
                        int(self.grid[y, x]) in [DARK_SMOKE, LIGHT_SMOKE]):
                    self.grid[y, x] = EMPTY
                expired.append(pos)
            else:
                self.smoke_lifetimes[pos] -= 1

        for pos in expired:
            del self.smoke_lifetimes[pos]

    def _is_empty(self, x, y):
        """Check if a cell is empty and within bounds"""
        return (0 <= x < self.width and 0 <= y < self.height and
                self.next_grid[y, x] == EMPTY)

    def _is_within_bounds(self, x, y):
        """Check if coordinates are within grid bounds"""
        return 0 <= x < self.width and 0 <= y < self.height

    def _update_sand(self, x, y):
        """Update sand behavior"""
        # Try to move down
        if self._is_empty(x, y + 1):
            self.next_grid[y, x] = EMPTY
            self.next_grid[y + 1, x] = SAND
        # If can't move down, try diagonal
        elif (self._is_empty(x - 1, y + 1) or self._is_empty(x + 1, y + 1)):
            self.next_grid[y, x] = EMPTY

            options = []
            if self._is_empty(x - 1, y + 1):
                options.append((x - 1, y + 1))
            if self._is_empty(x + 1, y + 1):
                options.append((x + 1, y + 1))

            if options:
                nx, ny = random.choice(options)
                self.next_grid[ny, nx] = SAND

    def _update_wood(self, x, y):
        """Update wood behavior"""
        # Wood falls down if there's space below
        if self._is_empty(x, y + 1):
            self.next_grid[y, x] = EMPTY
            self.next_grid[y + 1, x] = WOOD
            return  # Wood moved, stop checking for fire

    def _update_fire(self, x, y):
        """Update fire behavior"""

        # Check if wood below
        if 0 <= y + 1 < self.height and int(self.grid[y + 1, x]) == WOOD:
            self.next_grid[y + 1, x] = FIRE
            self.next_grid[y, x] = DARK_SMOKE
            return

        # Fire tries to move randomly downward when possible
        directions = [(0, 1), (-1, 1), (1, 1)]  # Down, Down-left, Down-right
        random.shuffle(directions)

        moved = False
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self._is_empty(nx, ny):
                self.next_grid[y, x] = EMPTY
                self.next_grid[ny, nx] = FIRE
                moved = True
                break

        # Fire did not move
        if not moved:
            # Check if wood below
            if (0 <= y + 1 < self.height and int(self.grid[y + 1, x]) == WOOD):
                self.next_grid[y + 1, x] = FIRE
                self.next_grid[y, x] = DARK_SMOKE
            else:
                self.next_grid[y, x] = LIGHT_SMOKE

    def _update_smoke(self, x, y, smoke_type):
        """Update smoke behavior"""
        # Smoke rises upward
        directions = [(0, -1), (-1, -1), (1, -1)]  # Up, Up-left, Up-right
        random.shuffle(directions)
        nx, ny = 0, 0
        moved = False
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self._is_empty(nx, ny):
                self.next_grid[y, x] = EMPTY
                self.next_grid[ny, nx] = smoke_type

                # Transfer the lifetime to the new position
                if (x, y) in self.smoke_lifetimes:
                    self.smoke_lifetimes[(nx, ny)] = self.smoke_lifetimes[(x, y)]
                    del self.smoke_lifetimes[(x, y)]
                else:
                    # Create new lifetime
                    if smoke_type == DARK_SMOKE:
                        self.smoke_lifetimes[(nx, ny)] = random.randint(10, 20)
                    else:
                        self.smoke_lifetimes[(nx, ny)] = random.randint(7, 10)

                moved = True
                break

        # If can't move upward, try sideways
        if not moved:
            sideways = [(-1, 0), (1, 0)]  # Left, Right
            random.shuffle(sideways)

            for dx, dy in sideways:
                nx, ny = x + dx, y + dy
                if self._is_empty(nx, ny):
                    self.next_grid[y, x] = EMPTY
                    self.next_grid[ny, nx] = smoke_type

                    moved = True
                    break

        if moved:
            # Transfer the lifetime
            if (x, y) in self.smoke_lifetimes:
                self.smoke_lifetimes[(nx, ny)] = self.smoke_lifetimes[(x, y)]
                del self.smoke_lifetimes[(x, y)]
            else:
                # Create new lifetime
                if smoke_type == DARK_SMOKE:
                    self.smoke_lifetimes[(nx, ny)] = random.randint(10, 20)
                else:
                    self.smoke_lifetimes[(nx, ny)] = random.randint(7, 10)

    def _update_water(self, x, y):
        """Update water behavior with complete conservation of mass"""
        water_amount = self.grid[y, x]
        MIN_WATER = 8.0
        MAX_WATER = 15.0

        # DOWNWARD FLOW
        if self._is_empty(x, y + 1):
            self.next_grid[y, x] = EMPTY
            self.next_grid[y + 1, x] = water_amount
            return

        # PRESSURE EQUALIZATION WITH WATER BELOW
        if y + 1 < self.height and MIN_WATER <= self.next_grid[y + 1, x] < MAX_WATER:
            # Water below, add this water to it
            below_amount = self.next_grid[y + 1, x]
            excess = water_amount - MIN_WATER
            if (below_amount + excess) > MAX_WATER:
                # Can't fit all excess below, keep some here
                transfer = MAX_WATER - below_amount
                self.next_grid[y, x] = water_amount - transfer
                self.next_grid[y + 1, x] = MAX_WATER
            else:
                # Transfer all excess
                self.next_grid[y, x] = EMPTY
                self.next_grid[y + 1, x] = below_amount + (water_amount - MIN_WATER)
            return

        # HORIZONTAL FLOW
        left_empty = self._is_empty(x - 1, y)
        right_empty = self._is_empty(x + 1, y)

        if left_empty or right_empty:
            excess = water_amount - MIN_WATER
            if excess > 0.5:
                if left_empty and right_empty:
                    self.next_grid[y, x] = MIN_WATER + excess / 3
                    self.next_grid[y, x - 1] = MIN_WATER + excess / 3
                    self.next_grid[y, x + 1] = MIN_WATER + excess / 3
                elif left_empty:
                    self.next_grid[y, x] = MIN_WATER + excess / 2
                    self.next_grid[y, x - 1] = MIN_WATER + excess / 2
                elif right_empty:
                    self.next_grid[y, x] = MIN_WATER + excess / 2
                    self.next_grid[y, x + 1] = MIN_WATER + excess / 2
                return

        # EQUALIZE WITH EXISTING WATER AT SIDES
        side_cells = []
        total_water = water_amount
        cells_count = 1

        for dx in [-1, 1]:
            nx = x + dx
            if not self._is_within_bounds(nx, y):
                continue
            if self.next_grid[y, nx] >= MIN_WATER:
                side_amount = self.next_grid[y, nx]
                side_cells.append((nx, side_amount))
                total_water += side_amount
                cells_count += 1

        if side_cells:
            # Calculate equilibrium level
            equilibrium = total_water / cells_count
            self.next_grid[y, x] = equilibrium

            for nx, side_amount in side_cells:
                self.next_grid[y, nx] = equilibrium

            return

        self.next_grid[y, x] = water_amount

    def _update_balloon(self, x, y):
        """Update balloon behavior"""
        directions = [(0, -1), (-1, -1), (1, -1)]  # Up, Up-left, Up-right
        random.shuffle(directions)
        dx, dy = directions[0]
        nx, ny = x + dx, y + dy
        if self._is_empty(nx, ny):
            self.next_grid[y, x] = EMPTY
            self.next_grid[ny, nx] = BALLOON
        else:
            self.next_grid[y, x] = EMPTY

    def add_element(self, x, y, element_type):
        """Add an element at the specified position"""
        if 0 <= x < self.width and 0 <= y < self.height and self.grid[y, x] != WALL:
            # self.grid[y, x] = element_type
            if element_type == WATER:
                # Random water amount between 8 and 15
                self.grid[y, x] = 15
            else:
                self.grid[y, x] = element_type

            # Initialize smoke lifetime if needed
            if element_type in [DARK_SMOKE, LIGHT_SMOKE]:
                # Create new lifetime
                if element_type == DARK_SMOKE:
                    self.smoke_lifetimes[(x, y)] = random.randint(10, 20)
                elif element_type == LIGHT_SMOKE:
                    self.smoke_lifetimes[(x, y)] = random.randint(7, 10)


def run_simulation(width, height):
    """Run the cellular automaton simulation"""
    ca = CellularAutomaton2D(width, height)
    ca.generate_cave()

    # Create figure and axis for animation
    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(np.zeros((height, width, 3)), origin='upper')

    # Button axes
    button_axes = {}
    elements = ["Sand", "Water", "Wood", "Fire", "Balloon", "Wall", "Clear"]
    button_height = 0.05
    button_width = 0.1
    spacing = 0.02

    for i, element in enumerate(elements):
        pos = [0.1 + i * (button_width + spacing), 0.01, button_width, button_height]
        button_axes[element] = plt.axes(pos)

    # Create buttons
    buttons = {}
    selected_element = [SAND]

    def create_button_callback(elem_type):
        def callback(event):
            selected_element[0] = elem_type

        return callback

    buttons["Sand"] = Button(button_axes["Sand"], "Sand")
    buttons["Sand"].on_clicked(create_button_callback(SAND))

    buttons["Water"] = Button(button_axes["Water"], "Water")
    buttons["Water"].on_clicked(create_button_callback(WATER))

    buttons["Wood"] = Button(button_axes["Wood"], "Wood")
    buttons["Wood"].on_clicked(create_button_callback(WOOD))

    buttons["Fire"] = Button(button_axes["Fire"], "Fire")
    buttons["Fire"].on_clicked(create_button_callback(FIRE))

    buttons["Balloon"] = Button(button_axes["Balloon"], "Balloon")
    buttons["Balloon"].on_clicked(create_button_callback(BALLOON))

    buttons["Wall"] = Button(button_axes["Wall"], "Wall")
    buttons["Wall"].on_clicked(create_button_callback(WALL))

    def clear_callback(event):
        ca.grid = ca.grid * 0
        ca.next_grid = ca.next_grid * 0
        ca.smoke_lifetimes = {}
        ca.generate_cave()

    buttons["Clear"] = Button(button_axes["Clear"], "Clear")
    buttons["Clear"].on_clicked(clear_callback)

    # Mouse interaction
    drawing = [False]

    def on_press(event):
        if event.inaxes == ax:
            drawing[0] = True
            # x, y = int(event.xdata), int(event.ydata)
            x, y = int(round(event.xdata)), int(round(event.ydata))
            ca.add_element(x, y, selected_element[0])

    def on_release(event):
        drawing[0] = False

    def on_motion(event):
        if drawing[0] and event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            ca.add_element(x, y, selected_element[0])

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    # Update function for the timer
    def update_frame():
        ca.update()
        # Convert grid to RGB
        rgb_grid = np.zeros((height, width, 3))
        for y in range(height):
            for x in range(width):
                cell_type = int(ca.grid[y, x])
                if cell_type >= WATER and cell_type <= 15:
                    # Use dynamic water color based on value
                    rgb_grid[y, x] = get_water_color(ca.grid[y, x])
                elif cell_type in COLORS:
                    rgb_grid[y, x] = COLORS[cell_type]
        img.set_array(rgb_grid)
        fig.canvas.draw_idle()
        return True  # Keep the timer running

    # Create a timer for regular updates
    timer = fig.canvas.new_timer(interval=500)  # Update every 100ms
    timer.add_callback(update_frame)
    timer.start()

    # Show the figure (will block until closed)
    plt.show()


def CellularAutomaton_2D():
    print("2D Cellular Automaton - Sand, Water, Fire and more")
    width, height = 100, 80
    run_simulation(width, height)
