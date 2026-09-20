import random
import tkinter as tk

# Practical 04 - Informed Search / A*
from agent import SearchAgent


class VisualGridHuntGame:
    """
    Grid-based environment for Practical 02, Practical 03 and Practical 04.

    Practical 02:
        - Simple Reflex Agent
        - Model-Based Agent

    Practical 03:
        - Search Agent
        - BFS / DFS / UCS
        - Offline planning

    Practical 04:
        - Informed Search
        - A* Search
        - Manhattan Distance Heuristic
        - Euclidean Distance Heuristic
    """

    DIRECTIONS = ["Up", "Right", "Down", "Left"]

    DELTAS = {
        "Up": (0, -1),
        "Right": (1, 0),
        "Down": (0, 1),
        "Left": (-1, 0),
    }

    def __init__(
        self,
        width=10,
        height=10,
        num_food=10,
        num_opponents=0,
        custom_walls=None,
    ):
        self.width = width
        self.height = height

        # =====================================================
        # AGENT STARTING POSITION
        # =====================================================

        self.agent_pos = [0, 0]

        # Initial direction
        self.facing = "Right"

        # =====================================================
        # WALLS
        # =====================================================

        if custom_walls is not None:

            self.walls = set(custom_walls)

        else:

            self.walls = set()

            possible_walls = [
                (x, y)
                for x in range(width)
                for y in range(height)
                if (x, y) not in [(0, 0), (1, 0)]
            ]

            random.shuffle(possible_walls)

            wall_count = max(
                5,
                (width * height) // 10
            )

            for wall in possible_walls[:wall_count]:

                self.walls.add(wall)

        # =====================================================
        # FOOD
        # =====================================================

        available_cells = [
            (x, y)
            for x in range(width)
            for y in range(height)
            if (x, y) not in self.walls
            and (x, y) != (0, 0)
        ]

        random.shuffle(available_cells)

        self.food_positions = set(
            available_cells[
                :min(num_food, len(available_cells))
            ]
        )

        # =====================================================
        # OPPONENTS
        # =====================================================

        available_for_opponents = [
            cell
            for cell in available_cells
            if cell not in self.food_positions
        ]

        random.shuffle(available_for_opponents)

        self.opponents = set(
            available_for_opponents[
                :min(
                    num_opponents,
                    len(available_for_opponents)
                )
            ]
        )

        # =====================================================
        # TOXIC TRAPS
        # =====================================================

        self.toxic_traps = set()

        possible_traps = [
            cell
            for cell in available_cells
            if cell not in self.food_positions
            and cell not in self.opponents
        ]

        random.shuffle(possible_traps)

        trap_count = min(
            5,
            len(possible_traps)
        )

        self.toxic_traps = set(
            possible_traps[:trap_count]
        )

        # =====================================================
        # GAME VARIABLES
        # =====================================================

        self.score = 0
        self.steps = 0
        self.collision = False

    # =========================================================
    # POSITION HELPERS
    # =========================================================

    def _next_position(self, direction):

        dx, dy = self.DELTAS[direction]

        return (
            self.agent_pos[0] + dx,
            self.agent_pos[1] + dy,
        )

    def _is_blocked(self, position):

        x, y = position

        # Outside grid
        if x < 0 or x >= self.width:
            return True

        if y < 0 or y >= self.height:
            return True

        # Wall
        if position in self.walls:
            return True

        return False

    def _relative_direction(self, offset):

        current_index = self.DIRECTIONS.index(
            self.facing
        )

        return self.DIRECTIONS[
            (current_index + offset) % 4
        ]

    # =========================================================
    # PERCEPT
    # =========================================================

    def get_percept(self):
        """
        Return percept information.

        Practical 03 and Practical 04 expose the world model:
            grid_size
            walls
            all_food
        """

        forward_direction = self._relative_direction(0)
        right_direction = self._relative_direction(1)
        back_direction = self._relative_direction(2)
        left_direction = self._relative_direction(3)

        forward_position = self._next_position(
            forward_direction
        )

        right_position = self._next_position(
            right_direction
        )

        back_position = self._next_position(
            back_direction
        )

        left_position = self._next_position(
            left_direction
        )

        return {

            # =================================================
            # PRACTICAL 02 PERCEPTS
            # =================================================

            "wall_ahead":
                self._is_blocked(
                    forward_position
                ),

            "wall_left":
                self._is_blocked(
                    left_position
                ),

            "wall_right":
                self._is_blocked(
                    right_position
                ),

            "wall_back":
                self._is_blocked(
                    back_position
                ),

            "food_here":
                tuple(self.agent_pos)
                in self.food_positions,

            "smells_toxin":
                tuple(self.agent_pos)
                in self.toxic_traps,

            "collision":
                self.collision,

            "facing":
                self.facing,

            # =================================================
            # WORLD MODEL FOR SEARCH
            # =================================================

            "grid_size":
                (self.width, self.height),

            "walls":
                list(self.walls),

            "all_food":
                list(self.food_positions),
        }

    # =========================================================
    # EXECUTE ACTION
    # =========================================================

    def execute_action(self, action):

        self.collision = False

        # =====================================================
        # TURN LEFT
        # =====================================================

        if action == "TurnLeft":

            current_index = self.DIRECTIONS.index(
                self.facing
            )

            self.facing = self.DIRECTIONS[
                (current_index - 1) % 4
            ]

        # =====================================================
        # TURN RIGHT
        # =====================================================

        elif action == "TurnRight":

            current_index = self.DIRECTIONS.index(
                self.facing
            )

            self.facing = self.DIRECTIONS[
                (current_index + 1) % 4
            ]

        # =====================================================
        # SUCK
        # =====================================================

        elif action == "Suck":

            current_position = tuple(
                self.agent_pos
            )

            if current_position in self.food_positions:

                self.food_positions.remove(
                    current_position
                )

                self.score += 10

        # =====================================================
        # MOVE FORWARD
        # =====================================================

        elif action == "MoveForward":

            next_position = self._next_position(
                self.facing
            )

            # -------------------------------------------------
            # Blocked
            # -------------------------------------------------

            if self._is_blocked(next_position):

                self.collision = True

                self.score -= 1

            else:

                self.agent_pos[0] = next_position[0]
                self.agent_pos[1] = next_position[1]

                self.steps += 1

                current_position = tuple(
                    self.agent_pos
                )

                # -------------------------------------------------
                # Collect food automatically
                # -------------------------------------------------

                if current_position in self.food_positions:

                    self.food_positions.remove(
                        current_position
                    )

                    self.score += 10

                # -------------------------------------------------
                # Opponent collision
                # -------------------------------------------------

                if current_position in self.opponents:

                    self.collision = True

                    self.score -= 5

        else:

            print(
                f"Unknown action: {action}"
            )

    # =========================================================
    # GAME END
    # =========================================================

    def is_done(self):

        return len(self.food_positions) == 0


# =============================================================
# SIMPLE REFLEX AGENT
# =============================================================

class SimpleReflexAgent:
    """
    Simple Reflex Agent from Practical 02.
    """

    def sense_and_act(self, percept):

        # Food here
        if percept["food_here"]:

            return "Suck"

        # Wall ahead
        if percept["wall_ahead"]:

            return "TurnRight"

        # Otherwise move
        return "MoveForward"


# =============================================================
# MODEL-BASED AGENT
# =============================================================

class ModelBasedAgent:
    """
    Model-Based Agent from Practical 02.
    """

    def __init__(self):

        self.last_action = None

        self.internal_position = [0, 0]

        self.facing = "Right"

    def sense_and_act(self, percept):

        # Food here
        if percept["food_here"]:

            self.last_action = "Suck"

            return "Suck"

        # Wall ahead
        if percept["wall_ahead"]:

            self.last_action = "TurnRight"

            return "TurnRight"

        # Move forward
        self.last_action = "MoveForward"

        return "MoveForward"


# =============================================================
# GUI
# =============================================================

class GridGameGUI:

    def __init__(
        self,
        root,
        width=10,
        height=10,
        cell_size=50,
        agent_type="search",
        search_algorithm="AStar",
    ):

        self.root = root

        self.width = width
        self.height = height

        self.cell_size = cell_size

        self.agent_type = agent_type

        self.search_algorithm = (
            search_algorithm.upper()
        )

        # =====================================================
        # ENVIRONMENT
        # =====================================================

        self.env = VisualGridHuntGame(
            width=width,
            height=height,
            num_food=10,
            num_opponents=0,
        )

        # =====================================================
        # CREATE AGENT
        # =====================================================

        self.create_agent()

        # =====================================================
        # WINDOW
        # =====================================================

        self.root.title(
            "IT3012 Practical 04 - Informed Search (A*)"
        )

        # =====================================================
        # CANVAS
        # =====================================================

        self.canvas = tk.Canvas(
            root,
            width=width * cell_size,
            height=height * cell_size,
            bg="white",
        )

        self.canvas.pack()

        # =====================================================
        # INFORMATION LABEL
        # =====================================================

        self.info_label = tk.Label(
            root,
            text="",
            font=("Arial", 12),
        )

        self.info_label.pack(
            pady=5
        )

        self.agent_label = tk.Label(
            root,
            text="",
            font=("Arial", 11),
        )

        self.agent_label.pack()

        # =====================================================
        # BUTTONS
        # =====================================================

        button_frame = tk.Frame(root)

        button_frame.pack(
            pady=5
        )

        self.step_button = tk.Button(
            button_frame,
            text="Step",
            command=self.step,
        )

        self.step_button.pack(
            side=tk.LEFT,
            padx=5,
        )

        self.run_button = tk.Button(
            button_frame,
            text="Run",
            command=self.run,
        )

        self.run_button.pack(
            side=tk.LEFT,
            padx=5,
        )

        self.reset_button = tk.Button(
            button_frame,
            text="Reset",
            command=self.reset,
        )

        self.reset_button.pack(
            side=tk.LEFT,
            padx=5,
        )

        # =====================================================
        # DRAW INITIAL STATE
        # =====================================================

        self.draw()

    # =========================================================
    # CREATE AGENT
    # =========================================================

    def create_agent(self):

        if self.agent_type == "simple":

            self.agent = SimpleReflexAgent()

        elif self.agent_type == "model":

            self.agent = ModelBasedAgent()

        elif self.agent_type == "search":

            self.agent = SearchAgent(
                self.search_algorithm
            )

        else:

            raise ValueError(
                f"Unknown agent type: {self.agent_type}"
            )

    # =========================================================
    # DRAW
    # =========================================================

    def draw(self):

        self.canvas.delete("all")

        # =====================================================
        # GRID
        # =====================================================

        for y in range(self.height):

            for x in range(self.width):

                x1 = x * self.cell_size
                y1 = y * self.cell_size

                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size

                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    outline="gray",
                )

        # =====================================================
        # WALLS
        # =====================================================

        for x, y in self.env.walls:

            x1 = x * self.cell_size
            y1 = y * self.cell_size

            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size

            self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill="black",
            )

        # =====================================================
        # TOXIC TRAPS
        # =====================================================

        for x, y in self.env.toxic_traps:

            x1 = x * self.cell_size
            y1 = y * self.cell_size

            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size

            self.canvas.create_oval(
                x1 + 10,
                y1 + 10,
                x2 - 10,
                y2 - 10,
                fill="orange",
            )

        # =====================================================
        # FOOD
        # =====================================================

        for x, y in self.env.food_positions:

            center_x = (
                x * self.cell_size
                + self.cell_size // 2
            )

            center_y = (
                y * self.cell_size
                + self.cell_size // 2
            )

            radius = 8

            self.canvas.create_oval(
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
                fill="green",
            )

        # =====================================================
        # OPPONENTS
        # =====================================================

        for x, y in self.env.opponents:

            x1 = x * self.cell_size + 10
            y1 = y * self.cell_size + 10

            x2 = (
                (x + 1) * self.cell_size
                - 10
            )

            y2 = (
                (y + 1) * self.cell_size
                - 10
            )

            self.canvas.create_oval(
                x1,
                y1,
                x2,
                y2,
                fill="red",
            )

        # =====================================================
        # AGENT
        # =====================================================

        x, y = self.env.agent_pos

        center_x = (
            x * self.cell_size
            + self.cell_size // 2
        )

        center_y = (
            y * self.cell_size
            + self.cell_size // 2
        )

        radius = 15

        self.canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            fill="blue",
        )

        # =====================================================
        # DIRECTION INDICATOR
        # =====================================================

        direction = self.env.facing

        if direction == "Up":

            end_x = center_x
            end_y = center_y - 20

        elif direction == "Right":

            end_x = center_x + 20
            end_y = center_y

        elif direction == "Down":

            end_x = center_x
            end_y = center_y + 20

        else:

            end_x = center_x - 20
            end_y = center_y

        self.canvas.create_line(
            center_x,
            center_y,
            end_x,
            end_y,
            width=3,
            arrow=tk.LAST,
        )

        # =====================================================
        # LABELS
        # =====================================================

        algorithm_text = ""

        if self.agent_type == "search":

            algorithm_text = (
                f" | Algorithm: {self.search_algorithm}"
            )

        self.info_label.config(
            text=(
                f"Score: {self.env.score}    "
                f"Steps: {self.env.steps}    "
                f"Food remaining: "
                f"{len(self.env.food_positions)}"
            )
        )

        self.agent_label.config(
            text=(
                f"Agent: {self.agent_type.upper()}"
                f"{algorithm_text}"
                f" | Facing: {self.env.facing}"
            )
        )

    # =========================================================
    # STEP
    # =========================================================

    def step(self):

        if self.env.is_done():

            self.info_label.config(
                text=(
                    f"Game completed! "
                    f"Final score: {self.env.score}"
                )
            )

            return

        # =====================================================
        # GET PERCEPT
        # =====================================================

        percept = self.env.get_percept()

        # =====================================================
        # AGENT DECIDES ACTION
        # =====================================================

        action = self.agent.sense_and_act(
            percept
        )

        # =====================================================
        # EXECUTE ACTION
        # =====================================================

        if action != "NoOp":

            self.env.execute_action(
                action
            )

        # =====================================================
        # REDRAW GUI
        # =====================================================

        self.draw()

    # =========================================================
    # RUN
    # =========================================================

    def run(self):

        if self.env.is_done():

            return

        self.step()

        if not self.env.is_done():

            self.root.after(
                150,
                self.run,
            )

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.env = VisualGridHuntGame(
            width=self.width,
            height=self.height,
            num_food=10,
            num_opponents=0,
        )

        self.create_agent()

        self.draw()


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    # =========================================================
    # PRACTICAL 04 CONFIGURATION
    # =========================================================

    # Agent:
    #
    # "simple" = Simple Reflex Agent
    # "model"  = Model-Based Agent
    # "search" = Search Agent
    #
    selected_agent = "search"

    # =========================================================
    # SEARCH ALGORITHM
    # =========================================================
    #
    # "BFS"   = Breadth-First Search
    # "DFS"   = Depth-First Search
    # "UCS"   = Uniform-Cost Search
    # "AStar" = A* Search
    #
    selected_algorithm = "AStar"

    # =========================================================
    # START GUI
    # =========================================================

    root = tk.Tk()

    app = GridGameGUI(
        root,
        width=10,
        height=10,
        cell_size=50,
        agent_type=selected_agent,
        search_algorithm=selected_algorithm,
    )

    root.mainloop()