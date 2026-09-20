from collections import deque
import heapq


class SearchAgent:
    """
    Goal-Based Search Agent for Practical 03.

    Supports:
        - BFS (Breadth-First Search)
        - DFS (Depth-First Search)
        - UCS (Uniform-Cost Search)

    The agent creates an offline plan and then
    executes one action from that plan at a time.
    """

    DIRECTIONS = ["Up", "Right", "Down", "Left"]

    DELTAS = {
        "Up": (0, 1),
        "Right": (1, 0),
        "Down": (0, -1),
        "Left": (-1, 0),
    }

    def __init__(self, algorithm="BFS"):
        self.plan = []
        self.active_algo = algorithm.upper()

        # Current estimated position.
        # The environment starts the agent at (0, 0).
        self.current_position = (0, 0)

        # Current facing direction.
        self.current_facing = "Right"

    # ==========================================================
    # Helper Methods
    # ==========================================================

    def _get_neighbors(self, state, grid_size, walls):
        """
        Return valid neighboring states.

        Each result is:
            (next_state, direction, cost)
        """

        x, y = state
        width, height = grid_size

        neighbors = []

        for direction in self.DIRECTIONS:
            dx, dy = self.DELTAS[direction]

            next_state = (
                x + dx,
                y + dy,
            )

            nx, ny = next_state

            # Check grid boundaries
            if nx < 0 or nx >= width:
                continue

            if ny < 0 or ny >= height:
                continue

            # Check walls
            if next_state in walls:
                continue

            neighbors.append(
                (
                    next_state,
                    direction,
                    1,
                )
            )

        return neighbors

    # ==========================================================
    # BFS
    # ==========================================================

    def bfs_search(
        self,
        start,
        goal,
        grid_size,
        walls,
    ):
        """
        Breadth-First Search.

        Uses FIFO queue.
        Finds the shallowest path first.
        """

        queue = deque()

        queue.append(
            (
                start,
                [],
            )
        )

        # Prevent repeated states
        reached = {start}

        while queue:

            state, path = queue.popleft()

            # Goal reached
            if state == goal:
                return path

            for next_state, direction, cost in self._get_neighbors(
                state,
                grid_size,
                walls,
            ):

                if next_state in reached:
                    continue

                reached.add(next_state)

                new_path = path + [direction]

                queue.append(
                    (
                        next_state,
                        new_path,
                    )
                )

        # No solution
        return []

    # ==========================================================
    # DFS
    # ==========================================================

    def dfs_search(
        self,
        start,
        goal,
        grid_size,
        walls,
    ):
        """
        Depth-First Search.

        Uses LIFO stack.
        Explores the deepest available path first.
        """

        stack = []

        stack.append(
            (
                start,
                [],
            )
        )

        # Prevent repeated states
        reached = {start}

        while stack:

            state, path = stack.pop()

            # Goal reached
            if state == goal:
                return path

            neighbors = self._get_neighbors(
                state,
                grid_size,
                walls,
            )

            # Reverse so the first direction is explored first
            for next_state, direction, cost in reversed(
                neighbors
            ):

                if next_state in reached:
                    continue

                reached.add(next_state)

                new_path = path + [direction]

                stack.append(
                    (
                        next_state,
                        new_path,
                    )
                )

        # No solution
        return []

    # ==========================================================
    # UCS
    # ==========================================================

    def ucs_search(
        self,
        start,
        goal,
        grid_size,
        walls,
    ):
        """
        Uniform-Cost Search.

        Uses a priority queue ordered by total path cost g(n).
        """

        priority_queue = []

        counter = 0

        heapq.heappush(
            priority_queue,
            (
                0,
                counter,
                start,
                [],
            ),
        )

        reached_cost = {
            start: 0
        }

        while priority_queue:

            cost, _, state, path = heapq.heappop(
                priority_queue
            )

            # Ignore an outdated queue entry
            if cost > reached_cost.get(
                state,
                float("inf"),
            ):
                continue

            # Goal reached
            if state == goal:
                return path

            for next_state, direction, step_cost in self._get_neighbors(
                state,
                grid_size,
                walls,
            ):

                new_cost = cost + step_cost

                old_cost = reached_cost.get(
                    next_state,
                    float("inf"),
                )

                if new_cost < old_cost:

                    reached_cost[next_state] = new_cost

                    counter += 1

                    new_path = path + [direction]

                    heapq.heappush(
                        priority_queue,
                        (
                            new_cost,
                            counter,
                            next_state,
                            new_path,
                        ),
                    )

        # No solution
        return []

    # ==========================================================
    # Select Search Algorithm
    # ==========================================================

    def search(
        self,
        start,
        goal,
        grid_size,
        walls,
    ):
        """
        Run the selected search algorithm.
        """

        algorithm = self.active_algo.upper()

        if algorithm == "BFS":
            return self.bfs_search(
                start,
                goal,
                grid_size,
                walls,
            )

        if algorithm == "DFS":
            return self.dfs_search(
                start,
                goal,
                grid_size,
                walls,
            )

        if algorithm == "UCS":
            return self.ucs_search(
                start,
                goal,
                grid_size,
                walls,
            )

        raise ValueError(
            f"Unknown search algorithm: {algorithm}"
        )

    # ==========================================================
    # Find Closest Food
    # ==========================================================

    def _find_closest_food(
        self,
        start,
        food_positions,
        grid_size,
        walls,
    ):
        """
        Find a reachable food location.

        BFS is used to determine the shortest distance
        to each food location.
        """

        best_food = None
        best_path = None

        for food in food_positions:

            path = self.bfs_search(
                start,
                food,
                grid_size,
                walls,
            )

            # Food is unreachable
            if not path and food != start:
                continue

            if best_path is None or len(path) < len(best_path):

                best_food = food
                best_path = path

        return best_food, best_path

    # ==========================================================
    # Convert Directions into Game Actions
    # ==========================================================

    def _direction_to_actions(
        self,
        directions,
    ):
        """
        Convert absolute directions such as:

            Up
            Right
            Down
            Left

        into the environment actions:

            TurnLeft
            TurnRight
            MoveForward
        """

        actions = []

        facing = self.current_facing

        for target_direction in directions:

            current_index = self.DIRECTIONS.index(
                facing
            )

            target_index = self.DIRECTIONS.index(
                target_direction
            )

            difference = (
                target_index - current_index
            ) % 4

            # No turn required
            if difference == 0:
                pass

            # Turn right
            elif difference == 1:
                actions.append("TurnRight")
                facing = target_direction

            # Turn around
            elif difference == 2:
                actions.append("TurnRight")
                actions.append("TurnRight")
                facing = target_direction

            # Turn left
            elif difference == 3:
                actions.append("TurnLeft")
                facing = target_direction

            # Move forward
            actions.append("MoveForward")

        return actions

    # ==========================================================
    # Sense and Act
    # ==========================================================

    def sense_and_act(self, percept):
        """
        Main agent loop.

        When the plan is empty:
            1. Read world model.
            2. Find food.
            3. Run selected search algorithm.
            4. Store the plan.

        Then execute one action from the plan.
        """

        # ------------------------------------------------------
        # If food is already here
        # ------------------------------------------------------

        if percept.get("food_here", False):

            self.plan = []

            self.current_position = self.current_position

            return "Suck"

        # ------------------------------------------------------
        # Update current facing from percept
        # ------------------------------------------------------

        if "facing" in percept:
            self.current_facing = percept["facing"]

        # ------------------------------------------------------
        # Create a new offline plan when required
        # ------------------------------------------------------

        if not self.plan:

            grid_size = percept.get(
                "grid_size"
            )

            walls = set(
                percept.get(
                    "walls",
                    [],
                )
            )

            all_food = list(
                percept.get(
                    "all_food",
                    [],
                )
            )

            # No food remaining
            if not all_food:
                return "Suck"

            # Find closest food
            target_food, _ = self._find_closest_food(
                self.current_position,
                all_food,
                grid_size,
                walls,
            )

            # No reachable food
            if target_food is None:
                return "TurnRight"

            # Run selected search algorithm
            path = self.search(
                self.current_position,
                target_food,
                grid_size,
                walls,
            )

            # Convert search path to physical actions
            self.plan = self._direction_to_actions(
                path
            )

        # ------------------------------------------------------
        # Execute next planned action
        # ------------------------------------------------------

        if self.plan:

            action = self.plan.pop(0)

            # Update internal model after movement
            if action == "TurnLeft":

                index = self.DIRECTIONS.index(
                    self.current_facing
                )

                self.current_facing = self.DIRECTIONS[
                    (index - 1) % 4
                ]

            elif action == "TurnRight":

                index = self.DIRECTIONS.index(
                    self.current_facing
                )

                self.current_facing = self.DIRECTIONS[
                    (index + 1) % 4
                ]

            elif action == "MoveForward":

                dx, dy = self.DELTAS[
                    self.current_facing
                ]

                self.current_position = (
                    self.current_position[0] + dx,
                    self.current_position[1] + dy,
                )

            return action

        return "Suck"
