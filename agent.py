from collections import deque
import heapq
import math


class SearchAgent:
    """
    Goal-Based Search Agent for Practical 03 and Practical 04.

    Supports:
        - BFS (Breadth-First Search)
        - DFS (Depth-First Search)
        - UCS (Uniform-Cost Search)
        - AStar (A* Informed Search)

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
    # Practical 04 - Manhattan Distance
    # ==========================================================

    def manhattan_distance(self, pos, goal):
        """
        Calculate Manhattan Distance.

        Formula:
            h(n) = |x1 - x2| + |y1 - y2|

        This is suitable for a 4-way movement grid.
        """

        x1, y1 = pos
        x2, y2 = goal

        return abs(x1 - x2) + abs(y1 - y2)

    # ==========================================================
    # Practical 04 - Euclidean Distance
    # ==========================================================

    def euclidean_distance(self, pos, goal):
        """
        Calculate Euclidean Distance.

        Formula:
            h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2)
        """

        x1, y1 = pos
        x2, y2 = goal

        return math.sqrt(
            (x1 - x2) ** 2
            + (y1 - y2) ** 2
        )

    # ==========================================================
    # Practical 04 - A* Search
    # ==========================================================

    def astar_search(
        self,
        start_pos,
        goal_pos,
        walls,
        grid_size,
        heuristic_type="manhattan",
    ):
        """
        A* Search.

        Evaluation function:

            f(n) = g(n) + h(n)

        where:
            g(n) = cost from start to current node
            h(n) = estimated cost from current node to goal
            f(n) = total estimated cost
        """

        priority_queue = []

        reached_states = set()

        counter = 0

        # ------------------------------------------------------
        # Calculate initial heuristic
        # ------------------------------------------------------

        if heuristic_type.lower() == "euclidean":

            h_start = self.euclidean_distance(
                start_pos,
                goal_pos,
            )

        else:

            h_start = self.manhattan_distance(
                start_pos,
                goal_pos,
            )

        # Initial path cost
        g_start = 0

        # A* evaluation
        f_start = g_start + h_start

        # ------------------------------------------------------
        # Add starting node
        # ------------------------------------------------------

        heapq.heappush(
            priority_queue,
            (
                f_start,
                g_start,
                counter,
                start_pos,
                [],
            ),
        )

        # ------------------------------------------------------
        # Search loop
        # ------------------------------------------------------

        while priority_queue:

            (
                f_cost,
                g_cost,
                _,
                current_pos,
                path_taken,
            ) = heapq.heappop(
                priority_queue
            )

            # --------------------------------------------------
            # Goal reached
            # --------------------------------------------------

            if current_pos == goal_pos:
                return path_taken

            # --------------------------------------------------
            # Skip already explored states
            # --------------------------------------------------

            if current_pos in reached_states:
                continue

            reached_states.add(current_pos)

            # --------------------------------------------------
            # Expand neighboring states
            # --------------------------------------------------

            for (
                next_pos,
                direction,
                step_cost,
            ) in self._get_neighbors(
                current_pos,
                grid_size,
                walls,
            ):

                # Skip already reached states
                if next_pos in reached_states:
                    continue

                # New path cost
                new_g = g_cost + step_cost

                # --------------------------------------------------
                # Calculate heuristic
                # --------------------------------------------------

                if heuristic_type.lower() == "euclidean":

                    new_h = self.euclidean_distance(
                        next_pos,
                        goal_pos,
                    )

                else:

                    new_h = self.manhattan_distance(
                        next_pos,
                        goal_pos,
                    )

                # --------------------------------------------------
                # Calculate f(n)
                # --------------------------------------------------

                new_f = new_g + new_h

                # --------------------------------------------------
                # Create new path
                # --------------------------------------------------

                new_path = path_taken + [direction]

                counter += 1

                # --------------------------------------------------
                # Add node to priority queue
                # --------------------------------------------------

                heapq.heappush(
                    priority_queue,
                    (
                        new_f,
                        new_g,
                        counter,
                        next_pos,
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

        if algorithm in ("ASTAR", "A*"):

            return self.astar_search(
                start_pos=start,
                goal_pos=goal,
                walls=walls,
                grid_size=grid_size,
                heuristic_type="manhattan",
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

            if (
                best_path is None
                or len(path) < len(best_path)
            ):

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

            # --------------------------------------------------
            # Find closest food
            # --------------------------------------------------

            target_food, _ = self._find_closest_food(
                self.current_position,
                all_food,
                grid_size,
                walls,
            )

            # No reachable food
            if target_food is None:
                return "TurnRight"

            # --------------------------------------------------
            # Run selected search algorithm
            # --------------------------------------------------

            if self.active_algo.upper() in (
                "ASTAR",
                "A*",
            ):

                path = self.astar_search(
                    start_pos=self.current_position,
                    goal_pos=target_food,
                    walls=walls,
                    grid_size=grid_size,
                    heuristic_type="manhattan",
                )

            else:

                path = self.search(
                    self.current_position,
                    target_food,
                    grid_size,
                    walls,
                )

            # --------------------------------------------------
            # Convert search path to physical actions
            # --------------------------------------------------

            self.plan = self._direction_to_actions(
                path
            )

        # ------------------------------------------------------
        # Execute next planned action
        # ------------------------------------------------------

        if self.plan:

            action = self.plan.pop(0)

            # --------------------------------------------------
            # Turn Left
            # --------------------------------------------------

            if action == "TurnLeft":

                index = self.DIRECTIONS.index(
                    self.current_facing
                )

                self.current_facing = self.DIRECTIONS[
                    (index - 1) % 4
                ]

            # --------------------------------------------------
            # Turn Right
            # --------------------------------------------------

            elif action == "TurnRight":

                index = self.DIRECTIONS.index(
                    self.current_facing
                )

                self.current_facing = self.DIRECTIONS[
                    (index + 1) % 4
                ]

            # --------------------------------------------------
            # Move Forward
            # --------------------------------------------------

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