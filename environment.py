"""
environment.py
--------------
A from-scratch implementation of the Frozen Lake grid-world environment.

No external RL frameworks (Gymnasium, OpenAI Gym, etc.) are used anywhere
in this file. The environment is a pure Python class that maintains its
own state, transition logic, and reward structure.

Grid legend:
    S : Start state
    F : Frozen (safe) state
    H : Hole (terminal, failure)
    G : Goal (terminal, success)

Actions:
    0 = Left
    1 = Down
    2 = Right
    3 = Up
"""

import random


DEFAULT_MAP = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

# Action constants for readability throughout the codebase.
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
ACTION_NAMES = {LEFT: "Left", DOWN: "Down", RIGHT: "Right", UP: "Up"}
ACTION_SYMBOLS = {LEFT: "\u2190", DOWN: "\u2193", RIGHT: "\u2192", UP: "\u2191"}

# (row_delta, col_delta) for each action.
ACTION_DELTAS = {
    LEFT: (0, -1),
    DOWN: (1, 0),
    RIGHT: (0, 1),
    UP: (-1, 0),
}


class FrozenLakeEnv:
    """
    A custom Frozen Lake environment built entirely from first principles.

    Parameters
    ----------
    grid_map : list[str], optional
        Rows of the grid as strings made of {S, F, H, G}. Defaults to the
        standard 8x8 map given in the assignment.
    is_slippery : bool, optional
        If True, the environment becomes stochastic: with some probability
        the agent's action is replaced by a random perpendicular action,
        simulating slipping on ice (Bonus Option A). Defaults to False
        (deterministic transitions).
    slip_prob : float, optional
        Probability that the intended action "slips" into a different
        action when is_slippery=True. Defaults to 1/3, matching the
        classic Frozen Lake convention (1/3 intended, 1/3 each side-slip).
    step_penalty : float, optional
        Small negative reward applied on every non-terminal step, used to
        encourage shorter paths. Defaults to 0.0 (sparse reward).
    """

    def __init__(self, grid_map=None, is_slippery=False, slip_prob=1.0 / 3.0,
                 step_penalty=0.0):
        self.grid_map = grid_map if grid_map is not None else DEFAULT_MAP
        self.n_rows = len(self.grid_map)
        self.n_cols = len(self.grid_map[0])
        self.is_slippery = is_slippery
        self.slip_prob = slip_prob
        self.step_penalty = step_penalty

        self.n_states = self.n_rows * self.n_cols
        self.n_actions = 4

        # Locate start, holes, and goal from the map.
        self.start_state = None
        self.hole_states = set()
        self.goal_state = None

        for r, row in enumerate(self.grid_map):
            for c, cell in enumerate(row):
                state_idx = self._to_index(r, c)
                if cell == "S":
                    self.start_state = state_idx
                elif cell == "H":
                    self.hole_states.add(state_idx)
                elif cell == "G":
                    self.goal_state = state_idx

        if self.start_state is None or self.goal_state is None:
            raise ValueError("Grid map must contain exactly one 'S' and one 'G'.")

        self.current_state = self.start_state
        self.done = False

    # ------------------------------------------------------------------ #
    # Coordinate <-> index helpers
    # ------------------------------------------------------------------ #
    def _to_index(self, row, col):
        return row * self.n_cols + col

    def _to_coords(self, state_idx):
        return divmod(state_idx, self.n_cols)

    # ------------------------------------------------------------------ #
    # Core API required by the assignment
    # ------------------------------------------------------------------ #
    def reset(self):
        """
        Reset the environment to the start state.

        Returns
        -------
        int : the starting state index.
        """
        self.current_state = self.start_state
        self.done = False
        return self.current_state

    def step(self, action):
        """
        Apply an action to the environment and advance one time step.

        Parameters
        ----------
        action : int
            One of {0: Left, 1: Down, 2: Right, 3: Up}.

        Returns
        -------
        next_state : int
        reward : float
        done : bool
        info : dict
        """
        if self.done:
            # Once terminal, further steps are no-ops returning the same state.
            return self.current_state, 0.0, True, {"reason": "episode_already_terminal"}

        actual_action = self._resolve_action(action)
        row, col = self._to_coords(self.current_state)
        d_row, d_col = ACTION_DELTAS[actual_action]
        new_row = row + d_row
        new_col = col + d_col

        # Enforce movement boundaries: if the move would leave the grid,
        # the agent stays in place (bumps into the wall).
        if 0 <= new_row < self.n_rows and 0 <= new_col < self.n_cols:
            next_state = self._to_index(new_row, new_col)
        else:
            next_state = self.current_state  # blocked by wall

        self.current_state = next_state

        reward, done = self._compute_reward_and_done(next_state)
        self.done = done

        info = {
            "intended_action": action,
            "actual_action": actual_action,
            "slipped": actual_action != action,
        }
        return next_state, reward, done, info

    def render(self):
        """
        Print a human-readable view of the grid with the agent's
        current position marked as 'A' (unless on the goal or a hole,
        where the original symbol is kept for clarity).
        """
        row, col = self._to_coords(self.current_state)
        lines = []
        for r in range(self.n_rows):
            line_chars = []
            for c in range(self.n_cols):
                cell = self.grid_map[r][c]
                if (r, c) == (row, col) and cell == "F":
                    line_chars.append("A")
                elif (r, c) == (row, col) and cell == "S":
                    line_chars.append("A")
                else:
                    line_chars.append(cell)
            lines.append("".join(line_chars))
        print("\n".join(lines))
        print()  # trailing blank line for readability
        return "\n".join(lines)

    def get_state(self):
        """Return the current state index."""
        return self.current_state

    def is_terminal(self, state=None):
        """
        Return True if the given state (or current state if None) is
        terminal, i.e. a Hole or the Goal.
        """
        s = self.current_state if state is None else state
        return s in self.hole_states or s == self.goal_state

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _resolve_action(self, action):
        """
        If is_slippery is True, the intended action may be replaced by a
        perpendicular action with probability self.slip_prob, simulating
        slipping on ice (Bonus Option A: stochastic transitions). The
        remaining probability mass keeps the intended action.
        """
        if not self.is_slippery:
            return action

        if random.random() < self.slip_prob:
            # Slip sideways: pick one of the two actions perpendicular
            # to the intended direction.
            perpendicular = {
                LEFT: [UP, DOWN],
                RIGHT: [UP, DOWN],
                UP: [LEFT, RIGHT],
                DOWN: [LEFT, RIGHT],
            }[action]
            return random.choice(perpendicular)
        return action

    def _compute_reward_and_done(self, state):
        """
        Reward structure:
            +1.0  for reaching the Goal
            -1.0  for falling into a Hole
            step_penalty (default 0.0) for any other (non-terminal) move
        """
        if state == self.goal_state:
            return 1.0, True
        if state in self.hole_states:
            return -1.0, True
        return self.step_penalty, False

    # ------------------------------------------------------------------ #
    # Convenience accessors used by agent/training/evaluation scripts
    # ------------------------------------------------------------------ #
    @property
    def shape(self):
        return (self.n_rows, self.n_cols)

    def state_to_coords(self, state):
        return self._to_coords(state)

    def coords_to_state(self, row, col):
        return self._to_index(row, col)
