"""
utils.py
--------
Shared helper functions for rendering the learned policy as a grid and
for other small reusable pieces used by train.py / evaluate.py.
"""

from environment import ACTION_SYMBOLS


def policy_to_grid_string(env, policy):
    """
    Convert a flat policy array into a human-readable grid string,
    using arrow symbols for actions and H/G for terminal states.

    Parameters
    ----------
    env : FrozenLakeEnv
        The environment the policy was learned on (used for map layout).
    policy : array-like of shape (n_states,)
        Greedy action index per state, e.g. from agent.get_policy().

    Returns
    -------
    str : multi-line string representing the policy grid.
    """
    lines = []
    for r in range(env.n_rows):
        row_symbols = []
        for c in range(env.n_cols):
            state = env.coords_to_state(r, c)
            cell = env.grid_map[r][c]
            if cell == "H":
                row_symbols.append("H")
            elif cell == "G":
                row_symbols.append("G")
            else:
                action = policy[state]
                row_symbols.append(ACTION_SYMBOLS[action])
        lines.append(" ".join(row_symbols))
    return "\n".join(lines)


def print_policy(env, policy):
    """Print the policy grid to stdout."""
    print(policy_to_grid_string(env, policy))
