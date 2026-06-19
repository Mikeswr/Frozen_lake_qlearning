"""
agent.py
--------
A from-scratch tabular Q-Learning agent.

Implements:
    - Q-table initialization
    - Epsilon-greedy action selection
    - Epsilon decay
    - The Q-Learning update rule:
          Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
"""

import random
import numpy as np


class QLearningAgent:
    """
    Tabular Q-Learning agent.

    Parameters
    ----------
    n_states : int
        Number of discrete states in the environment.
    n_actions : int
        Number of discrete actions available to the agent.
    alpha : float
        Learning rate (step size for Q-table updates).
    gamma : float
        Discount factor for future rewards.
    epsilon_start : float
        Initial exploration rate for epsilon-greedy action selection.
    epsilon_min : float
        Minimum exploration rate epsilon decays to.
    epsilon_decay : float
        Multiplicative decay factor applied to epsilon after each episode
        (epsilon <- max(epsilon_min, epsilon * epsilon_decay)).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.9995,
                 seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Q-table initialized to zeros: shape (n_states, n_actions).
        self.q_table = np.zeros((n_states, n_actions))

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    def select_action(self, state, greedy=False):
        """
        Choose an action using epsilon-greedy exploration.

        Parameters
        ----------
        state : int
            Current state index.
        greedy : bool
            If True, always exploit (used at evaluation time); ignores
            epsilon entirely.

        Returns
        -------
        int : selected action.
        """
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return self._best_action(state)

    def _best_action(self, state):
        """
        Return the action with the highest Q-value for a given state.
        Ties are broken randomly to avoid a fixed directional bias.
        """
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)
        return int(np.random.choice(best_actions))

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def update(self, state, action, reward, next_state, done):
        """
        Apply the Q-Learning update rule:

            Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        If the episode has terminated (done=True), the future value term
        max_a' Q(s', a') is treated as 0, since there is no next action.
        """
        best_next_q = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        """Decay epsilon multiplicatively, floored at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_epsilon(self):
        """Reset epsilon back to its initial value (useful for re-runs)."""
        self.epsilon = self.epsilon_start

    # ------------------------------------------------------------------ #
    # Policy extraction
    # ------------------------------------------------------------------ #
    def get_policy(self):
        """
        Extract the greedy policy from the Q-table.

        Returns
        -------
        np.ndarray of shape (n_states,) with the best action per state.
        """
        return np.argmax(self.q_table, axis=1)

    def get_state_value(self, state):
        """Return V(s) = max_a Q(s,a) for a given state."""
        return np.max(self.q_table[state])
