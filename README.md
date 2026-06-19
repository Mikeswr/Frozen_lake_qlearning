# Frozen Lake from First Principles Using Q-Learning

A complete Reinforcement Learning solution to the classic Frozen Lake
grid-world problem, implemented **entirely from scratch in Python** —
no Gymnasium, OpenAI Gym, Stable-Baselines, RLlib, or any other RL
framework is used anywhere in this repository.

DCIT 614 — Reinforcement Learning, University of Ghana, Department of
Computer Science (Semester II, 2025/2026).

---

## 1. Introduction

### What is Reinforcement Learning?

Reinforcement Learning (RL) is a branch of machine learning in which an
**agent** learns to make decisions by interacting with an **environment**.
At each time step the agent observes a **state**, chooses an **action**,
and receives a **reward** signal along with a new state. Over many such
interactions, the agent updates its behaviour — its **policy** — to
maximize the total (discounted) reward it accumulates over time. Unlike
supervised learning, RL agents are not told the correct action directly;
they must discover good behaviour through trial and error, balancing
**exploration** (trying new actions to gather information) against
**exploitation** (using what has already been learned to act well).

### What is Frozen Lake?

Frozen Lake is a classic grid-world benchmark used to study RL
algorithms. An agent starts at a fixed cell on a frozen lake represented
as a grid and must reach a goal cell without falling into any holes in
the ice. Each cell is one of:

- **S** — Start state
- **F** — Frozen (safe) state the agent can walk on
- **H** — Hole — stepping here ends the episode in failure
- **G** — Goal — stepping here ends the episode in success

The agent can move Left, Down, Right, or Up. Because the goal can be
many steps away and obstacles must be avoided, the agent must learn a
multi-step plan purely from the reward signal it receives — making it a
good, small testbed for tabular RL algorithms such as Q-Learning.

---

## 2. Environment Design

The environment is implemented in `environment.py` as the
`FrozenLakeEnv` class, built from first principles with no external RL
library.

### State Representation

States are represented as **single integer indices** in
`[0, n_rows * n_cols - 1]`, computed from `(row, col)` via
`state = row * n_cols + col`. This keeps the Q-table a simple 2D array
of shape `(n_states, n_actions)`. Helper methods `state_to_coords()`
and `coords_to_state()` convert between the two representations
whenever needed (e.g. for rendering).

### Action Representation

Four discrete actions, matching the assignment specification:

| Action | Value |
|--------|-------|
| Left   | 0     |
| Down   | 1     |
| Right  | 2     |
| Up     | 3     |

Moving into a wall (off the grid) leaves the agent in place rather than
wrapping around or erroring — this is enforced directly in `step()`.

### Reward Structure

| Outcome                | Reward |
|-------------------------|--------|
| Reaching the Goal (G)   | +1.0   |
| Falling into a Hole (H) | -1.0   |
| Any other move          | 0.0    |

This is the classic *sparse* Frozen Lake reward signal: the agent gets
no feedback at all until the episode actually ends, which makes
discovering the goal via random exploration alone non-trivial — a key
reason Q-Learning's bootstrapped value propagation is useful here.

### Core API

`FrozenLakeEnv` implements the required interface:

- `reset()` — returns the agent to the start state, returns the start
  state index.
- `step(action)` — applies an action, returns `(next_state, reward,
  done, info)`.
- `render()` — prints a text view of the grid with the agent's current
  position.
- `get_state()` — returns the current state index.
- `is_terminal(state=None)` — returns whether a state (default: current
  state) is a Hole or the Goal.

---

## 3. Q-Learning Algorithm

### Description

Q-Learning is a **model-free, off-policy** RL algorithm that learns the
value of taking a given action in a given state, denoted `Q(s, a)`,
without needing to know the environment's transition probabilities in
advance. It maintains a table of Q-values for every (state, action)
pair and updates that table after every step using the agent's own
experience.

### The Update Equation

Implemented exactly as specified, in `agent.py`:

```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
```

Where:
- `alpha` (learning rate) controls how much new information overrides
  old estimates.
- `gamma` (discount factor) controls how much future rewards are valued
  relative to immediate ones.
- `max_a' Q(s', a')` is the agent's current best estimate of the value
  of the next state, used as a bootstrapped target. This term is `0`
  when the episode has terminated, since there is no next action.

### Exploration Strategy

The agent uses **epsilon-greedy** exploration: with probability
`epsilon` it picks a uniformly random action (explore), and otherwise
picks the action with the highest current Q-value for that state
(exploit), breaking ties randomly to avoid directional bias.

`epsilon` starts at `1.0` (fully random / pure exploration) and decays
multiplicatively after every episode:

```
epsilon <- max(epsilon_min, epsilon * epsilon_decay)
```

until it reaches a small floor (`epsilon_min = 0.01`), ensuring the
agent keeps a small amount of exploration even late in training.

---

## 4. Training Procedure

Training is run from `train.py`, which:

1. Builds a `FrozenLakeEnv` and a `QLearningAgent`.
2. Runs the agent for `n_episodes` episodes, each capped at
   `max_steps` steps.
3. At every step: selects an action (epsilon-greedy), applies it to the
   environment, updates the Q-table, then moves to the next state.
4. Decays epsilon once per completed episode.
5. Records reward, success/failure, epsilon, and step count for every
   episode.
6. Repeats this for **four different hyperparameter configurations**
   (Part C requires experimenting with learning rate, discount factor,
   and exploration rate) and saves the best-performing Q-table.

### Hyperparameters Used

| Experiment       | alpha (LR) | gamma (discount) | epsilon decay | Episodes |
|-------------------|-----------|-------------------|----------------|----------|
| `baseline`        | 0.10      | 0.99              | 0.9995         | 10,000   |
| `high_alpha`      | 0.50      | 0.99              | 0.9995         | 10,000   |
| `low_gamma`       | 0.10      | 0.90              | 0.9995         | 10,000   |
| `fast_eps_decay`  | 0.10      | 0.99              | 0.9990         | 10,000   |

All other settings are shared: `epsilon_start = 1.0`, `epsilon_min =
0.01`, `max_steps = 200` per episode, deterministic (non-slippery)
transitions, seed = 42 for reproducibility.

---

## 5. Results

Results below are from the most recent run (`results/experiment_summary.json`
and `results/evaluation_results.json` contain the exact numbers — these
can vary slightly between runs since Q-Learning is stochastic).

### Hyperparameter Comparison (success rate over all 10,000 training episodes)

| Experiment       | Success Rate (training) |
|-------------------|--------------------------|
| baseline          | ~82.6%                   |
| high_alpha        | ~77.6%                   |
| low_gamma         | ~80.2%                   |
| **fast_eps_decay**| **~88.7% (best)**        |

The `fast_eps_decay` configuration converged fastest because epsilon
dropped to its floor earlier, giving the agent more episodes to exploit
an already-decent policy rather than continuing to explore randomly.

### Final Evaluation (greedy policy, 200 episodes, no exploration)

| Metric              | Value   |
|----------------------|---------|
| Success Rate (%)     | 100.00  |
| Average Reward       | 1.0000  |
| Number of Failures   | 0       |
| Number of Successes  | 200     |

Once exploration is turned off and the agent acts purely greedily on
its learned Q-table, it reaches the goal in every evaluation episode.

### Learned Policy (greedy, arrows show the best action per state)

```
→ → ↓ ↓ ↓ ↓ ← ↓
→ → → → → ↓ ↓ ↓
↑ ↑ ↑ H ↑ → → ↓
↑ → ↑ H ↑ ↑ ↓ ↓
↑ ↑ ← H → → → ↓
← H H ← ↑ ↑ H ↓
← H ↓ ← H ↑ H ↓
← ← ← H ← ← ← G
```

`H` marks holes and `G` marks the goal; every other cell shows the
action the agent learned is best from that cell.

### Discussion

- The reward curve (see `results/training_curves_fast_eps_decay.png`)
  shows the classic Q-Learning shape: a slow, noisy start while epsilon
  is high and the agent is mostly acting randomly, then a steep climb
  as the Q-table starts encoding a path to the goal, then a plateau
  near 100% success once the policy has converged.
- A higher learning rate (`high_alpha`) made training *noisier* and
  slightly slower to converge — large updates can overwrite useful
  Q-values before they stabilize.
- A lower discount factor (`low_gamma`) made the agent slightly less
  willing to plan far ahead, since rewards far in the future are
  discounted more heavily, but the 8x8 map is small enough that this
  had only a modest effect.
- Faster epsilon decay helped here because the state space is small
  (64 states); the agent did not need as much exploration time to
  cover it.

---

## 6. Bonus Task Implemented

**Option B — Visualize training performance using graphs.**

`train.py` automatically produces and saves, for every hyperparameter
configuration, a three-panel figure to `results/training_curves_<tag>.png`
containing:

1. Episode reward over training (raw + 100-episode rolling average).
2. Rolling success rate (100-episode window).
3. Epsilon decay over training.

The environment also supports **Option A (stochastic transitions)** as
an optional feature — pass `is_slippery=True` to `FrozenLakeEnv` to
enable slipping, where the intended action is replaced by a
perpendicular action with probability `slip_prob` (default 1/3) — but
this is not enabled by default since Option B was selected as the
primary bonus deliverable.

---

## 7. Execution Instructions

### Requirements

```bash
pip install -r requirements.txt
```

### Train the agent

Runs all four hyperparameter experiments, prints progress, saves
Q-tables, training-curve plots, and a JSON summary to `results/`:

```bash
python train.py
```

### Evaluate the best trained agent

Loads `results/qtable_best.npy` and runs 200 greedy evaluation
episodes, printing the success rate, average reward, failures,
successes, and the final learned policy grid:

```bash
python evaluate.py
```

### Project Structure

```
frozen-lake-qlearning/
├── environment.py     # FrozenLakeEnv: state, transitions, rewards
├── agent.py            # QLearningAgent: Q-table, epsilon-greedy, updates
├── utils.py             # Policy-to-grid rendering helper
├── train.py             # Training loop + hyperparameter experiments + plots
├── evaluate.py          # Evaluation loop (>=100 episodes) + policy display
├── requirements.txt
├── README.md
├── report.pdf
└── results/             # Q-tables, plots, and JSON summaries (generated)
```

---

## Author

Name: _______________________
Student ID: _______________________
