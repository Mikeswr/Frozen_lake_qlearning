# Frozen Lake from First Principles Using Q-Learning

This is a Reinforcement Learning project that solves the classic Frozen Lake grid world problem. Everything here is built from scratch in Python. No Gymnasium, no OpenAI Gym, no Stable Baselines, no RLlib. Just plain Python, NumPy, and Matplotlib.

DCIT 614, Reinforcement Learning, University of Ghana, Department of Computer Science, Semester II, 2025/2026.

## 1. Introduction

### What is Reinforcement Learning?

Reinforcement Learning is a way for an agent to learn how to act by trying things and seeing what happens. The agent looks at the current state, picks an action, and gets a reward back along with a new state. Over time it figures out which actions lead to good outcomes and which ones don't. Nobody tells it the right answer directly. It has to explore and learn from trial and error, while also using what it already knows to act well when it can.

### What is Frozen Lake?

Frozen Lake is a small grid world used a lot in RL teaching and practice. The agent starts on one cell of the grid and needs to reach a goal cell without falling into any holes. Each cell on the grid is one of these:

* S, the start cell
* F, frozen ground, safe to walk on
* H, a hole, if you step here the episode ends and you fail
* G, the goal, if you reach here the episode ends and you win

The agent can move left, down, right, or up. Since the goal might be far away and there are holes to avoid, the agent has to learn a path using only the reward signal it gets along the way. That makes this a good small problem for testing out an algorithm like Q-Learning.

## 2. Environment Design

The environment lives in `environment.py` as the `FrozenLakeEnv` class. It's written from scratch with no outside RL library.

### State Representation

Each state is just a single number from 0 to 63, worked out from the row and column as `state = row * 8 + col`. This makes the Q-table a simple 2D array shaped like `(64, 4)`. There are helper functions to go back and forth between `(row, col)` and the state number whenever that's needed, like when rendering the grid or showing the policy.

### Action Representation

There are four actions, matching what the assignment asked for:

| Action | Value |
|--------|-------|
| Left   | 0     |
| Down   | 1     |
| Right  | 2     |
| Up     | 3     |

If the agent tries to move off the edge of the grid, it just stays where it is instead of wrapping around or throwing an error. That's handled directly inside `step()`.

### Reward Structure

| Outcome                 | Reward |
|--------------------------|--------|
| Reaching the Goal (G)    | +1.0   |
| Falling into a Hole (H)  | -1.0   |
| Any other move           | 0.0    |

This is the classic sparse reward setup for Frozen Lake. The agent gets nothing at all until the episode actually ends, which makes it harder to find the goal through random moves alone. That's part of why Q-Learning, which carries value information backward through the table over many episodes, is useful here.

### Core API

`FrozenLakeEnv` has the functions the assignment asked for:

* `reset()`, sends the agent back to the start and returns the start state.
* `step(action)`, applies the action and returns `(next_state, reward, done, info)`.
* `render()`, prints a text view of the grid showing where the agent is.
* `get_state()`, returns the current state.
* `is_terminal(state=None)`, tells you if a state (or the current one by default) is a hole or the goal.

## 3. Q-Learning Algorithm

### Description

Q-Learning is a model free, off policy algorithm. That means it learns the value of taking an action in a state without needing to know how the environment works ahead of time. It keeps a table of values, one for every state and action pair, and updates that table after every step based on what actually happened.

### The Update Equation

This is implemented in `agent.py`, exactly as given in the assignment:

```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
```

Here, `alpha` is the learning rate. It controls how much a new experience changes what's already in the table. `gamma` is the discount factor, it controls how much the agent cares about future rewards versus rewards right now. `max_a' Q(s', a')` is the agent's current best guess at how good the next state is. If the episode just ended, this term is set to 0 since there's no next action to take.

### Exploration Strategy

The agent picks actions using epsilon greedy exploration. With probability `epsilon` it picks a random action just to see what happens. Otherwise it picks whichever action currently has the highest Q-value for that state. If there's a tie, it picks randomly between the tied actions so it doesn't always favor one direction by accident.

`epsilon` starts at `1.0`, so early on the agent is basically just exploring randomly. After every episode it shrinks a bit:

```
epsilon <- max(epsilon_min, epsilon * epsilon_decay)
```

This keeps going until epsilon hits a small floor, `epsilon_min = 0.01`, so the agent always keeps a tiny bit of randomness even once it's mostly figured things out.

## 4. Training Procedure

Training happens in `train.py`. Here is what it does:

1. Sets up a `FrozenLakeEnv` and a `QLearningAgent`.
2. Runs the agent for a set number of episodes, each one capped at a max number of steps.
3. At each step, picks an action with epsilon greedy, applies it, updates the Q-table, moves to the next state.
4. Decays epsilon once per episode.
5. Keeps track of reward, success or failure, epsilon, and step count for every episode.
6. Repeats this across four different hyperparameter setups, since the assignment asks for some experimentation with learning rate, discount factor, and exploration rate, then saves whichever Q-table did best.

### Hyperparameters Used

| Experiment       | alpha (learning rate) | gamma (discount) | epsilon decay | Episodes |
|-------------------|------------------------|-------------------|-----------------|----------|
| baseline          | 0.10                   | 0.99              | 0.9995          | 10,000   |
| high_alpha        | 0.50                   | 0.99              | 0.9995          | 10,000   |
| low_gamma         | 0.10                   | 0.90              | 0.9995          | 10,000   |
| fast_eps_decay    | 0.10                   | 0.99              | 0.9990          | 10,000   |

Everything else stayed the same across all four runs: `epsilon_start = 1.0`, `epsilon_min = 0.01`, max 200 steps per episode, no slipping on the ice, and a fixed random seed of 42 so the results can be reproduced.

## 5. Results

These numbers are from the most recent run. You can find the exact figures in `results/experiment_summary.json` and `results/evaluation_results.json`. They might shift slightly if you rerun training since Q-Learning involves randomness.

### Hyperparameter Comparison (training success rate across all 10,000 episodes)

| Experiment       | Success Rate (training) |
|-------------------|---------------------------|
| baseline          | about 82.6%                |
| high_alpha        | about 77.6%                |
| low_gamma         | about 80.2%                |
| fast_eps_decay    | about 88.7% (best one)      |

The fast_eps_decay setup did the best because epsilon dropped to its floor sooner, which gave the agent more episodes to actually use the policy it already had instead of continuing to explore randomly.

### Final Evaluation (greedy policy, 200 episodes, no exploration)

| Metric                | Value   |
|------------------------|---------|
| Success Rate (%)       | 100.00  |
| Average Reward         | 1.0000  |
| Number of Failures     | 0       |
| Number of Successes    | 200     |

Once exploration is turned off and the agent just acts on what it learned, it reaches the goal every single time across all 200 evaluation episodes.

### Learned Policy (greedy, arrows show the best action per state)

```
right right down  down  down  down  left  down
right right right right right down  down  down
up    up    up    H     up    right right down
up    right up    H     up    up    down  down
up    up    left  H     right right right down
left  H     H     left  up    up    H     down
left  H     down  left  H     up    H     down
left  left  left  H     left  left  left  G
```

H marks holes, G marks the goal, and every other cell shows the action the agent decided was best from that spot.

### Discussion

* The reward curve, which you can see in `results/training_curves_fast_eps_decay.png`, shows the usual Q-Learning shape. It starts off slow and noisy while epsilon is high and the agent is mostly guessing, then climbs steeply once the Q-table starts pointing toward the goal, and finally levels off near 100% once the policy settles.
* A higher learning rate (high_alpha) made training noisier and a bit slower to settle down. Big updates can overwrite good Q-values before they have had a chance to stabilize.
* A lower discount factor (low_gamma) made the agent care a little less about planning far ahead, since future rewards get discounted more. The map is small enough that this did not change much though.
* Decaying epsilon faster worked well here since the grid only has 64 states. The agent did not need a ton of exploration time to cover all of it.

## 6. Bonus Task Implemented

Option B, visualizing training performance with graphs.

`train.py` automatically builds and saves a three panel figure for every hyperparameter setup, saved to `results/training_curves_<tag>.png`. It includes:

1. Episode reward over training, both raw and a 100 episode rolling average.
2. Rolling success rate (100 episode window).
3. Epsilon decay over training.

The environment also supports Option A, stochastic transitions, as an extra feature if you want it. Just pass `is_slippery=True` when creating `FrozenLakeEnv` and the intended action will sometimes get swapped for a sideways one, controlled by `slip_prob` (1/3 by default). This is not turned on by default since Option B was the main bonus task here.

## 7. Execution Instructions

### Requirements

```bash
pip install -r requirements.txt
```

### Train the agent

This runs all four hyperparameter experiments, prints progress as it goes, and saves Q-tables, plots, and a JSON summary into `results/`:

```bash
python train.py
```

### Evaluate the best trained agent

This loads `results/qtable_best.npy` and runs 200 greedy evaluation episodes, then prints the success rate, average reward, failures, successes, and the final policy grid:

```bash
python evaluate.py
```

### Project Structure

```
frozen-lake-qlearning/
├── environment.py     # FrozenLakeEnv: state, transitions, rewards
├── agent.py             # QLearningAgent: Q-table, epsilon greedy, updates
├── utils.py             # Policy to grid rendering helper
├── train.py             # Training loop, hyperparameter experiments, plots
├── evaluate.py          # Evaluation loop (200 episodes), policy display
├── requirements.txt
├── README.md
├── report.pdf
└── results/             # Q-tables, plots, and JSON summaries (generated)
```

## Author

Name: Michael Nimako Frempong
Student ID: 22425053
GitHub Repository: [Frozen_lake_qlearning](https://github.com/Mikeswr/Frozen_lake_qlearning)
Repository Link: https://github.com/Mikeswr/Frozen_lake_qlearning
