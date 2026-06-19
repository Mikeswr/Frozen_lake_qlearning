"""
evaluate.py
-----------
Loads the best trained Q-table and evaluates the greedy policy over a
number of episodes (>= 100, as required), reporting:
    - Success Rate (%)
    - Average Reward
    - Number of Failures
    - Number of Successful Runs

Also displays the final extracted policy in grid form.

Run:
    python evaluate.py
"""

import os
import json
import numpy as np

from environment import FrozenLakeEnv
from agent import QLearningAgent
from utils import print_policy

RESULTS_DIR = "results"
QTABLE_PATH = os.path.join(RESULTS_DIR, "qtable_best.npy")


def evaluate_policy(env, agent, n_episodes=200, max_steps=200):
    """
    Run the greedy (exploitation-only) policy for n_episodes and
    collect evaluation statistics.
    """
    rewards = []
    successes = 0
    failures = 0

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.select_action(state, greedy=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
        if state == env.goal_state:
            successes += 1
        else:
            failures += 1

    return {
        "n_episodes": n_episodes,
        "success_rate": successes / n_episodes * 100,
        "average_reward": float(np.mean(rewards)),
        "n_failures": failures,
        "n_successes": successes,
    }


def main():
    if not os.path.exists(QTABLE_PATH):
        raise FileNotFoundError(
            f"Could not find '{QTABLE_PATH}'. Run `python train.py` first."
        )

    env = FrozenLakeEnv(is_slippery=False)
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
    agent.q_table = np.load(QTABLE_PATH)

    n_eval_episodes = 200  # >= 100 as required by the assignment
    results = evaluate_policy(env, agent, n_episodes=n_eval_episodes)

    print("================ EVALUATION RESULTS ================")
    print(f"Episodes evaluated   : {results['n_episodes']}")
    print(f"Success Rate (%)     : {results['success_rate']:.2f}")
    print(f"Average Reward       : {results['average_reward']:.4f}")
    print(f"Number of Failures   : {results['n_failures']}")
    print(f"Number of Successes  : {results['n_successes']}")

    print("\n================ LEARNED POLICY (Greedy) ================")
    policy = agent.get_policy()
    print_policy(env, policy)

    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved evaluation results -> "
          f"{os.path.join(RESULTS_DIR, 'evaluation_results.json')}")


if __name__ == "__main__":
    main()
