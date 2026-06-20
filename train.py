"""
train.py
--------
Trains the Q-Learning agent on the FrozenLakeEnv from scratch, records
training statistics (episode rewards, success rate, epsilon over time),
experiments with different hyperparameter settings, extracts and displays
the learned policy, and saves results/plots to the results/ directory.

Bonus Option B implemented: training performance is visualized with
matplotlib graphs (reward curve, rolling success rate, epsilon decay).

Run:
    python train.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving figures
import matplotlib.pyplot as plt

from environment import FrozenLakeEnv
from agent import QLearningAgent
from utils import policy_to_grid_string, print_policy

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_agent(env, agent, n_episodes=10000, max_steps=200, verbose_every=1000):
    """
    Run the Q-Learning training loop.

    Returns a dict of training statistics.
    """
    episode_rewards = []
    episode_successes = []   # 1 if goal reached, 0 otherwise
    epsilon_history = []
    steps_per_episode = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        success = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                if state == env.goal_state:
                    success = 1
                break

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_successes.append(success)
        epsilon_history.append(agent.epsilon)
        steps_per_episode.append(step + 1)

        if verbose_every and episode % verbose_every == 0:
            recent_success_rate = np.mean(episode_successes[-verbose_every:]) * 100
            recent_avg_reward = np.mean(episode_rewards[-verbose_every:])
            print(f"Episode {episode:6d} | "
                  f"Success rate (last {verbose_every}): {recent_success_rate:5.1f}% | "
                  f"Avg reward: {recent_avg_reward:6.3f} | "
                  f"Epsilon: {agent.epsilon:.4f}")

    stats = {
        "episode_rewards": episode_rewards,
        "episode_successes": episode_successes,
        "epsilon_history": epsilon_history,
        "steps_per_episode": steps_per_episode,
        "n_successful_episodes": int(np.sum(episode_successes)),
        "overall_success_rate": float(np.mean(episode_successes) * 100),
    }
    return stats


def rolling_average(values, window=100):
    values = np.array(values, dtype=float)
    if len(values) < window:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_training_results(stats, tag, window=100):
    """Bonus Option B: visualize training performance with graphs."""
    rewards = stats["episode_rewards"]
    successes = stats["episode_successes"]
    epsilons = stats["epsilon_history"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 11))

    # 1. Episode reward + rolling average
    axes[0].plot(rewards, color="lightgray", linewidth=0.5, label="Episode reward")
    if len(rewards) >= window:
        roll = rolling_average(rewards, window)
        axes[0].plot(range(window, window + len(roll)), roll,
                     color="steelblue", linewidth=2,
                     label=f"{window}-episode rolling avg")
    axes[0].set_title(f"Episode Reward Over Training ({tag})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend(loc="lower right")

    # 2. Rolling success rate
    if len(successes) >= window:
        roll_success = rolling_average(successes, window) * 100
        axes[1].plot(range(window, window + len(roll_success)), roll_success,
                     color="seagreen", linewidth=2)
    axes[1].set_title(f"Rolling Success Rate ({window}-episode window)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_ylim(-5, 105)

    # 3. Epsilon decay
    axes[2].plot(epsilons, color="darkorange", linewidth=2)
    axes[2].set_title("Epsilon Decay Over Training")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"training_curves_{tag}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training plot -> {out_path}")
    return out_path


def run_experiment(tag, alpha, gamma, epsilon_decay, n_episodes=10000, seed=42):
    """Train one configuration and save its Q-table, stats, and plots."""
    print(f"\n=== Experiment: {tag} "
          f"(alpha={alpha}, gamma={gamma}, epsilon_decay={epsilon_decay}) ===")

    env = FrozenLakeEnv(is_slippery=False)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )

    stats = train_agent(env, agent, n_episodes=n_episodes)
    plot_training_results(stats, tag)

    np.save(os.path.join(RESULTS_DIR, f"qtable_{tag}.npy"), agent.q_table)

    policy = agent.get_policy()
    policy_str = policy_to_grid_string(env, policy)
    print(f"\nLearned policy ({tag}):")
    print(policy_str)

    summary = {
        "tag": tag,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_decay": epsilon_decay,
        "n_episodes": n_episodes,
        "n_successful_episodes": stats["n_successful_episodes"],
        "overall_success_rate": stats["overall_success_rate"],
        "final_epsilon": agent.epsilon,
        "policy_grid": policy_str,
    }
    return env, agent, stats, summary


def main():
    n_episodes = 10000

    # Part C requires experimenting with learning rate, discount factor,
    # and exploration rate. We run a baseline plus three variations,
    # each changing one hyperparameter at a time.
    experiment_configs = [
        dict(tag="baseline",        alpha=0.1,  gamma=0.99, epsilon_decay=0.9995),
        dict(tag="high_alpha",      alpha=0.5,  gamma=0.99, epsilon_decay=0.9995),
        dict(tag="low_gamma",       alpha=0.1,  gamma=0.90, epsilon_decay=0.9995),
        dict(tag="fast_eps_decay",  alpha=0.1,  gamma=0.99, epsilon_decay=0.999),
    ]

    all_summaries = []
    best_env, best_agent, best_stats = None, None, None
    best_success_rate = -1

    for cfg in experiment_configs:
        env, agent, stats, summary = run_experiment(n_episodes=n_episodes, **cfg)
        all_summaries.append(summary)

        if summary["overall_success_rate"] > best_success_rate:
            best_success_rate = summary["overall_success_rate"]
            best_env, best_agent, best_stats = env, agent, stats
            best_tag = cfg["tag"]

    # Save the comparison summary as JSON for the report/README.
    with open(os.path.join(RESULTS_DIR, "experiment_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n================ HYPERPARAMETER COMPARISON ================")
    for s in all_summaries:
        print(f"{s['tag']:15s} | alpha={s['alpha']:.2f} gamma={s['gamma']:.2f} "
              f"eps_decay={s['epsilon_decay']:.4f} | "
              f"success_rate={s['overall_success_rate']:5.2f}% | "
              f"successes={s['n_successful_episodes']}/{n_episodes}")

    print(f"\nBest configuration: {best_tag} "
          f"with {best_success_rate:.2f}% overall success rate.")

    # Save the best Q-table as the "main" one used by evaluate.py
    np.save(os.path.join(RESULTS_DIR, "qtable_best.npy"), best_agent.q_table)
    with open(os.path.join(RESULTS_DIR, "best_config.json"), "w") as f:
        json.dump({"tag": best_tag, "success_rate": best_success_rate}, f, indent=2)

    print(f"\nBest Q-table saved -> {os.path.join(RESULTS_DIR, 'qtable_best.npy')}")
    print("Run `python evaluate.py` next to evaluate the best trained agent.")


if __name__ == "__main__":
    main()
