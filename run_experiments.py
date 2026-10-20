"""Reproduce every number and chart in the README.

    python run_experiments.py            # ~1 minute

Outputs go to ./results
"""
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from srarcs import RoomEnv, RandomReflexAgent, SweepReflexAgent, ModelBasedAgent, run_episode

SEED = 42
TRIALS = 500
SIZES = [4, 6, 8, 10]
DENSITIES = [0.2, 0.4, 0.6]
COLORS = {"Random Reflex (v1)": "#d1495b", "Sweep Reflex": "#00798c", "Model-Based (memory + BFS)": "#edae49"}


def make_agents(seed):
    return [RandomReflexAgent(random.Random(seed)), SweepReflexAgent(), ModelBasedAgent()]


def legacy_replication():
    """Exactly the v1 setup: 5x5, 40% dirt, random reflex agent, 50-step cap."""
    rng = np.random.default_rng(SEED)
    agent = RandomReflexAgent(random.Random(SEED))
    rows = []
    for _ in range(5000):
        env = RoomEnv(5, 0.4, rng)
        rows.append(run_episode(agent, env, max_steps=50))
    df = pd.DataFrame(rows)
    return df


def benchmark():
    rows = []
    for n in SIZES:
        for p in DENSITIES:
            # Same rooms and start positions for every agent -> fair comparison
            for t in range(TRIALS):
                seed = SEED * 1_000_003 + n * 10_007 + int(p * 100) * 101 + t
                for agent in make_agents(seed):
                    env = RoomEnv(n, p, np.random.default_rng(seed))
                    r = run_episode(agent, env, max_steps=50 * n * n)
                    rows.append({"agent": agent.name, "size": n, "dirt_prob": p, "trial": t, **r})
    return pd.DataFrame(rows)


def sample_traces():
    """One full episode per agent on the same 6x6 room -> a clean trace dataset."""
    out = []
    seed = 7
    for agent in make_agents(seed):
        env = RoomEnv(6, 0.4, np.random.default_rng(seed))
        _, trace = run_episode(agent, env, max_steps=2000, record=True)
        for t in trace:
            out.append((agent.name, *t))
    cols = ["agent", "step", "agent_x", "agent_y", "action", "dirty_left", "cleaned_pct"]
    return pd.DataFrame(out, columns=cols)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def chart_scaling(df):
    d = df[df.dirt_prob == 0.4].groupby(["agent", "size"]).steps.mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, g in d.groupby("agent"):
        ax.plot(g["size"], g.steps, marker="o", lw=2.4, label=name, color=COLORS[name])
    ax.set_yscale("log")
    ax.set_xticks(SIZES)
    ax.set_xlabel("Room size (N x N)")
    ax.set_ylabel("Mean steps to clean (log scale)")
    ax.set_title("Cost of cleaning grows far faster for the random agent\n(40% dirt, 500 rooms per point)", fontsize=11, loc="left")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig("results/scaling.png", dpi=160)
    plt.close(fig)


def chart_legacy(leg):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(leg.final_cleaned_pct, bins=np.arange(40, 105, 4), color="#d1495b", edgecolor="white")
    ax.set_xlabel("Room cleaned when the 50-step limit hit (%)")
    ax.set_ylabel("Episodes (of 5000)")
    solved = leg.solved.mean() * 100
    ax.set_title(f"v1 setup (5x5, 50 steps): only {solved:.0f}% of runs finish the job", fontsize=11, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig("results/legacy_v1_completion.png", dpi=160)
    plt.close(fig)


def chart_efficiency(df):
    d = df[(df["size"] == 8)].copy()
    d["wasted_pct"] = 100 * (d.steps - d.initial_dirt) / d.steps
    g = d.groupby(["agent", "dirt_prob"]).wasted_pct.mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.26
    for i, (name, gg) in enumerate(g.groupby("agent")):
        ax.bar(np.arange(len(DENSITIES)) + (i - 1) * w, gg.wasted_pct, w, label=name, color=COLORS[name])
    ax.set_xticks(range(len(DENSITIES)))
    ax.set_xticklabels([f"{int(p*100)}% dirt" for p in DENSITIES])
    ax.set_ylabel("Steps spent NOT cleaning (%)")
    ax.set_title("Wasted effort on an 8x8 room", fontsize=11, loc="left")
    ax.legend(frameon=False, loc="lower left")
    style(ax)
    fig.tight_layout()
    fig.savefig("results/wasted_effort.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    leg = legacy_replication()
    print("LEGACY v1 | solved within 50 steps: %.1f%% | mean cleaned: %.1f%% | mean steps: %.1f"
          % (leg.solved.mean() * 100, leg.final_cleaned_pct.mean(), leg.steps.mean()))
    leg.to_csv("results/legacy_v1_replication.csv", index=False)

    df = benchmark()
    df.to_csv("results/benchmark_episodes.csv", index=False)
    summary = (df.groupby(["agent", "size", "dirt_prob"])
                 .agg(solved_pct=("solved", lambda s: 100 * s.mean()),
                      mean_steps=("steps", "mean"),
                      median_steps=("steps", "median"),
                      p95_steps=("steps", lambda s: s.quantile(0.95)),
                      mean_bumps=("bumps", "mean"))
                 .round(2).reset_index())
    summary.to_csv("results/benchmark_summary.csv", index=False)
    print(summary[summary.dirt_prob == 0.4].to_string(index=False))

    sample_traces().to_csv("results/sample_traces.csv", index=False)
    chart_scaling(df)
    chart_legacy(leg)
    chart_efficiency(df)
    print("done")
