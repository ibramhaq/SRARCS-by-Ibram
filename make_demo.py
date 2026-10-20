"""Render assets/demo.gif: the three agents cleaning the SAME 8x8 room side by side."""
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from srarcs import RoomEnv, RandomReflexAgent, SweepReflexAgent, ModelBasedAgent

N, SEED, MAX_FRAMES = 8, 11, 110
agents = [RandomReflexAgent(random.Random(SEED)), SweepReflexAgent(), ModelBasedAgent()]
envs = [RoomEnv(N, 0.4, np.random.default_rng(SEED)) for _ in agents]

# Record per-step snapshots
history = []
for agent, env in zip(agents, envs):
    agent.reset()
    percept = env.percept()
    snaps = [(env.grid.copy(), env.x, env.y, 0)]
    while not env.done and env.steps < 3000:
        percept = env.step(agent.act(percept))
        snaps.append((env.grid.copy(), env.x, env.y, env.steps))
    history.append(snaps)

longest = max(len(h) for h in history)
stride = max(1, longest // MAX_FRAMES)
frame_ids = list(range(0, longest, stride)) + [longest - 1] * 8  # hold final frame


def draw(i):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.1), dpi=80)
    for ax, agent, h in zip(axes, agents, history):
        grid, x, y, steps = h[min(i, len(h) - 1)]
        finished = i >= len(h) - 1
        ax.imshow(grid, cmap="YlOrBr", vmin=0, vmax=1.6)
        ax.plot(y, x, "o", ms=17, color="#00798c", mec="white", mew=2)
        ax.set_xticks(np.arange(-.5, N, 1)); ax.set_yticks(np.arange(-.5, N, 1))
        ax.grid(color="white", lw=1.5); ax.tick_params(length=0, labelbottom=False, labelleft=False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(agent.name, fontsize=11, fontweight="bold")
        tag = f"DONE in {steps} steps" if finished else f"step {steps}  |  dirty left: {int(grid.sum())}"
        ax.set_xlabel(tag, fontsize=10, color="#2a7a3b" if finished else "#333")
    fig.suptitle("SRARCS v2 - same room, same start tile (brown = dirty, blue = agent)", fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).convert("P", palette=Image.ADAPTIVE)
    plt.close(fig)
    return img


frames = [draw(i) for i in frame_ids]
frames[0].save("assets/demo.gif", save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=True)
print("frames:", len(frames), "| steps per agent:", [len(h) - 1 for h in history])
