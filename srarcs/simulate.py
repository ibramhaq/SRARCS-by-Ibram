"""Episode runner."""
from .env import RoomEnv


def run_episode(agent, env, max_steps, record=False):
    """Run one episode on an already-reset env. Returns a result dict (+ trace)."""
    agent.reset()
    percept = env.percept()
    trace = []
    if record:
        trace.append((0, env.x, env.y, "START", env.dirty_left, env.cleaned_pct))
    while not env.done and env.steps < max_steps:
        action = agent.act(percept)
        percept = env.step(action)
        if record:
            trace.append((env.steps, env.x, env.y, action, env.dirty_left, env.cleaned_pct))
    result = {
        "solved": env.done,
        "steps": env.steps,
        "moves": env.moves,
        "bumps": env.bumps,
        "initial_dirt": env.initial_dirt,
        "final_cleaned_pct": env.cleaned_pct,
    }
    return (result, trace) if record else result
