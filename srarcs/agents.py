"""Three cleaning agents of increasing sophistication.

1. RandomReflexAgent  - the original SRARCS agent: if dirty -> clean, else move randomly.
2. SweepReflexAgent   - still a pure reflex agent (no memory), but its movement rule
                        is a function of its position, tracing a closed loop through
                        the room. Requires an even grid size (Hamiltonian cycle).
3. ModelBasedAgent    - keeps an internal model (visited tiles) and walks by shortest
                        path to the nearest unvisited tile.
"""
import random
from collections import deque

from .env import MOVES


class RandomReflexAgent:
    name = "Random Reflex (v1)"

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def reset(self):
        pass

    def act(self, percept):
        _, _, dirty, _ = percept
        return "CLEAN" if dirty else self.rng.choice(list(MOVES))


class SweepReflexAgent:
    name = "Sweep Reflex"

    def reset(self):
        pass

    def act(self, percept):
        x, y, dirty, n = percept
        if n % 2:
            raise ValueError("SweepReflexAgent needs an even grid size")
        if dirty:
            return "CLEAN"
        # Fixed rule per position -> closed loop visiting every tile
        if x == 0:
            return "RIGHT" if y < n - 1 else "DOWN"
        if y == 0:
            return "UP"                       # return lane (column 0)
        if x % 2 == 1:                         # odd rows run right -> left
            if y > 1:
                return "LEFT"
            return "LEFT" if x == n - 1 else "DOWN"
        return "RIGHT" if y < n - 1 else "DOWN"  # even rows run left -> right


class ModelBasedAgent:
    name = "Model-Based (memory + BFS)"

    def reset(self):
        self.visited = set()

    def act(self, percept):
        x, y, dirty, n = percept
        self.visited.add((x, y))
        if dirty:
            return "CLEAN"
        return self._first_step_to_nearest_unvisited(x, y, n)

    def _first_step_to_nearest_unvisited(self, x, y, n):
        queue = deque([((x, y), None)])
        seen = {(x, y)}
        while queue:
            (cx, cy), first = queue.popleft()
            if (cx, cy) not in self.visited:
                return first
            for action, (dx, dy) in MOVES.items():
                nxt = (cx + dx, cy + dy)
                if 0 <= nxt[0] < n and 0 <= nxt[1] < n and nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, first or action))
        return "CLEAN"  # everything visited: nothing left to do
