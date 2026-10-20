"""Grid-world environment for the room-cleaning agents."""
import numpy as np

MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


class RoomEnv:
    """N x N room. 1 = dirty tile, 0 = clean tile.

    Every action (CLEAN or any move, including a move that bumps into a wall)
    costs exactly one step, matching the original SRARCS v1 rules.
    """

    def __init__(self, size=5, dirt_prob=0.4, rng=None):
        self.size = size
        self.dirt_prob = dirt_prob
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        n = self.size
        self.grid = (self.rng.random((n, n)) < self.dirt_prob).astype(int)
        if self.grid.sum() == 0:  # guarantee at least one dirty tile
            self.grid[self.rng.integers(n), self.rng.integers(n)] = 1
        self.initial_dirt = int(self.grid.sum())
        self.x = int(self.rng.integers(n))
        self.y = int(self.rng.integers(n))
        self.steps = 0
        self.moves = 0
        self.bumps = 0
        return self.percept()

    def percept(self):
        """What the agent is allowed to see: its own tile and its position."""
        return (self.x, self.y, bool(self.grid[self.x, self.y]), self.size)

    @property
    def dirty_left(self):
        return int(self.grid.sum())

    @property
    def cleaned_pct(self):
        return round(100 * (1 - self.grid.sum() / self.grid.size), 2)

    @property
    def done(self):
        return self.dirty_left == 0

    def step(self, action):
        self.steps += 1
        if action == "CLEAN":
            self.grid[self.x, self.y] = 0
        else:
            dx, dy = MOVES[action]
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                self.x, self.y = nx, ny
                self.moves += 1
            else:
                self.bumps += 1
        return self.percept()
