import random
import numpy as np

# -----------------------------
# 1. Environment Creation
# -----------------------------
# 5x5 room grid
# 1 = Dirty cell
# 0 = Clean cell

ROOM_SIZE = 5
room = np.random.choice([0, 1], size=(ROOM_SIZE, ROOM_SIZE), p=[0.6, 0.4])

# -----------------------------
# 2. Agent Initialization
# -----------------------------
agent_x = random.randint(0, ROOM_SIZE - 1)
agent_y = random.randint(0, ROOM_SIZE - 1)

# -----------------------------
# Helper Functions
# -----------------------------
def display_room(room, agent_x, agent_y):
    print("\nRoom State:")
    for i in range(ROOM_SIZE):
        for j in range(ROOM_SIZE):
            if i == agent_x and j == agent_y:
                print("A", end=" ")  # Agent position
            else:
                print(room[i][j], end=" ")
        print()
    print()

def cleaned_percentage(room):
    total_cells = ROOM_SIZE * ROOM_SIZE
    dirty_cells = np.sum(room)
    return round((1 - dirty_cells / total_cells) * 100, 2)

# -----------------------------
# 3. Simple Reflex Agent Logic
# -----------------------------
steps = 0
MAX_STEPS = 50

print("Initial Environment")
display_room(room, agent_x, agent_y)

while np.sum(room) > 0 and steps < MAX_STEPS:
    steps += 1

    # Perception
    if room[agent_x][agent_y] == 1:
        # Action: Clean
        room[agent_x][agent_y] = 0
        print(f"Step {steps}: Dirt detected → CLEANING")
    else:
        # Action: Move randomly
        move = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

        if move == "UP" and agent_x > 0:
            agent_x -= 1
        elif move == "DOWN" and agent_x < ROOM_SIZE - 1:
            agent_x += 1
        elif move == "LEFT" and agent_y > 0:
            agent_y -= 1
        elif move == "RIGHT" and agent_y < ROOM_SIZE - 1:
            agent_y += 1

        print(f"Step {steps}: No dirt → MOVING {move}")

    display_room(room, agent_x, agent_y)
    print(f"Cleaned: {cleaned_percentage(room)}%")

# -----------------------------
# 4. Termination
# -----------------------------
print("\nSimulation Finished")
print(f"Total Steps Taken: {steps}")
print(f"Final Cleaning Percentage: {cleaned_percentage(room)}%")
