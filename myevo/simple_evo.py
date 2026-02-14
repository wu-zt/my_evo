import math
import os
import random

import numpy as np

from PIL import Image
import imageio
from PIL.ImageChops import offset
from pygifsicle import optimize

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot


def is_connected(structure):
    if not np.any(structure):
        return False
    # measure the size of matrix,and then give the number to(rows,cols)
    rows, cols = structure.shape
    visited = np.zeros((rows, cols), dtype=bool)

    # Find first not vacant box of robot
    start_r, start_c = -1, -1
    for r in range(rows):
        for c in range(cols):
            if structure[r, c] != 0:
                start_r, start_c = r, c
                break
        if start_r != -1:
            break

    if start_r == -1:
        return False

    # BFS
    queue = [(start_r, start_c)]
    visited[start_r, start_c] = True
    count = 0

    while queue:
        r, c = queue.pop(0)
        count += 1

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if structure[nr, nc] != 0 and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    # Check if we visited all non-zero elements
    total_non_zero = np.count_nonzero(structure)
    return count == total_non_zero


# add the robot to the world
# the run_simulation function evaluate how good a specific robot is
def run_simulation(world, robot_structures, view=False):
    if not is_connected(robot_structures):
        return [-1000]

    try:
        world.remove_object("robot001")
    except:
        pass
    # insert the new robot structures into the environment(3,1)
    try:
        world.add_from_array(
            name='robot001',
            structure=robot_structures,
            x=3,
            y=1,
        )
    except:
        return [-1000]

    # prepare the simulation
    sim = EvoSim(world)
    sim.reset()

    viewer = EvoViewer(sim, resolution=(400, 200))

    try:
        input = sim.get_dim_action_space('robot001')
    except:
        return [-1000]

    print(input)

    def random_control_array(n):
        control = []
        for i in range(n):
            control.append(random.random())
        return np.array(control)

    def sin_control_array(n, time):
        control = []
        frequency = 0.33
        offset = 0.1

        for i in range(n):
            control.append(np.sin(time * frequency + i * offset))

        return np.array(control)

    # caculate the numbers of actuators,if no actuators return -1000
    try:
        robot_stars_pos = sim.object_pos_at_time(sim.get_time(), "robot001")
    except:
        return [-1000]

    robot_stars_pos = np.mean(robot_stars_pos, axis=1)

    for steps in range(200):
        sim.step()
        sim.set_action('robot001', sin_control_array(input, steps))
        # viewer.render()
        if view:
            viewer.render(mode="screen")

    robot_end_pos = sim.object_pos_at_time(sim.get_time(), "robot001")
    robot_end_pos = np.mean(robot_end_pos, axis=1)

    distance = robot_end_pos - robot_stars_pos

    return distance


def mutate(structure):
    new_structure = structure.copy()
    rows, cols = new_structure.shape

    r = random.randint(0, rows - 1)
    c = random.randint(0, cols - 1)

    # 0: Empty, 1: Rigid, 2: Soft, 3: H-Act, 4: V-Act
    new_structure[r, c] = random.randint(0, 4)

    return new_structure


world = EvoWorld.from_json('my_environment001.json')

# Initialize with a random robot
best_robot, _ = sample_robot((5, 5))
distance = run_simulation(world, best_robot)
best_distance = distance[0]

for i in range(100):
    new_robot = mutate(best_robot)

    distance = run_simulation(world, new_robot)

    if distance[0] > best_distance:
        best_distance = distance[0]
        best_robot = new_robot
        print(f"Generation {i}: New best distance {best_distance}")

print(best_distance)
print(best_robot)

run_simulation(world, best_robot, view=True)