import math
import os
import random

import numpy as np

from PIL import Image
import imageio
from PIL.ImageChops import offset
from pygifsicle import optimize

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot

#add the robot to the world
#the run_simulation function evaluate how good a specific robot is
def run_simulation(world,robot_structure,view = False):
    try:
        world.remove_object("robot001")
    except:
        pass
    #insert the new robot structures into the environment(3,1)
    world.add_from_array(
        name='robot001',
        structure=robot_structures,
        x=3,
        y=1,
    )

    #prepare the simulation
    sim = EvoSim(world)
    sim.reset()

    viewer = EvoViewer(sim, resolution = (400,200))

    input = sim.get_dim_action_space('robot001')
    print(input)

    def random_control_array(n):
        control =[]
        for i in range(n):
            control.append(random.random())
        return np.array(control)

    def sin_control_array(n,time):
        control =[]
        frequency = 0.33
        offset = 0.1

        for i in range(n):
            control.append(np.sin(time*frequency + i*offset))

        return np.array(control)

    robot_stars_pos = sim.object_pos_at_time(sim.get_time(),"robot001")
    robot_stars_pos = np.mean(robot_stars_pos,axis=1)

    for steps in range(200):
        sim.step()
        sim.set_action('robot001', sin_control_array(input,steps))
        #viewer.render()
        if view:
            viewer.render(mode="screen")

    robot_end_pos = sim.object_pos_at_time(sim.get_time(),"robot001")
    robot_end_pos = np.mean(robot_end_pos,axis=1)


    distance = robot_end_pos - robot_stars_pos

    return distance


world = EvoWorld.from_json('my_environment001.json')
best_distance = 0
best_robot = None

for i in range(100):

    robot_structures,robot_connections = sample_robot((5,5))

    distance = run_simulation(world,robot_structures)
    if distance[0]>best_distance:
        best_distance = distance[0]
        best_robot = robot_structures

print(best_distance)
print(best_robot)

run_simulation(world,best_robot,view=True)