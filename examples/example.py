from mass import MASS
import numpy as np
import time

start_time = time.time()

simulator = MASS(
    bitmap = './maps/test-with-2-levels.bmp',
    batch_size = 1,
    observation_range = 5,
    agent_size = 0,
    amount_of_agents = 3,
    stigmergy_evaporation_speed = np.array([10, 20, 30]),
    stigmergy_colours = np.array([[255,64,0],[255,128,0],[255,255,0]]),
    inertia = 0,
    collision_detection = np.array([True, True]),
    max_steps = 1000,
    rendering_allowed = True
)

stig_actions = np.array([[[0,1],[1,2],[2,3]]])
actions = np.array([[[1,0],[1,0],[1,0]]])

while True:
    obs, rew, done, info = simulator.step(actions, stig_actions)
    print(obs, rew)
    time.sleep(1)
    simulator.render()
