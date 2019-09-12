import unittest as ut
import numpy as np
from mass import MASS

class TestMASS(ut.TestCase):
    def test_reward_collision_with_target(self):
        test_obj = MASS(bitmap = './tests/test-with-2-levels.bmp',
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
        test_obj.__targets = 0
        self.assertTrue(test_obj.__targets)
    
        
