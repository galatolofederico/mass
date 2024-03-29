from __future__ import division
from pyqtgraph.Qt import QtCore, QtGui
from PIL import Image
import numpy as np
import pyqtgraph as pg
import sys
import random
import threading

np.set_printoptions(threshold=sys.maxsize)
pg.setConfigOptions(imageAxisOrder='row-major')
pg.setConfigOption('background', 'w')


class MASS:
    def __init__(self, bitmap, batch_size, agent_size, observation_range, amount_of_agents,
                 stigmergy_evaporation_speed, stigmergy_colours, inertia, collision_detection, max_steps,
                 rendering_allowed=False, agent_colour=[255, 255, 255]):

        self.__init_simulator_parameters(bitmap, batch_size, agent_size, observation_range, amount_of_agents,
                                         stigmergy_evaporation_speed, stigmergy_colours, inertia, collision_detection,
                                         max_steps, rendering_allowed, agent_colour)

        self.__init_environment_parameters()
        self.__parse_bitmap()

        self.__init_agents_parameters()
        self.__init_agents()

        self.__init_stigmergy_space()
        
        self.__rendering_initialized = False

    def __init_simulator_parameters(self, bitmap, batch_size, agent_size, observation_range, amount_of_agents,
                                    stigmergy_evaporation_speed, stigmergy_colours, inertia, collision_detection,
                                    max_steps, rendering_allowed, agent_colour):

        self.__bitmap = bitmap
        self.__batch_size = batch_size
        self.__observation_range = observation_range
        self.__agent_size = agent_size
        self.__amount_of_agents = amount_of_agents

        # The values stored in the stigmergy space are int32 to avoid float precision issues
        # with the evaporation mechanism.
        self.__stigmergy_evaporation_speed = stigmergy_evaporation_speed.astype(int)
        self.__stigmergy_colours = stigmergy_colours

        self.__inertia = inertia
        self.__collision_detection = collision_detection
        self.__max_steps = max_steps
        self.__rendering_allowed = rendering_allowed
        self.__agent_colour = agent_colour

    def __init_environment_parameters(self):
        self.__environment_bitmap = None
        self.__targets = np.array([])
        self.__collision = np.array([])
        self.__no_collision = np.array([])  # Initialized in parse_bitmap(), but currently not used

    def __parse_bitmap(self):
        input_array = np.asarray(Image.open(self.__bitmap))
        rgb_bit_array = np.unpackbits(input_array, axis=2)
        # rgb_bit_array is a matrix of pixels, where each cell (each pixel) is a 24-bit array

        collision = []
        no_collision = []
        level_founded = 0

        for i in range(0, 24):
            level = rgb_bit_array[:, :, i]
            # Only levels with at least 1 item are inserted in the environment
            if np.any(level):
                if level_founded == 0:
                    # First level is composed of targets
                    self.__targets = np.asarray(level)
                    self.__environment_bitmap = np.full((self.__targets.shape[0], self.__targets.shape[1], 3), 0)
                else:
                    if self.__collision_detection[level_founded - 1]:
                        collision.append(level)
                    else:
                        no_collision.append(level)

                self.__environment_bitmap[level == 1, :] = (
                        self.__environment_bitmap[level == 1, :] + self.__get_colour(i))
                level_founded += 1

        if not collision:
            collision = np.zeros((1, self.__targets.shape[0], self.__targets.shape[1]))
        else:
            collision = np.asarray(collision)

        if not no_collision:
            no_collision = np.zeros((1, self.__targets.shape[0], self.__targets.shape[1]))
        else:
            no_collision = np.asarray(no_collision)

        self.__collision = np.sum(collision, axis=0)
        self.__collision[self.__collision > 0] = 1

        self.__no_collision = np.sum(no_collision, axis=0)
        self.__no_collision[self.__no_collision > 0] = 1

    def __get_colour(self, i):
        colour = np.zeros(shape=3)
        i = 24 - i - 1

        colour[2 - i // 8] = 2 ** (i % 8)
        return colour

    def __init_agents_parameters(self):
        self.__agents_position_float = None  # Contains the not approximated position. Not used for drawing
        self.__agents_position = np.full((self.__batch_size, self.__amount_of_agents, 2), -1)
        self.__agents_velocity = np.zeros((self.__batch_size, self.__amount_of_agents, 2))

        self.__drawn_agents = np.zeros((self.__batch_size, self.__amount_of_agents,
                                        self.__targets.shape[0], self.__targets.shape[1]))

        self.__targets_achieved = np.zeros((self.__batch_size, self.__amount_of_agents,
                                            self.__targets.shape[0], self.__targets.shape[1]))
        self.__current_steps = 0

    def __init_agents(self):
        for batch_index in range(self.__batch_size):
            agent_index = 0
            while agent_index < len(self.__agents_position[batch_index]):
                self.__agents_position[batch_index, agent_index] = np.asarray([
                    random.randint(0, self.__targets.shape[0] - 1),
                    random.randint(0, self.__targets.shape[1] - 1)
                ])

                # An agent is correctly positioned if it's rendered completely inside the map and
                # it doesn't collide with environment or other agents
                if not self.__out_of_map(batch_index, agent_index, self.__agent_size):
                    self.__draw_agent(batch_index, agent_index)
                    if not self.__detect_collision(batch_index):
                        agent_index += 1

        self.__agents_position_float = np.copy(self.__agents_position).astype(float)

    def __draw_agent(self, batch_index, agent_index):
        radius = self.__agent_size
        position_axis_0 = self.__agents_position[batch_index, agent_index, 0]
        position_axis_1 = self.__agents_position[batch_index, agent_index, 1]
        interval_axis_0, interval_axis_1 = self.__drawing_boundaries(position_axis_0, position_axis_1, radius)

        agent_level = np.zeros((self.__targets.shape[0], self.__targets.shape[1]))
        agent_level[interval_axis_0, interval_axis_1] = 1
        self.__drawn_agents[batch_index, agent_index] = agent_level

    def __drawing_boundaries(self, position_axis_0, position_axis_1, radius):
        start_point_axis_0 = position_axis_0 - radius
        end_point_axis_0 = position_axis_0 + radius + 1
        start_point_axis_1 = position_axis_1 - radius
        end_point_axis_1 = position_axis_1 + radius + 1

        if position_axis_0 - radius < 0:
            start_point_axis_0 = 0
        elif position_axis_0 - radius >= self.__targets.shape[0]:
            start_point_axis_0 = self.__targets.shape[0]

        if position_axis_0 + radius < 0:
            end_point_axis_0 = 0
        elif position_axis_0 + radius >= self.__targets.shape[0]:
            end_point_axis_0 = self.__targets.shape[0]

        if position_axis_1 - radius < 0:
            start_point_axis_1 = 0
        elif position_axis_1 - radius >= self.__targets.shape[1]:
            start_point_axis_1 = self.__targets.shape[1]

        if position_axis_1 + radius < 0:
            end_point_axis_1 = 0
        elif position_axis_1 + radius >= self.__targets.shape[1]:
            end_point_axis_1 = self.__targets.shape[1]

        return slice(start_point_axis_0, end_point_axis_0), slice(start_point_axis_1, end_point_axis_1)

    def __detect_collision(self, batch_index):
        collision_level = self.__collision[np.newaxis, ...]
        collision_detection = np.append(self.__drawn_agents[batch_index], collision_level, axis=0)
        collision_detection = np.sum(collision_detection, axis=0)

        if np.any(collision_detection > 1):
            return True

        return False

    def __out_of_map(self, batch_index, agent_index, radius):
        position_axis_0 = self.__agents_position[batch_index, agent_index, 0]
        position_axis_1 = self.__agents_position[batch_index, agent_index, 1]

        if (position_axis_0 - radius < 0 or position_axis_0 - radius >= self.__targets.shape[0] or
                position_axis_0 + radius < 0 or position_axis_0 + radius >= self.__targets.shape[0]):
            return True

        if (position_axis_1 - radius < 0 or position_axis_1 - radius >= self.__targets.shape[1] or
                position_axis_1 + radius < 0 or position_axis_1 + radius >= self.__targets.shape[1]):
            return True

        return False

    def __init_stigmergy_space(self):
        # The values stored in the stigmergy space are int32 to avoid float precision issues
        # (e.g. zero not reached when subtracting)
        self.__stigmergy_space = np.zeros((self.__batch_size,
                                           self.__stigmergy_evaporation_speed.shape[0],
                                           self.__targets.shape[0],
                                           self.__targets.shape[1]),
                                          int)

    def __init_rendering_parameters(self):
        self.__image = np.full((self.__targets.shape[0], self.__targets.shape[1], 3), 0)
        self.__image_semaphore = None

        if self.__rendering_allowed:
            if self.__batch_size > 1:
                raise ValueError("Rendering is allowed only when batch_size is equal to 1")

            self.__image_semaphore = threading.Lock()
            rendering = threading.Thread(target=self.__init_rendering)
            rendering.start()
            self.__rendering_initialized = True

    def __init_rendering(self):
        app = QtGui.QApplication([])

        # Create window with GraphicsView widget
        w = pg.GraphicsView()
        w.show()
        w.showMaximized()
        w.setWindowTitle('Simulator')

        view = pg.ViewBox()
        view.invertY()
        view.setAspectLocked(True)
        w.setCentralItem(view)

        self.__image_semaphore.acquire()
        img = pg.ImageItem(self.__image)
        view.addItem(img)
        self.__image_semaphore.release()

        timer = QtCore.QTimer()
        timer.timeout.connect(lambda: self.__update_rendering(view))
        timer.start(100)

        # Start Qt event loop unless running in interactive mode or using pyside.
        app.instance().exec()

    def __update_rendering(self, view):
        self.__image_semaphore.acquire()

        view.clear()
        img = pg.ImageItem(self.__image)
        view.addItem(img)

        self.__image_semaphore.release()

    def __stigmergy_evaporation(self):
        evaporation_levels = np.zeros((self.__stigmergy_evaporation_speed.shape[0],
                                       self.__targets.shape[0],
                                       self.__targets.shape[1]),
                                      int)

        for index in range(self.__stigmergy_evaporation_speed.shape[0]):
            evaporation_levels[index] = np.full((self.__targets.shape[0], self.__targets.shape[1]),
                                                self.__stigmergy_evaporation_speed[index])

        for batch_index in range(self.__batch_size):
            self.__stigmergy_space[batch_index] -= evaporation_levels

        self.__stigmergy_space[self.__stigmergy_space < 0] = 0

    def __update_stigmergy_space(self, stigmergy_actions):
        for batch_index in range(self.__batch_size):
            for agent_index in range(self.__amount_of_agents):
                position_axis_0 = self.__agents_position[batch_index, agent_index, 0]
                position_axis_1 = self.__agents_position[batch_index, agent_index, 1]
                stig_level = int(stigmergy_actions[batch_index, agent_index, 0])
                stig_radius = int(stigmergy_actions[batch_index, agent_index, 1])

                if stig_level == -1:  # No pheromone release
                    continue

                interval_axis_0, interval_axis_1 = self.__drawing_boundaries(position_axis_0,
                                                                             position_axis_1,
                                                                             stig_radius)

                # 100 is the intensity of the pheromone released.
                # It is the same for all the agents and for every pheromone type.
                self.__stigmergy_space[batch_index][stig_level][interval_axis_0, interval_axis_1] += 100

    def __update_agents(self, agents_actions, rewards_table, observations_table):
        for batch_index in range(self.__batch_size):
            for agent_index in range(self.__amount_of_agents):
                self.__update_velocity(batch_index, agent_index, agents_actions)
                self.__update_position(batch_index, agent_index)
                self.__draw_agent(batch_index, agent_index)
                self.__target_achieved(batch_index, agent_index)
                rewards_table[batch_index, agent_index] = self.__reward(batch_index, agent_index)
                observations_table[batch_index, agent_index] = self.__get_observation(batch_index, agent_index)

    def __update_velocity(self, batch_index, agent_index, agents_actions):
        agent_velocity = self.__agents_velocity[batch_index, agent_index]
        agent_command = agents_actions[batch_index, agent_index]

        self.__agents_velocity[batch_index, agent_index] = (agent_velocity * self.__inertia +
                                                            agent_command * (1 - self.__inertia))

    def __update_position(self, batch_index, agent_index):
        agent_position = self.__agents_position_float[batch_index, agent_index]
        agent_velocity = self.__agents_velocity[batch_index, agent_index]
        t_constant = 1

        self.__agents_position_float[batch_index, agent_index] = agent_position + agent_velocity * t_constant
        self.__agents_position = np.copy(self.__agents_position_float).astype(int)

    def __target_achieved(self, batch_index, agent_index):
        target_collision = self.__drawn_agents[batch_index, agent_index] + self.__targets
        self.__targets_achieved[batch_index][agent_index][target_collision > 1] = 1

    def __reward(self, batch_index, agent_index):
        average_distance = self.__average_distance_to_targets(batch_index, agent_index)
        alpha = 0.5
        num_target_achieved = np.count_nonzero(self.__targets_achieved[batch_index, agent_index])

        reward_score = 1 / average_distance + alpha * num_target_achieved

        if self.__detect_collision(batch_index):
            return reward_score * (-1)

        if self.__out_of_map(batch_index, agent_index, self.__agent_size):
            return reward_score * (-2)

        return reward_score

    def __average_distance_to_targets(self, batch_index, agent_index):
        targets_not_achieved = self.__targets + self.__targets_achieved[batch_index, agent_index]
        targets_not_achieved[targets_not_achieved > 1] = 0
        average_distance = 1

        if np.any(targets_not_achieved):
            targets_positions = np.array(list(zip(*np.nonzero(targets_not_achieved))))
            agent_position = np.repeat(self.__agents_position[batch_index, agent_index].reshape(1, 2),
                                       targets_positions.shape[0], 0)

            average_distance = np.average(
                np.sqrt(np.sum(np.power(np.subtract(agent_position, targets_positions), 2), 1)))

        return average_distance

    def __get_observation(self, batch_index, agent_index):
        position_axis_0 = self.__agents_position[batch_index, agent_index, 0]
        position_axis_1 = self.__agents_position[batch_index, agent_index, 1]
        observation_radius = self.__agent_size + self.__observation_range

        agents = np.sum(self.__drawn_agents[batch_index], axis=0)
        stigmergy_space = np.sum(self.__stigmergy_space[batch_index], axis=0)
        
        environment = np.stack((self.__targets,  self.__collision, agents, stigmergy_space)) # shape = (4, x, x)
        environment = environment.reshape(((*environment.shape[1:], environment.shape[0]))) # shape = (x, x, 4)
        
        if self.__out_of_map(batch_index, agent_index, observation_radius):
            # Map view enlargement: the agent will see -1 if a space is outside the map
            # The position of the agent is reevaluated according to the new map dimensions
            environment, position_axis_0, position_axis_1 = self.__enlarge_map_view(position_axis_0, position_axis_1,
                                                                                    observation_radius, environment)

        observation = environment[position_axis_0 - observation_radius: position_axis_0 + observation_radius + 1,
                                  position_axis_1 - observation_radius: position_axis_1 + observation_radius + 1]

        return observation

    def __enlarge_map_view(self, position_axis_0, position_axis_1, observation_radius, environment):
        enlargement_axis_before_0 = 0
        enlargement_axis_after_0 = 0
        enlargement_axis_before_1 = 0
        enlargement_axis_after_1 = 0

        if position_axis_0 - observation_radius < 0:
            enlargement_axis_before_0 = abs(position_axis_0 - observation_radius)

        if position_axis_0 + observation_radius >= self.__targets.shape[0]:
            enlargement_axis_after_0 = position_axis_0 + observation_radius - (self.__targets.shape[0] - 1)

        if position_axis_1 - observation_radius < 0:
            enlargement_axis_before_1 = abs(position_axis_1 - observation_radius)

        if position_axis_1 + observation_radius >= self.__targets.shape[1]:
            enlargement_axis_after_1 = position_axis_1 + observation_radius - (self.__targets.shape[1] - 1)
        
        enlarged_map = np.pad(environment, [(enlargement_axis_before_0, enlargement_axis_after_0),
                                            (enlargement_axis_before_1, enlargement_axis_after_1),
                                            (0, 0)],
                              constant_values=(-1))

        return enlarged_map, position_axis_0 + enlargement_axis_before_0, position_axis_1 + enlargement_axis_before_1

    def __environment_info(self):
        info = {
            "Agents position - float": self.__agents_position_float,
            "Agents position": self.__agents_position,
            "Agents velocity": self.__agents_velocity,
            "Targets achieved": self.__targets_achieved,
            "Stigmergy Space":  self.__stigmergy_space
        }

        return info

    def render(self):
        if self.__rendering_allowed:
            if not self.__rendering_initialized: self.__init_rendering_parameters()
            environment = np.copy(self.__environment_bitmap)
            agents = np.sum(self.__drawn_agents[0], axis=0)
            stigmergy_space = self.__stigmergy_space[0]

            environment[agents == 1, :] = environment[agents == 1, :] + self.__agent_colour
            environment[environment > 255] = environment[environment > 255]//2

            for index in range(stigmergy_space.shape[0]):
                environment[stigmergy_space[index] > 0, :] += self.__stigmergy_colours[index]
                environment[environment > 255] = environment[environment > 255]//2

            self.__image_semaphore.acquire()
            np.copyto(self.__image, environment)
            self.__image_semaphore.release()

    def reset(self):
        observation_dimension = 2 * (self.__agent_size + self.__observation_range) + 1
        observations_table = np.zeros((self.__batch_size, self.__amount_of_agents,
                                       observation_dimension, observation_dimension, 4))

        self.__init_agents_parameters()
        self.__init_agents()
        self.__init_stigmergy_space()

        # Initial observation
        for batch_index in range(self.__batch_size):
            for agent_index in range(self.__amount_of_agents):
                observations_table[batch_index, agent_index] = self.__get_observation(batch_index, agent_index)

        return observations_table

    def step(self, agents_actions, stigmergy_actions):
        self.__current_steps += 1
        if self.__current_steps > self.__max_steps:
            return None, None, True, "Maximum number of steps reached"

        observation_dimension = 2*(self.__agent_size + self.__observation_range) + 1
        observations_table = np.zeros((self.__batch_size, self.__amount_of_agents,
                                       observation_dimension, observation_dimension, 4))

        rewards_table = np.zeros((self.__batch_size, self.__amount_of_agents, 1))

        self.__stigmergy_evaporation()
        self.__update_stigmergy_space(stigmergy_actions)
        self.__update_agents(agents_actions, rewards_table, observations_table)

        return observations_table, rewards_table, False, self.__environment_info()
