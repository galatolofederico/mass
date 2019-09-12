from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
import numpy as np

class Stigmergy (Agent):
    
    def __init__(self, unique_id, model, stig_value, stig_radius, evaporation_value):
        super().__init__(unique_id, model)
        self.init_value = stig_value
        self.init_radius = stig_radius
        self.evaporation = evaporation_value
        self.value = stig_value

    def observe(self):
        neighborhood = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        return neighborhood 

    def move(self, direction):
        pass

    def step(self, direction):
        self.value = (self.value - self.evaporation) if self.value > 0 else 0
        if self.value == 0:
            self.model.stigmergies.remove(self)
            print("%s has been removed" % self.unique_id)
        return self.value


class Prey(Agent):
    def move(self):
        neighborhood = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        print( self.pos)
        new_position = (self.pos[0] + 1, self.pos[1] + 1)
        self.model.grid.move_agent(self, new_position)

    def step(self):
        self.move()


class StigmergyPrey (Model):
    def __init__(self, N, width, height):
        self.stigmergies = []
        self.preys = []
        self.grid = MultiGrid(width, height, True)

        for i in range(N):
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            agent = None
            if i % 2 == 0:
                agent = Stigmergy(i, self, 20*i, i, 1)
                self.stigmergies.append(agent)
            else:
                agent = Prey(i, self)
                self.preys.append(agent)
            
            self.grid.place_agent(agent, (x, y))

    def step(self, directions):
        observations = []
        for stigmergy, direction in zip(self.stigmergies, directions):
            observations.append(stigmergy.step(direction))

        for prey in self.preys:
            prey.step()

        return observations

model = StigmergyPrey(3, 10, 10)
actions = [[1,1], [1,0], [0,1]]

for i in range(100):
    print(model.step(actions))

import matplotlib.pyplot as plt
agent_counts = np.zeros((model.grid.width, model.grid.height))
for cell in model.grid.coord_iter():
    cell_content, x, y = cell
    agent_count = len(cell_content)
    agent_counts[x][y] = agent_count
plt.imshow(agent_counts, interpolation='nearest')
plt.show()
