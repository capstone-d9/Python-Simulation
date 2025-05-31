import numpy as np
from utils import SensorAgent, SensorPlacementSimulation, random_initial_temp

frame = np.full((550, 550, 3), [177, 220, 234], dtype=np.uint8)

frame = random_initial_temp(frame=frame)

sensor_agents = [
    SensorAgent('region1', 138, 413, 15, 15, [255, 0, 0]),
    SensorAgent('region2', 138, 138, 15, 15, [0, 255, 0]),
    SensorAgent('region3', 413, 138, 15, 15, [0, 0, 255]),
    SensorAgent('region4', 413, 413, 15, 15, [255, 0, 255]),
]

pond_args = {
    'width': 550, 'height': 550, 
    'color':  [0, 220, 234],
    'max_T': 38,
    'min_T': 24,
    'min_pH': 6.0,
    'max_pH': 8,
    'initial_frame': frame
}

sim = SensorPlacementSimulation(sensor_agents=sensor_agents, pond_args=pond_args)
sim.initSensor()
sim.simulate(num_iter=12)
sim.Animate(saveMode=True)