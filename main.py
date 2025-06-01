import numpy as np
from utils import SensorAgent, SensorPlacementSimulation, random_initial_temp, WaterWualitySimulation

frame = np.full((550, 550, 3), [177, 220, 234], dtype=np.uint8)
frame = random_initial_temp(frame=frame, num_regions=5)

sensor_simulation = [
    [
        SensorAgent('region1', 45, 505, 15, 15, [0, 255, 0]),
        SensorAgent('region2', 45, 45, 15, 15, [255, 0, 0]),
        SensorAgent('region3', 505, 505, 15, 15, [255, 255, 0]),
        SensorAgent('region4', 505, 45, 15, 15, [0, 0, 255]),
    ],
    [
        SensorAgent('region1', 45, 45, 15, 15, [255, 0, 0]),
        SensorAgent('region2', 505, 45, 15, 15, [0, 255, 0]),
        SensorAgent('region3', 275, 505, 15, 15, [0, 0, 255]),
    ],
    [
        SensorAgent('region1', 505, 505, 15, 15, [255, 0, 0]),
        SensorAgent('region2', 45, 45, 15, 15, [0, 255, 0]),
    ],
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

sim = SensorPlacementSimulation(sensor_agents=sensor_simulation[2], pond_args=pond_args)
sim.initSensor()
sim.simulate(num_iter=24)
sim.Animate(saveMode=True, file_path=f'result/simulation-with-2-sensors.gif')