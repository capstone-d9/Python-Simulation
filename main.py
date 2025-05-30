from utils import WaterWualitySimulation

ecosystem = WaterWualitySimulation(
    sensor_args = [
        {'name': 'region1', 'xpos': 138, 'ypos':413, 'color': [255, 0, 0], 'size':(15, 15)},
        {'name': 'region2', 'xpos': 138, 'ypos':138, 'color': [0, 255, 0], 'size':(15, 15)},
        {'name': 'region3', 'xpos': 413, 'ypos':138, 'color': [0, 0, 255], 'size':(15, 15)},
        {'name': 'region4', 'xpos': 413, 'ypos':413, 'color': [255, 0, 255], 'size':(15, 15)},
    ],
)

# must do
ecosystem.initSensor()
ecosystem.simulate(num_sim=24*2)

# ecosystem.AnimateSensors(sensor_names=['region1', 'region3'])
# ecosystem.AnimateDOOverview()

