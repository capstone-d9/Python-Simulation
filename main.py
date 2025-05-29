from utils import WaterWualitySimulation
import matplotlib.pyplot as plt

waterSim = WaterWualitySimulation()
waterSim.initSensor()
waterSim.simulate(num_sim=12)
# print(waterSim.tell())
waterSim.AnimateFrame()