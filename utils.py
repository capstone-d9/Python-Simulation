import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from tqdm import tqdm

SIMULATION_PATH = "results"

# untuk nyimpen dan store menjadi svg, mp4 dan lain lain
class ConstructFrame:
    def __init__(self, data: list[dict]):
        self.__frame__ = data
        pass
    
    def Animation(self, name, save_mode=False):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].set_title("Simulation")
        ax[1].set_title("Percentage of Fertilizer on the Farm")
        
        ax[0].axis('off')
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Coverage")
        
        ax[1].set_ylim(0, 1)
        
        
        def update(idx):
            return 1, 1
        
        ani = FuncAnimation(fig, update, frames=len(self.frameStore), interval=50)
        
        
        if save_mode:
            ani.save(f"{self.simulation_result_dir}/{name}")
        
        plt.show()
        

class WaterQuality:
    def __init__(self):
        pass
    
class GenerateSimulation:
    def __init__(self):
        pass