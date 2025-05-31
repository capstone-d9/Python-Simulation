import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from datetime import datetime
from tqdm import tqdm

class FishAgent:
    def __init__(
        self,
        xpos,
        ypos,
        direction='upward',
        fish_args={"width": 2, "height": 5, "pad": 1}
    ):
        self.__fish_color__ = [255, 125, 0]
        self.__fish_head_color__ = [125, 0, 0]
        
        self.fish_w = fish_args['width']
        self.fish_h = fish_args['height']
        
        self.xpos = xpos
        self.ypos = ypos
        self.direction = direction

    def moveFish(self, direction='upward', step=1, isSteady=False):
        if not isSteady:
            self.direction = direction
            if direction == 'left':
                self.xpos -= step
            elif direction == 'right':
                self.xpos += step
            elif direction == 'backward':
                self.ypos += step
            else:  # 'upward'
                self.ypos -= step
        else:
            pass

    def FollowMostHightOxygen(self, oxygentSource, stop_at_range=10, fish_step=1):
        fish_x, fish_y = self.xpos, self.ypos

        # Hitung jarak kuadrat ke tiap sumber oksigen
        distances = [(fish_x - ox[0])**2 + (fish_y - ox[1])**2 for ox in oxygentSource]
        min_dist = min(distances)
        min_idx = distances.index(min_dist)
        target = oxygentSource[min_idx]

        # Tentukan arah ke target oksigen
        horizontal_move = 'left' if target[0] < fish_x else 'right'
        vertical_move = 'upward' if target[1] < fish_y else 'backward'

        if min_dist < stop_at_range:
            # 50% tetap
            if random.random() < 0.9:
                self.moveFish(isSteady=True)
            # 50% bergerak
            else:
                # 50% bergerak ke horizontal
                if random.random() < 0.5:
                    self.moveFish(direction=horizontal_move, step=fish_step)
                # 50% bergerak ke vertikal
                else:
                    self.moveFish(direction=vertical_move, step=fish_step)
        else:
            # 50% horizontal,
            if random.random() < 0.5:
                self.moveFish(direction=horizontal_move, step=fish_step)
            # 50% vertikal
            else:
                self.moveFish(direction=vertical_move, step=fish_step)
    
class FishWaterQualitySimulation:
    def __init__(
        self, 
        width_pond = 512,
        height_pond = 512,
        pond_color = [177, 220, 234]  
    ):
        self.__pond_color__ = pond_color
        self.initial__frame = np.full((width_pond, height_pond, 3), pond_color, dtype=np.uint8)
        self.width_pond = width_pond
        self.height_pond = height_pond
        
        self.__frameDatas__ = [self.initial__frame]
    
    def drawFish(
        self, 
        fish_w, 
        fish_h, 
        x_pos, 
        y_pos, 
        fish_color = [255, 125, 0],
        fish_head_color = [125, 0, 0],
        to='upward', 
        frame=None
    ):
        if frame == None:
            frame = self.initial__frame.copy()
        frame_ = frame.copy()
        
        if to == 'left':
            frame_[y_pos - fish_w // 2:y_pos + fish_w // 2, x_pos - fish_h // 2:x_pos + fish_h // 2] = fish_color
            frame_[y_pos - fish_w // 2:y_pos + fish_w // 2, x_pos - fish_h // 2:x_pos - fish_h // 3] = fish_head_color

        elif to == 'right':
            frame_[y_pos - fish_w // 2:y_pos + fish_w // 2, x_pos - fish_h // 2:x_pos + fish_h // 2] = fish_color
            frame_[y_pos - fish_w // 2:y_pos + fish_w // 2, x_pos + fish_h // 3:x_pos + fish_h // 2] = fish_head_color

        elif to == 'backward':
            frame_[y_pos - fish_h // 2:y_pos + fish_h // 2, x_pos - fish_w // 2:x_pos + fish_w // 2] = fish_color
            frame_[y_pos + fish_h // 3:y_pos + fish_h // 2, x_pos - fish_w // 2:x_pos + fish_w // 2] = fish_head_color
            
        else:
            frame_[y_pos - fish_h // 2:y_pos + fish_h // 2, x_pos - fish_w // 2:x_pos + fish_w // 2] = fish_color
            frame_[y_pos - fish_h // 2:y_pos - fish_h // 3, x_pos - fish_w // 2:x_pos + fish_w // 2] = fish_head_color
        
        return frame_
    
    def AnimateFrame(self):
        fig, ax = plt.subplots()
        im = ax.imshow(self.__frameDatas__[0])
        
        def update(frame_idx):
            im.set_array(self.__frameDatas__[frame_idx])
            return [im]
        
        ai = FuncAnimation(fig, update, frames=len(self.__frameDatas__), interval=100, blit=True)
        
        plt.show()
