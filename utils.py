import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import math as m
import matplotlib.gridspec as gridspec
from datetime import datetime
import os
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

class SensorAgent:
    def __init__(self, name, xpos, ypos, w, h, color):
        self.name = name
        self.xpos = xpos
        self.ypos = ypos
        self.sensor_width = w
        self.sensor_height = h
        self.sensor_color = color
    
    def giveValue(self, val):
        return val

class WaterWualitySimulation:
    def __init__(
        self,
        sensor_args = [
            {'name': 'pH1', 'xpos': 138, 'ypos':413, 'color': [255, 0, 0], 'size':(15, 15)},
            {'name': 'pH2', 'xpos': 138, 'ypos':138, 'color': [255, 0, 0], 'size':(15, 15)},
            {'name': 'pH3', 'xpos': 413, 'ypos':138, 'color': [255, 0, 0], 'size':(15, 15)},
            {'name': 'pH4', 'xpos': 413, 'ypos':413, 'color': [255, 0, 0], 'size':(15, 15)},
        ],
        pond_args = {'width': 550, 'height': 550, 'color':  [177, 220, 234]},
    ):
        self.pond_w = pond_args['width']
        self.pond_h = pond_args['height']
        self.pond_color = pond_args['color']
        
        self.initial__frame = np.full((self.pond_h, self.pond_w, 3), self.pond_color, dtype=np.uint8)
        self.__frameDatas__ = [self.initial__frame]
        
        self.sensors = sensor_args
        self.__calculated_do_rate_change__ = {}
    
    def initSensor(self):
        for sensor in self.sensors:
            ypos_temp = sensor['ypos']
            xpos_temp = sensor['xpos']
            width_temp = sensor['size'][0]
            height_temp = sensor['size'][1]
            
            self.initial__frame[ypos_temp - height_temp // 2:ypos_temp + height_temp // 2, xpos_temp - width_temp // 2:xpos_temp + width_temp // 2] = sensor['color']
            self.__frameDatas__.append(self.initial__frame)
            self.__calculated_do_rate_change__[sensor['name']] = []
            
    def calcutaleDoRateChange(
        self, 
        pH,
        T,
        NH4 = 1.0,
        DO = 5.0,
        params= {
            'k_a20': 0.5, 'theta_a': 1.024,
            'r_nitr': 4.57, 'theta_n': 1.08, 'pK_a': 6.5, 'pK_b': 9.0,
            'mu_max': 10.0, 'theta_f': 1.07, 'I': 1000.0, 'K_I': 200.0, 'A': 5.0,
            'a': 24.715, 'B': 0.5, 'W': 100.0
        }
    ):
        # Reaerasi
        k_a = params['k_a20'] * params['theta_a']**(T - 20)
        DO_sat = 14.652 - 0.41022*T + 0.007991*T**2 - 0.000077774*T**3
        reaerasi = k_a * (DO_sat - DO)
        
        # Nitrifikasi
        fT_n = params['theta_n']**(T - 20)
        fpH = 1 / (1 + 10**(params['pK_a'] - pH)) * 1 / (1 + 10**(pH - params['pK_b']))
        nitrifikasi = params['r_nitr'] * fT_n * fpH * NH4
        
        # Fotosintesis (siang, I/(I+K_I) ≈ 1 jika I >> K_I)
        fT_f = params['theta_f']**(T - 20)
        fotosintesis = params['mu_max'] * fT_f * (params['I']/(params['I'] + params['K_I'])) * params['A']
        
        # Respirasi ikan
        respirasi = params['a'] * params['B'] * params['W']**(-0.237) * np.exp(0.063 * T)
    
        return reaerasi - nitrifikasi + fotosintesis - respirasi
    
    def RandomInputSensor(self, time, pH_initial = 6, T_initial = 26):
        pH = random.random() * (8 - pH_initial) + pH_initial
        T = self.PickTemp(time, base_T=T_initial)
        
        calculated_do_rate_change = self.calcutaleDoRateChange(pH=pH, T=T) / 24
        
        return calculated_do_rate_change
    
    # time in minutes
    def PickTemp(self, time, base_T = 6):
        # Siang hari (pukul 06:00 - 18:00)
        if 6 * 60 <= time <= 18 * 60:
            # Normalisasi waktu siang dari 0 ke pi (pagi ke sore)
            normalized_time = (time - 6 * 60) / (12 * 60) * m.pi
            # Suhu naik mengikuti kurva sinus dari base_T (min) ke base_T + delta_T (maks)
            delta_T = 4  # kenaikan suhu maksimal dari suhu dasar saat siang
            return base_T + delta_T * m.sin(normalized_time)
        
        else:
            return base_T
    
    def simulate(self, num_sim=24*60):
        for time in tqdm(range(num_sim)):
            if time % 60 == 0:
                pass
            
            self.__frameDatas__.append(self.initial__frame)
            for key in self.__calculated_do_rate_change__:
                temp = self.RandomInputSensor(time, pH_initial=random.randint(6, 8), T_initial=random.randint(26, 30))
                self.__calculated_do_rate_change__[key].append(temp)

    def AnimateFrame(self, saveMode=False, file_path=None):
        if file_path is None:
            file_path = f"result/result{datetime.now().strftime('%Y%m%d%H%M%S')}.gif"

        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        ax0 = fig.add_subplot(gs[:, 0])
        ax0.set_title("Simulation")
        ax0.axis('off')

        # Grafik DO Rate Changes
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[0, 2])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[1, 2])

        # Set title dan axis DO
        axs = [ax1, ax2, ax3, ax4]
        max_val = max([max(value) for key, value in self.__calculated_do_rate_change__.items()])
        min_val = min([min(value) for key, value in self.__calculated_do_rate_change__.items()])
        
        len_data = max([len(value) for key, value in self.__calculated_do_rate_change__.items()])

        for ax in axs:
            ax.set_xlabel("Time Hour")
            ax.set_ylabel("Rate Changes (mg/L·jam)")
            ax.set_ylim(min_val - 0.5,
                        max_val + 0.5)
            ax.set_xlim(0, len_data)

        t = np.linspace(1, len_data + 2, num=len_data)

        im = ax0.imshow(self.initial__frame)
        line1, = ax1.plot([], [], color='blue')
        line2, = ax2.plot([], [], color='blue')
        line3, = ax3.plot([], [], color='blue')
        line4, = ax4.plot([], [], color='blue')
        
        def update(idx):
            im.set_array(self.__frameDatas__[idx])
            for key, line in zip(self.__calculated_do_rate_change__, [line1, line2, line3, line4]):
                line.set_data(t[:idx], self.__calculated_do_rate_change__[key][:idx])
            return [im, line1, line2, line3, line4]

        plt.tight_layout()
        ani = FuncAnimation(fig, update, frames=len(self.__frameDatas__), interval=200)
        if saveMode:
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            ani.save(file_path)
        
        plt.show()
