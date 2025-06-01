import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from datetime import datetime
import os
from tqdm import tqdm
from itertools import combinations

class SensorAgent:
    def __init__(self, name, xpos, ypos, w, h, color):
        self.name = name
        self.xpos = xpos
        self.ypos = ypos
        self.sensor_width = w
        self.sensor_height = h
        self.sensor_color = color
        self.__storedTampvalue__ = []
        self.__storedpHvalue__ = []
    
    def takeTempValue(self, val):
        self.__storedTampvalue__.append(val)
    
    def takepHValue(self, val):
        self.__storedpHvalue__.append(val)
    
    @property
    def stored_value(self):
        return self.__storedTampvalue__, self.__storedpHvalue__


class SensorPlacementSimulation:
    def __init__(
        self, 
        sensor_agents,
        pond_args = {
            'width': 550, 'height': 550,
            'color':  [177, 220, 234],
            'max_T': 38,
            'min_T': 24,
            'min_pH': 6.0,
            'max_pH': 8,
            'initial_frame': np.full((550, 550, 3), [177, 220, 234], dtype=np.uint8)
        },        
    ):
        self.pond_w = pond_args['width']
        self.pond_h = pond_args['height']
        self.pond_color = pond_args['color']
        
        self.pond_max_T = pond_args['max_T']
        self.pond_min_T = pond_args['min_T']
        self.pond_min_pH = pond_args['min_pH']
        self.pond_max_pH = pond_args['max_pH']
        
        self.initial__frame = pond_args['initial_frame']
        self.__frameDatas__ = [self.initial__frame.copy()]
        
        self.__pondTemp__ = [self.changeTempFrameToTemp(self.initial__frame.copy())]
        self.__pondpH__ = [self.changePHFrameTopH(self.initial__frame.copy())]
        
        self.__sensor_frames__ = [np.full((550, 550, 3), [177, 220, 234], dtype=np.uint8)]
        
        self.sensors = sensor_agents
        self.sensor_agents = {}
        self.__temp_sensor__ = {}
        self.__pH_sensor__ = {}
        
    def initSensor(self):
        for sensor in self.sensors:
            ypos_temp = sensor.ypos
            xpos_temp = sensor.xpos
            width_temp = sensor.sensor_width 
            height_temp = sensor.sensor_height
            color = sensor.sensor_color
            
            self.initial__frame[ypos_temp - height_temp // 2:ypos_temp + height_temp // 2, xpos_temp - width_temp // 2:xpos_temp + width_temp // 2] = color
            self.__sensor_frames__[0][ypos_temp - height_temp // 2:ypos_temp + height_temp // 2, xpos_temp - width_temp // 2:xpos_temp + width_temp // 2] = color
            self.__frameDatas__.append(self.initial__frame)
            
            self.__sensor_frames__.append(self.__sensor_frames__[0])
            
            self.sensor_agents[sensor.name] = sensor
    
    def samplingTemp(self, sensor):
        frame_value = self.__pondTemp__[-1][sensor.ypos, sensor.xpos]
        val = frame_value
        sensor.takeTempValue(val)
    
    def samplingpH(self, sensor):
        frame_value = self.__pondpH__[-1][sensor.ypos, sensor.xpos]
        val = frame_value
        sensor.takepHValue(val)
    
    def changeTempFrame(self, frame):
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame harus memiliki shape (H, W, 3)")

        red_channel = frame[:, :, 0]
        return red_channel.astype(np.uint8)
    
    def changePHFrame(self, frame):
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame harus memiliki shape (H, W, 3)")

        red_channel = frame[:, :, 1]
        return red_channel.astype(np.uint8)

    def changeTempFrameToTemp(self, frame):
        red_channel = self.changeTempFrame(frame)
        val = self.pond_min_T + (self.pond_max_T - self.pond_min_T) * (red_channel - 177) / 78
        val = np.clip(val, self.pond_min_T, self.pond_max_T)
        return val
    
    def changePHFrameTopH(self, frame):
        green_channel = self.changePHFrame(frame)
        val = self.pond_min_pH + (self.pond_max_pH - self.pond_min_pH) * green_channel / 255
        # val = np.clip(val, self.pond_min_pH, self.pond_max_pH)
        return val
    
    def sunHeat(self, t):
        suhu_min = self.pond_min_T
        suhu_max = self.pond_max_T
        intensity = np.sin((np.pi / 24) * (t - 6))
        # intensity = np.clip(intensity, 0, 1)

        heat_map = np.tile(np.linspace(1.0, 0.6, self.pond_w), (self.pond_h, 1))
        T = suhu_min + (suhu_max - suhu_min) * intensity * heat_map

        return T

    def changeTempbyNeighborhood(self, alpha=0.1, time=0):
        last_frame = np.pad(self.__pondTemp__[-1], pad_width=1, mode='edge')
        new = self.__pondTemp__[-1].copy()
        sun_matrix = self.sunHeat(time)  # bentuknya matriks (pond_h, pond_w)

        for h in range(1, self.pond_h + 1):
            for w in range(1, self.pond_w + 1):
                # Koordinat piksel saat ini (tanpa padding)
                # px, py = w - 1, h - 1
                # aerator_x, aerator_y = 413, 138

                current_value = last_frame[h, w]
                neighbors = [
                    last_frame[h - 1, w],
                    last_frame[h + 1, w],
                    last_frame[h, w - 1],
                    last_frame[h, w + 1],
                    last_frame[h + 1, w + 1],
                    last_frame[h - 1, w + 1],
                    last_frame[h + 1, w - 1],
                    last_frame[h - 1, w - 1],
                ]
                neighbor_avg = sum(neighbors) / 8

                diffusion = alpha * (neighbor_avg - current_value)
                sun_effect = 0.05 * (sun_matrix[h - 1, w - 1] - current_value)

                updated_value = current_value + diffusion + sun_effect
                new[h - 1, w - 1] = updated_value
        
        self.__pondTemp__.append(new)

                 
    def changePHbyNeighborhood(self):
        prev = self.__pondpH__[-1]
        new = prev.copy()

        for h in range(self.pond_h):
            for w in range(self.pond_w):
                neighbors = []

                for dh in [-1, 0, 1]:
                    for dw in [-1, 0, 1]:
                        nh, nw = h + dh, w + dw
                        if 0 <= nh < self.pond_h and 0 <= nw < self.pond_w and (dh != 0 or dw != 0):
                            neighbors.append(prev[nh, nw])

                if neighbors:
                    neighborhood_avg = np.mean(neighbors)
                    noise = np.random.normal(loc=0.0, scale=0.05)
                    new[h, w] = 0.9 * prev[h, w] + 0.1 * neighborhood_avg + noise

        self.__pondpH__.append(new)

    
    def simulate(self, num_iter=5):
        for t in tqdm(range(num_iter)):
            self.changeTempbyNeighborhood(time=t)
            self.changePHbyNeighborhood()
            for sensor in self.sensors:
                self.samplingpH(sensor)
                self.samplingTemp(sensor)
        
    def msedata(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        # ss_res = np.sum((y_true - y_pred) ** 2)
        # ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        # return 1 - ss_res / ss_tot if ss_tot != 0 else 0
        return np.mean((y_pred - y_true)**2)

    def rTemp(self, up_to):
        sensor_names = []
        sensor_val = []
        for sensor1, sensor2 in combinations(self.sensors, 2):
            r_squared = self.r_squared(sensor1.stored_value[0][:up_to], sensor2.stored_value[0][:up_to])
            sensor_names.append(f"{sensor1.name}-{sensor2.name}")
            sensor_val.append(r_squared)
        return sensor_names, sensor_val
    
    def r_squared(self, d1, d2):
        if len(d1) != len(d2):
            raise ValueError("Input lists must have the same length.")
        if len(d1) < 2:
            return 0

        n = len(d1)
        sum_x = sum(d1)
        sum_y = sum(d2)
        sum_x2 = sum(x ** 2 for x in d1)
        sum_y2 = sum(y ** 2 for y in d2)
        sum_xy = sum(x * y for x, y in zip(d1, d2))

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

        if denominator == 0:
            return 0

        r = numerator / denominator
        return r ** 2

    def rpH(self, up_to):
        sensor_names = []
        sensor_val = []
        for sensor1, sensor2 in combinations(self.sensors, 2):
            r_squared = self.r_squared(sensor1.stored_value[1][:up_to], sensor2.stored_value[1][:up_to])
            sensor_names.append(f"{sensor1.name}-{sensor2.name}")
            sensor_val.append(r_squared)
        return sensor_names, sensor_val

    def Animate(self, file_path=None, saveMode=False):
        if file_path is None:
            file_path = f"result/result{datetime.now().strftime('%Y%m%d%H%M%S')}.gif"

        fig, ax = plt.subplots(2, 3, figsize=(12, 6))
        ax = ax.flatten()

        ax[0].set_title("Simulation Temp Condition")
        ax[1].set_title("Simulation pH Condition")
        ax[2].set_title("Sensor Placement")
        ax[3].set_title("Temperature Value Captured")
        ax[4].set_title("pH Value Captured")
        ax[5].set_title("R Squared Score")

        for i in [0, 1, 2]:
            ax[i].axis('off')

        im_temp = ax[0].imshow(self.__pondTemp__[0], cmap='cividis', vmax=self.pond_max_T - 10, vmin=self.pond_min_T + 10)
        fig.colorbar(im_temp, ax=ax[0])
        im_ph = ax[1].imshow(self.__pondpH__[0], cmap='viridis', vmax=self.pond_max_pH, vmin=self.pond_min_pH)
        fig.colorbar(im_ph, ax=ax[1])
        im_sensor = ax[2].imshow(self.__frameDatas__[-1])
        num_frames = min(len(self.__pondTemp__), len(self.__pondpH__))

        # Line plots per sensor
        lines = {}
        for sensor in self.sensors:
            color = np.array(sensor.sensor_color) / 255.0
            name = sensor.name
            lines[name] = {
                'T': ax[3].plot([], [], label=name, color=color)[0],
                'pH': ax[4].plot([], [], label=name, color=color)[0],
            }
        line_temp = ax[5].plot([], [], label="Temperature Corelation", marker='o', color='red')[0]
        line_ph = ax[5].plot([], [], label="pH Corelation", marker='x', color='blue')[0]

        ax[3].legend(loc='upper right', fontsize='small')
        ax[4].legend(loc='upper right', fontsize='small')
        ax[5].legend(loc='upper right', fontsize='small')
        
        def update(idx):
            im_temp.set_array(self.__pondTemp__[idx])
            im_ph.set_array(self.__pondpH__[idx])
            im_sensor.set_array(self.__sensor_frames__[-1])

            updated_artists = [im_temp, im_ph, im_sensor]

            for sensor in self.sensors:
                name = sensor.name
                temp_vals, ph_vals = sensor.stored_value
                x_vals = list(range(min(idx, len(temp_vals))))

                lines[name]['T'].set_data(x_vals, temp_vals[:idx])
                lines[name]['pH'].set_data(x_vals, ph_vals[:idx])

                updated_artists.extend([lines[name]['T'], lines[name]['pH']])
            
            names, temp_cor = self.rTemp(idx)
            names, val_cor = self.rpH(idx)
            line_temp.set_data(list(range(len(names))), temp_cor)
            line_ph.set_data(list(range(len(names))), val_cor)
            
            ax[5].set_xticks(list(range(len(names))))
            ax[5].set_xticklabels(names, rotation=90, ha='right')
            
            updated_artists.extend([line_temp, line_ph])

            for a in [3, 4, 5]:
                ax[a].relim()
                ax[a].autoscale_view()

            return updated_artists

        plt.tight_layout()
        ani = FuncAnimation(fig, update, frames=num_frames, interval=200, blit=False)

        if saveMode:
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            ani.save(file_path)

        plt.show()

def random_initial_temp(
        frame,
        num_regions=5, min_size=30, max_size=100,
        box_color=(random.randint(177, 255), random.randint(220, 225), 234)
    ):
    height, width, c = frame.shape

    for _ in range(num_regions):
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)

        frame[y:y+h, x:x+w] = box_color

    return frame