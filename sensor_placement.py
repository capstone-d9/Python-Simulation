import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from datetime import datetime
import os
from tqdm import tqdm

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
        
        # pixels
        self.__pondTemp__ = [self.initial__frame.copy()]
        self.__pondpH__ = [self.initial__frame.copy()]
        
        self.sensors = sensor_agents
        self.sensor_agents = {}
        self.__temp_sensor__ = {}
        self.__pH_sensor__ = {}
        
        self.anjing = []
    
    def initSensor(self):
        for sensor in self.sensors:
            ypos_temp = sensor.ypos
            xpos_temp = sensor.xpos
            width_temp = sensor.sensor_width 
            height_temp = sensor.sensor_height
            color = sensor.sensor_color
            
            self.initial__frame[ypos_temp - height_temp // 2:ypos_temp + height_temp // 2, xpos_temp - width_temp // 2:xpos_temp + width_temp // 2] = color
            self.__frameDatas__.append(self.initial__frame)
            
            self.sensor_agents[sensor.name] = sensor
        
    def pixelToPH(self, px):
        val = self.pond_min_pH + (self.pond_max_pH - self.pond_min_pH) * px / 255
        if val <= self.pond_min_pH:
            return self.pond_min_pH
        elif val >= self.pond_max_pH:
            return self.pond_max_pH
        else:
            return val
        
    def pixelToTemp(self, px):
        val = 177 + 78 * (px - self.pond_min_T) / (self.pond_max_T - self.pond_min_T)
        if val <= self.pond_min_T:
            return self.pond_min_T
        elif val >= self.pond_max_T:
            return self.pond_max_T
        else:
            return val
    
    def samplingTemp(self, sensor):
        frame_value = self.__pondTemp__[-1][sensor.ypos, sensor.xpos]
        val = self.pixelToTemp(frame_value[0])
        sensor.takeTempValue(val)
    
    def samplingpH(self, sensor):
        frame_value = self.__pondpH__[-1][sensor.ypos, sensor.xpos]
        val = self.pixelToPH(frame_value[1])
        sensor.takepHValue(val)
        
    def sunHeat(self, t):
        """
        t : waktu dalam menit (0-1440)
        """
        suhu_min = 12
        suhu_max = 34.0

        # Ubah periode sinyal harian (1440 menit = 24 jam)
        T = suhu_min + (suhu_max - suhu_min) * np.sin((np.pi / 720) * (t - 360))  # 360 = jam 6 pagi
        T = np.clip(T, suhu_min, suhu_max)

        return T

    def changeTempbyNeighborhood(self, alpha=1, time=0):
        last_frame = np.pad(self.__pondTemp__[-1].copy(), pad_width=1, mode='constant', constant_values=0)
        new = self.__pondTemp__[-1].copy()
        for h in range(1, self.pond_h + 1):
            for w in range(1, self.pond_w + 1):
                current_value = int(new[h-1, w-1][0])
                north = int(last_frame[h - 1, w][1])
                south = int(last_frame[h + 1, w][1])
                west  = int(last_frame[h, w - 1][1])
                east  = int(last_frame[h, w + 1][1])
                
                sun_T = self.sunHeat(time)
                
                change_value = current_value - (north + south + west + east) / 4 + sun_T * 0.05
                change_value = int(change_value * alpha)
                
                new[h-1, w-1][0] += change_value
                
                if new[h-1, w-1][0] + change_value < 0:
                    new[h-1, w-1][0] = 0
                elif new[h-1, w-1][0] + change_value > 255:
                    new[h-1, w-1][0]
                else:
                    new[h-1, w-1][0] += change_value
                    
        self.__pondTemp__.append(new)
                    
    def changePHbyNeighborhood(self):
        new = self.__pondpH__[-1].copy()
        for h in range(self.pond_h):
            for w in range(self.pond_w):
                change_value = -2 + random.random() * 4
                new[h, w][1] += int(change_value)

        self.__pondpH__.append(new)
    
    def simulate(self, num_iter=5):
        for t in tqdm(range(num_iter)):
            self.changeTempbyNeighborhood(time=t)
            self.changePHbyNeighborhood()
            for sensor in self.sensors:
                self.samplingpH(sensor)
                self.samplingTemp(sensor)
        
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
        ax[5].set_title("Correlation")

        for i in [0, 1, 2]:
            ax[i].axis('off')

        im_temp = ax[0].imshow(self.__pondTemp__[0])
        im_ph = ax[1].imshow(self.__pondpH__[0])
        im_sensor = ax[2].imshow(self.__frameDatas__[-1])
        num_frames = min(len(self.__pondTemp__), len(self.__pondpH__))

        lines = {}
        for sensor in self.sensors:
            color = np.array(sensor.sensor_color) / 255.0
            name = sensor.name
            lines[name] = {
                'T': ax[3].plot([], [], label=name, color=color)[0],
                'pH': ax[4].plot([], [], label=name, color=color)[0]
            }

        for i in [3, 4, 5]:
            ax[i].legend(loc='upper right', fontsize='small')

        def update(idx):
            im_temp.set_array(self.__pondTemp__[idx])
            im_ph.set_array(self.__pondpH__[idx])
            im_sensor.set_array(self.__frameDatas__[-1])

            for sensor in self.sensors:
                name = sensor.name
                temp_vals, ph_vals = sensor.stored_value

                x_vals = list(range(idx))
                lines[name]['T'].set_data(x_vals, temp_vals[:idx])
                lines[name]['pH'].set_data(x_vals, ph_vals[:idx])

            ax[3].relim()
            ax[3].autoscale_view()
            ax[4].relim()
            ax[4].autoscale_view()

            # # Update correlation plot
            # ax[5].cla()
            # ax[5].set_title("Correlation (R²)")
            # temp_corr = self.corelationTemp(idx)
            # ph_corr = self.corelationPH(idx)

            y_labels = []
            bar_values = []
            bar_colors = []

            for sensor in self.sensors:
                name = sensor.name
                y_labels.extend([f"{name} T", f"{name} pH"])
                # bar_values.extend([temp_corr[name], ph_corr[name]])
                sensor_color = np.array(sensor.sensor_color) / 255.0
                bar_colors.extend([sensor_color, sensor_color * 0.7])  # T dan pH dibedakan

            y_pos = np.arange(len(y_labels))
            ax[5].barh(y_pos, bar_values, color=bar_colors)
            ax[5].set_yticks(y_pos)
            ax[5].set_yticklabels(y_labels)
            ax[5].set_xlim(-1, 1)

            return [im_temp, im_ph, im_sensor] + [line for sensor_lines in lines.values() for line in sensor_lines.values()]

        plt.tight_layout()
        ani = FuncAnimation(fig, update, frames=num_frames, interval=200, blit=False)

        if saveMode:
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            ani.save(file_path)

        plt.show()

    def r_squared(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0

    def corelationTemp(self, up_to):
        results = {}
        for sensor in self.sensors:
            temps = sensor.__storedTampvalue__[:up_to]
            time = list(range(len(temps)))
            if len(temps) > 1:
                results[sensor.name] = self.r_squared(time, temps)
            else:
                results[sensor.name] = 0
        return results

    def corelationPH(self, up_to):
        results = {}
        for sensor in self.sensors:
            phs = sensor.__storedpHvalue__[:up_to]
            time = list(range(len(phs)))
            if len(phs) > 1:
                results[sensor.name] = self.r_squared(time, phs)
            else:
                results[sensor.name] = 0
        return results
        
    
def random_initial_temp(
        frame,
        num_regions=5, min_size=30, max_size=100,
        box_color=(0, 220, 234)
    ):
    height, width, c = frame.shape

    for _ in range(num_regions):
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        x = random.randint(0, width - w)
        y = random.randint(0, height - h)

        # Gambar kotak pada area frame
        frame[y:y+h, x:x+w] = box_color

    return frame