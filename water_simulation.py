import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from datetime import datetime
import os
from tqdm import tqdm

class WaterWualitySimulation:
    def __init__(
        self,
        sensor_args = [
            {'name': 'region1', 'xpos': 138, 'ypos':413, 'color': [255, 0, 0], 'size':(15, 15)},
            {'name': 'region2', 'xpos': 138, 'ypos':138, 'color': [0, 255, 0], 'size':(15, 15)},
            {'name': 'region3', 'xpos': 413, 'ypos':138, 'color': [0, 0, 255], 'size':(15, 15)},
            {'name': 'region4', 'xpos': 413, 'ypos':413, 'color': [255, 0, 255], 'size':(15, 15)},
        ],
        pond_args = {'width': 550, 'height': 550, 'color':  [177, 220, 234], 'initial_frame': np.full((550, 550, 3), [177, 220, 234], dtype=np.uint8)},
    ):
        self.pond_w = pond_args['width']
        self.pond_h = pond_args['height']
        self.pond_color = pond_args['color']
        
        self.initial__frame = pond_args['initial_frame']
        self.__frameDatas__ = [self.initial__frame]
        
        self.sensors = sensor_args
        self.__calculated_do_rate_change__ = {}
        self.__calcutaed_estimated_do__ = {}
        self.__temp_sensor__ = {}
        self.__pH_sensor__ = {}
    
    def initSensor(self):
        for sensor in self.sensors:
            ypos_temp = sensor['ypos']
            xpos_temp = sensor['xpos']
            width_temp = sensor['size'][0]
            height_temp = sensor['size'][1]
            
            self.initial__frame[ypos_temp - height_temp // 2:ypos_temp + height_temp // 2, xpos_temp - width_temp // 2:xpos_temp + width_temp // 2] = sensor['color']
            self.__frameDatas__.append(self.initial__frame)
            self.__calculated_do_rate_change__[sensor['name']] = []
            self.__calcutaed_estimated_do__[sensor['name']] = []
            self.__temp_sensor__[sensor['name']] = []
            self.__pH_sensor__[sensor['name']] = []
    
    def calculateDOEstimation(self, rate_change, do_before, interval):
        return rate_change * interval + do_before
    
    def dDOdt_reduced(self, DO, T, pH, params=None, debug=False):
        """
        Estimasi laju perubahan DO (mg/L/hari) menggunakan model tereduksi
        hanya dengan input suhu (T) dan pH berdasarkan sintesis dari Mwegoha et al. (2010).
        
        Args:
            DO (float): Kondisi DO awal (mg/L)
            T (float): Suhu air (°C)
            pH (float): pH air
            params (dict, optional): Parameter model yang dapat dikalibrasi.
                Keys:
                - k_r20: reaerasi pada 20°C (1/hari)
                - theta_r: faktor suhu reaerasi
                - mu_max: laju maksimum fotosintesis (mg/L/hari)
                - B_alg: biomassa alga (mg/L)
                - K_pH: konstanta pH untuk fotosintesis
                - pH_opt: pH optimal
                - R0: respirasi ikan dasar (mg/L/hari)
                - k_T: eksponen suhu respirasi ikan
                - R_max: respirasi fitoplankton per biomassa (1/hari)
                - k20: biodegradasi COD pada 20°C (1/hari)
                - theta_d: faktor suhu biodegradasi
                - K_DO: half-saturation DO untuk biodegradasi
                - COD0: konsentrasi COD (mg/L)
        Returns:
            dDO/dt (mg/L/hari)
        """
        # Default parameters (kalibrasi sesuaikan dengan data lapangan)
        if params is None:
            params = {
                'k_aeration': 0.2,
                'k_r20': 0.0015, # sesuai
                'theta_r': 1.024, # sesuai
                'mu_max': 2.2, # sesuai
                'B_alg': 10.0,
                'K_pH': 200, # sesuai
                'pH_opt': 6.8, # sesuai
                'R0': 3.0,
                'k_T': 0.063,
                'R_max': 1.0, 
                # 'R_max': 0.5, # sesuai
                'k20': 0.0015, # sesuai
                'theta_d': 0.25,
                'K_DO': 0.10, # sesuai
                'COD0': 50.0
            }
        
        # (I) Reaerasi
        k_r = (params['k_r20'] + params['k_aeration']) * params['theta_r']**(T - 20)
        DO_sat = 14.652 - 0.41022*T + 0.007991*T**2 - 0.000077774*T**3
        reaerasi = k_r * (DO_sat - DO)
        
        # (II) Fotosintesis (diimplementasikan dengan pH dan suhu)
        f_T = params['theta_r']**(T - 20)
        f_pH = params['K_pH'] / (params['K_pH'] + abs(pH - params['pH_opt']))
        fotosintesis = params['mu_max'] * params['B_alg'] * f_T * f_pH
        
        # (III) Respirasi ikan
        respirasi_ikan = params['R0'] * np.exp(params['k_T'] * T)
        
        # (IV) Respirasi fitoplankton
        # respirasi_fito = params['R_max'] * params['B_alg'] * f_T
        respirasi_fito = params['R_max'] * params['B_alg'] * f_T
        
        # (V) Biodegradasi bahan organik
        biodegradasi = params['k20'] * params['theta_d']**(T - 20) * (DO / (DO + params['K_DO'])) * params['COD0']
        
        # Total
        dDOdt = reaerasi + fotosintesis - respirasi_ikan - respirasi_fito
        
        if debug:
            print("===============================")
            print(f"dDO/dt: {dDOdt}")
            print(f"reaerasi: {reaerasi / 24}")
            print(f"fotosintesis: {fotosintesis / 24}")
            print(f"respirasi_ikan: {respirasi_ikan / 24}")
            print(f"respirasi_fito: {respirasi_fito / 24}")
            print(f"biodegradasi: {biodegradasi / 24}")
        
        return dDOdt
    
    def tempSimulation(self, t):
        suhu_min = 24.5
        suhu_max = 34.0
        T = suhu_min + (suhu_max - suhu_min) * np.sin((np.pi / 24) * (t - 6))
        T = np.clip(T, suhu_min, suhu_max)
        
        return T
    
    def pHSimulation(self, t=None):
        pH_rata2 = 6.5
        fluktuasi = 0.7
        pH = pH_rata2 + fluktuasi * random.random()
        pH = np.clip(pH, 6.5, 8.5)
        
        return pH
    
    def simulate(self, num_sim=24):
        initialDO = {sensor['name']: random.uniform(3, 5) for sensor in self.sensors}

        for time in range(num_sim):
            self.__frameDatas__.append(self.initial__frame)
            for key in self.__calculated_do_rate_change__:
                
                pH = self.pHSimulation(t=time) + random.random() * 1.2 - 0.6
                T = self.tempSimulation(t=time) + random.random() * 2 - 1
                ddo_dt = self.dDOdt_reduced(DO=initialDO[key], pH=pH, T=T) / 24
                DO_estimated = initialDO[key] + ddo_dt
                
                # Simpan data
                self.__calculated_do_rate_change__[key].append(ddo_dt)
                self.__pH_sensor__[key].append(pH)
                self.__temp_sensor__[key].append(T)
                self.__calcutaed_estimated_do__[key].append(DO_estimated)

                initialDO[key] = DO_estimated

    def AnimateAllFrame(self, saveMode=False, file_path=None):
        if file_path is None:
            file_path = f"result/result{datetime.now().strftime('%Y%m%d%H%M%S')}.gif"

        fig, ax = plt.subplots(2, 3, figsize=(12, 6))
        ax = ax.flatten()

        # Titles
        ax[0].set_title("Simulation Temp Condition")
        ax[1].set_title("Temperature Monitoring")
        ax[2].set_title("pH Monitoring")
        ax[3].set_title("Simulation pH Condition")
        ax[4].set_title("DO Rate Change")
        ax[5].set_title("Estimated DO Monitoring")

        # Turn off axis for image plots
        ax[0].axis('off')
        ax[3].axis('off')

        # Set plot limits
        num_frames = len(self.__frameDatas__)
        ax[1].set_xlim(0, num_frames)
        ax[2].set_xlim(0, num_frames)
        ax[4].set_xlim(0, num_frames)
        ax[5].set_xlim(0, num_frames)

        ax[1].set_ylim(15, 35)
        ax[2].set_ylim(6, 8)
        ax[4].set_ylim(-0.5, 0.5)
        ax[5].set_ylim(-1, 5)

        # Image plots
        im0 = ax[0].imshow(self.__frameDatas__[0])
        im1 = ax[3].imshow(self.__frameDatas__[0])

        # Prepare line plots
        lines = {sensor['name']: {} for sensor in self.sensors}
        for sensor in self.sensors:
            color = np.array(sensor['color']) / 255.0
            name = sensor['name']

            lines[name]['T'], = ax[1].plot([], [], label=name, color=color)
            lines[name]['pH'], = ax[2].plot([], [], label=name, color=color)
            lines[name]['DO_rate'], = ax[4].plot([], [], label=name, color=color)
            lines[name]['DO_est'], = ax[5].plot([], [], label=name, color=color)

        for i in [1, 2, 4, 5]:
            ax[i].legend(loc='upper right', fontsize='small')

        def update(idx):
            im0.set_array(self.__frameDatas__[idx])
            im1.set_array(self.__frameDatas__[idx])

            for sensor in self.sensors:
                name = sensor['name']
                x = list(range(idx))

                if len(self.__pH_sensor__[name]) > idx:
                    lines[name]['pH'].set_data(x, self.__pH_sensor__[name][:idx])
                if len(self.__temp_sensor__[name]) > idx:
                    lines[name]['T'].set_data(x, self.__temp_sensor__[name][:idx])
                if len(self.__calculated_do_rate_change__[name]) > idx:
                    lines[name]['DO_rate'].set_data(x, self.__calculated_do_rate_change__[name][:idx])
                if len(self.__calcutaed_estimated_do__[name]) > idx:
                    lines[name]['DO_est'].set_data(x, self.__calcutaed_estimated_do__[name][:idx])

            return [im0, im1] + [line for line_dict in lines.values() for line in line_dict.values()]

        plt.tight_layout()
        ani = FuncAnimation(fig, update, frames=num_frames, interval=200, blit=False)

        if saveMode:
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            ani.save(file_path)

        plt.show()

    def AnimateSensors(self, sensor_names, saveMode=False, file_path=None):
        for sensor_name in sensor_names:
            if sensor_name not in [s['name'] for s in self.sensors]:
                raise ValueError(f"Sensor '{sensor_name}' not found in self.sensors.")

        if file_path is None:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            joined_names = "_".join(sensor_names)
            file_path = f"result/result_{joined_names}_{timestamp}.gif"

        fig, ax = plt.subplots(2, 3, figsize=(12, 6))
        ax = ax.flatten()

        ax[0].set_title("Simulation Temp Condition")
        ax[1].set_title("Temperature Monitoring")
        ax[2].set_title("pH Monitoring")
        ax[3].set_title("Simulation pH Condition")
        ax[4].set_title("DO Rate Change")
        ax[5].set_title("Estimated DO Monitoring")

        ax[0].axis('off')
        ax[3].axis('off')

        num_frames = len(self.__frameDatas__)
        for i in [1, 2, 4, 5]:
            ax[i].set_xlim(0, num_frames)

        ax[1].set_ylim(15, 35)
        ax[2].set_ylim(6, 8)
        ax[4].set_ylim(-0.5, 0.5)
        ax[5].set_ylim(-1, 5)

        # Image plots
        im0 = ax[0].imshow(self.__frameDatas__[0])
        im1 = ax[3].imshow(self.__frameDatas__[0])

        lines = {name: {} for name in sensor_names}
        for sensor_name in sensor_names:
            sensor = next(s for s in self.sensors if s['name'] == sensor_name)
            color = np.array(sensor['color']) / 255.0

            lines[sensor_name]['T'], = ax[1].plot([], [], label=sensor_name, color=color)
            lines[sensor_name]['pH'], = ax[2].plot([], [], label=sensor_name, color=color)
            lines[sensor_name]['DO_rate'], = ax[4].plot([], [], label=sensor_name, color=color)
            lines[sensor_name]['DO_est'], = ax[5].plot([], [], label=sensor_name, color=color)

        for i in [1, 2, 4, 5]:
            ax[i].legend(loc='upper right', fontsize='small')

        def update(idx):
            im0.set_array(self.__frameDatas__[idx])
            im1.set_array(self.__frameDatas__[idx])
            x = list(range(idx + 1))

            for name in sensor_names:
                if len(self.__pH_sensor__[name]) > idx:
                    lines[name]['pH'].set_data(x, self.__pH_sensor__[name][:idx + 1])
                if len(self.__temp_sensor__[name]) > idx:
                    lines[name]['T'].set_data(x, self.__temp_sensor__[name][:idx + 1])
                if len(self.__calculated_do_rate_change__[name]) > idx:
                    lines[name]['DO_rate'].set_data(x, self.__calculated_do_rate_change__[name][:idx + 1])
                if len(self.__calcutaed_estimated_do__[name]) > idx:
                    lines[name]['DO_est'].set_data(x, self.__calcutaed_estimated_do__[name][:idx + 1])

            return [im0, im1] + [line for sensor_line in lines.values() for line in sensor_line.values()]

        plt.tight_layout()
        ani = FuncAnimation(fig, update, frames=num_frames, interval=200, blit=False)

        if saveMode:
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            ani.save(file_path)

        plt.show()

    def AnimateDOOverview(self, saveMode=False, file_path=None):
        if file_path is None:
            file_path = f"result/do_overview_{datetime.now().strftime('%Y%m%d%H%M%S')}.gif"

        fig, ax = plt.subplots(2, 2, figsize=(10, 6))
        ax = ax.flatten()

        # Set judul untuk tiap subplot
        ax[0].set_title("Simulation Temp Frame")
        ax[1].set_title("DO Rate Change")
        ax[2].set_title("Simulation pH Frame")
        ax[3].set_title("Estimated DO")

        # Image axes (frame suhu dan pH)
        ax[0].axis('off')
        ax[2].axis('off')

        # Line axes
        num_frames = len(self.__frameDatas__)
        ax[1].set_xlim(0, num_frames)
        ax[1].set_ylim(-0.5, 0.5)

        ax[3].set_xlim(0, num_frames)
        ax[3].set_ylim(-1, 5)

        # Image data (gunakan frame yang sama untuk suhu dan pH untuk sekarang)
        im_temp = ax[0].imshow(self.__frameDatas__[0])
        im_pH = ax[2].imshow(self.__frameDatas__[0])

        # Siapkan line untuk DO rate dan DO estimated
        lines = {sensor['name']: {} for sensor in self.sensors}
        for sensor in self.sensors:
            name = sensor['name']
            color = np.array(sensor['color']) / 255.0

            lines[name]['DO_rate'], = ax[1].plot([], [], label=name, color=color)
            lines[name]['DO_est'], = ax[3].plot([], [], label=name, color=color)

        ax[1].legend(loc='upper right', fontsize='small')
        ax[3].legend(loc='upper right', fontsize='small')

        def update(idx):
            im_temp.set_array(self.__frameDatas__[idx])
            im_pH.set_array(self.__frameDatas__[idx])

            for sensor in self.sensors:
                name = sensor['name']
                x = list(range(idx + 1))

                if len(self.__calculated_do_rate_change__[name]) > idx:
                    lines[name]['DO_rate'].set_data(x, self.__calculated_do_rate_change__[name][:idx + 1])
                if len(self.__calcutaed_estimated_do__[name]) > idx:
                    lines[name]['DO_est'].set_data(x, self.__calcutaed_estimated_do__[name][:idx + 1])

            return [im_temp, im_pH] + [line for d in lines.values() for line in d.values()]

        plt.tight_layout()
        ani = FuncAnimation(fig, update, frames=num_frames, interval=200, blit=False)

        if saveMode:
            dir_path = os.path.dirname(file_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            ani.save(file_path)

        plt.show()
