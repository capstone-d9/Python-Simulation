import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PID_Descret:
    def __init__(self, ki, kd, kp, dt, set_point):
        # tuning param
        self.ki = ki
        self.kd = kd
        self.kp = kp
        self.dt = dt
        
        # initial input pid value
        self.setpoint = set_point
        
        # private value of pid
        self.__prev_err__ = 0
        self.__integral_val__ = 0
        
        # pid log
        self.__input__ = []
        self.__output__ = []
        self.__errlogs__ = []
        
    def _logs(self):
        return self.__input__, self.__output__
        
    def __general_pid__(self, current_val):
        current_error = self.setpoint - current_val
        
        # proporsional
        prop_val = self.kp * current_error
        # integral
        self.__integral_val__ += current_error * self.dt
        integral_val = self.ki * self.__integral_val__
        # derivative
        deriv_val = self.kd * (current_error - self.__prev_err__) / self.dt
        
        self.__prev_err__ = current_error 
        pid_calculation = prop_val + integral_val + deriv_val
        
        # Logging
        self.__input__.append(current_val)
        self.__output__.append(pid_calculation)
        self.__errlogs__.append(current_error)
        
        return pid_calculation

    def simulate_pid(self, steps=50, initial_input_condition=2.5, plant_gain=0.05, gain_eq=None):
        C = initial_input_condition
        for _ in range(steps):
            Q = self.__general_pid__(C)
            delta_C = plant_gain * Q
            C += delta_C
        
        return self.__input__, self.__output__, self.__errlogs__
    
    def animate_response(self, savingMode=False):
        input_log = self.__input__
        output_log = self.__output__
        error_log = self.__errlogs__
        t = np.arange(len(input_log))
        setpoint = self.setpoint  # ambil nilai setpoint

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        titles = ['DO (C)', 'Output PID (Q)', 'Error', 'DO vs Error']

        # Inisialisasi garis untuk animasi
        line_input, = axs[0, 0].plot([], [], color='blue', label='DO (C)')
        line_output, = axs[0, 1].plot([], [], color='green', label='Output PID (Q)')
        line_error, = axs[1, 0].plot([], [], color='red', label='Error')
        line_do_mix, = axs[1, 1].plot([], [], color='blue', label='DO (C)')
        line_err_mix, = axs[1, 1].plot([], [], color='red', linestyle='--', label='Error')

        # Tambahkan garis setpoint pada subplot DO dan DO vs Error
        axs[0, 0].axhline(setpoint, color='black', linestyle=':', label='Setpoint')
        axs[1, 1].axhline(setpoint, color='black', linestyle=':', label='Setpoint')

        # Set sumbu dan properti setiap subplot
        all_vals = input_log + output_log + error_log + [setpoint]
        y_min = min(all_vals) - 1
        y_max = max(all_vals) + 1

        for i, ax in enumerate(axs.flatten()):
            ax.set_xlim(0, len(t))
            ax.set_ylim(y_min, y_max)
            ax.grid(True)
            ax.set_title(titles[i])
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.legend()

        def update(frame):
            line_input.set_data(t[:frame], input_log[:frame])
            line_output.set_data(t[:frame], output_log[:frame])
            line_error.set_data(t[:frame], error_log[:frame])
            line_do_mix.set_data(t[:frame], input_log[:frame])
            line_err_mix.set_data(t[:frame], error_log[:frame])
            return line_input, line_output, line_error, line_do_mix, line_err_mix

        ani = FuncAnimation(fig, update, frames=len(t), interval=100, blit=True)

        plt.tight_layout()
        if savingMode:
            ani.save("pid_simulation.gif", writer='pillow')
        plt.show()

pid = PID_Descret(kp=2.0, ki=1.0, kd=0.5, dt=0.5, set_point=3.0)
pid.simulate_pid(steps=250, initial_input_condition=1.5, plant_gain=0.05)
pid.animate_response()
