import os

import pandas as pd
import numpy as np
from scipy import integrate
from scipy import signal
import matplotlib.pyplot as plt

from .constanst import *
from .construct_laban_qualities import Laban_Dict

class DataProcessing():
    def __init__(self) -> None:
        pass

    def process_pendulum_trajectory_from_file():

        trajectory = pd.read_csv(pendulum_trajectory_filepath)
        trajectory = trajectory.to_numpy()
        aux_zeros = np.zeros((17,6))
        trajectory = np.concatenate((trajectory,aux_zeros),axis=0)
        timeline = np.linspace(0,len(trajectory),len(trajectory))
        print(np.shape(trajectory))

        fig5, axs5 = plt.subplots(2, 3)
        for n,ax in enumerate(axs5.flatten()):
            ax.plot(timeline,trajectory[:,n],linewidth=2.0)
            if n < 3:
                ax.set(ylabel='Linear velocity (m/s)', xlabel='Time (s)')
            else:
                ax.set(ylabel='Angular velocity (rad/s)', xlabel='Time (s)')
                
        fig5.suptitle('Kinematic Twist Double Pendulum', fontsize=20)
        plt.show()

    def integrate_tensor(self,input_tensor, dx= 0.000016):
        output_arr = []
        x = np.linspace(0, np.shape(input_tensor)[0]/100000, np.shape(input_tensor)[0])
        for cord in range(np.shape(input_tensor)[1]-3):
            integration_arr = integrate.cumulative_trapezoid(input_tensor[:,cord],x)
            integration_arr = signal.detrend(integration_arr)
            integration_arr = np.insert(integration_arr,-1,integration_arr[-1])
            output_arr.append(integration_arr)
        output_arr = np.asarray(output_arr).transpose(1,0)
        input_tensor[:,:3] = output_arr
        return input_tensor

    def filt(self,trajectory):
        b, a = signal.butter(3, 0.0001)
        for cord in range(np.shape(trajectory)[1]):
            trajectory[:,cord] = signal.filtfilt(b, a, trajectory[:,cord])
        return trajectory

    def calc_laban(self,trajectory, flag=False, mass=0.4):
        b, a = signal.butter(3, 0.0001)
        for cord in range(np.shape(trajectory)[1]):
            trajectory[:,cord] = signal.filtfilt(b, a, trajectory[:,cord])
        laban_obj = Laban_Dict(trajectory)
        df, signals_dict = laban_obj.start_process(human=flag, mass=mass)

        return df, signals_dict

    def load_files_from_paths(self, filepaths, integrate=False, save_tag=""):
        self.trajectory_human = np.load(filepaths[0])
        self.trajectory_network = np.load(filepaths[1])
        self.trajectory_robot = pd.read_csv(pendulum_trajectory_filepath)
        self.trajectory_robot = self.trajectory_robot.to_numpy()

        if integrate:
            trajectory_human = self.integrate_tensor(self.trajectory_human)
            np.save(os.path.join(trajectories_human_save_filepath, save_tag + ".npy"), trajectory_human)

    def load_data(self, trajectories, integrate=False, save_tag=""):
        self.trajectory_human = trajectories[0]
        self.trajectory_network = trajectories[1]
        self.trajectory_robot = trajectories[2]

        if integrate:
            trajectory_human = self.integrate_tensor(self.trajectory_human)
            np.save(os.path.join(trajectories_human_save_filepath, save_tag + ".npy"), trajectory_human)

    def start_processing(self):
        print("--------Pendulum Laban Qualities------")
        df, signals_dict_rob = self.calc_laban(self.trajectory_robot)
        print("--------Human Laban Qualities------")
        df, signals_dict = self.calc_laban(self.trajectory_human,flag=True)
        print("--------Network Output Laban Qualities------")
        _,_ = self.calc_laban(self.trajectory_network)

