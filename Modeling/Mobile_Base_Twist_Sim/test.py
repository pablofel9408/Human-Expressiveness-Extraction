import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utilities_mobile import filter_signal

def visualize_network_outputs(network_output_random, trajectory):
        diff = abs(np.shape(network_output_random)[0] - np.shape(trajectory)[0])
        trajectory = np.concatenate((trajectory,np.zeros((diff,np.shape(trajectory)[1]))),axis=0)
        timeline = np.linspace(0,np.shape(trajectory)[0],np.shape(trajectory)[0])
        fig5, axs5 = plt.subplots(2, 3)
        for n,ax in enumerate(axs5.flatten()):
            ax.plot(timeline,network_output_random[:,n],linewidth=2.0, color='r', label="Network Output")
            ax.plot(timeline,trajectory[:,n],linewidth=1.0, color="b", label="Original Motion")
            if n < 3:
                ax.set(ylabel='Linear velocity (m/s)', xlabel='Sample')
            else:
                ax.set(ylabel='Angular velocity (rad/s)', xlabel='Sample')
            ax.legend()     
        fig5.suptitle("Kinematic Twist Mobile Base - Network's Output", fontsize=20)

        for ax, col in zip(axs5[0], ["X", "Y", "Z"]):
            ax.set_title(col, fontsize=18)

        plt.show()

dirpath = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Mobile_Base_Twist_Sim\\Data"
trajectory = pd.read_csv(os.path.join(dirpath,"Robot\\trajectory_file_s_shape_short.csv"))
trajectory = trajectory.to_numpy()
trajectory = filter_signal(trajectory, cutoff_freq=0.8, fs=60)
# trajectory = trajectory[70:,:]
# np.savetxt("trajectory_file_s_shape_short.csv", trajectory, delimiter=",")

network_output = np.load(os.path.join(dirpath,"Network_Output\\trajectory_s_shape_random_human_input.npy"))
# network_output = network_output[70:,:]

print(np.shape(trajectory))
print(np.shape(network_output))
visualize_network_outputs(network_output,trajectory)