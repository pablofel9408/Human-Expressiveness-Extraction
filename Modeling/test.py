import numpy as np
import pandas as pd 
import os

import matplotlib.pyplot as plt

# original_trajectory = pd.read_csv("C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Pendulum_Twist_Sim\\Data\\Pendulum_Twist_Swing\\pendulum_twist.csv")
# trajectories_path = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Pendulum_Twist_Sim\\Data\\Pendulum_Twist_Network_Output\\Archive"
# trajectories_path = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Pendulum_Twist_Sim\\Data\\Human_Input\\Archive"
# trajectories_names = "network_output_double_pendulum_same_human_input.npy"
# trajectories_names = "human_random_input.npy"
trajectories_path = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Translation_Dataset_Analysis\\Plots\\Comparisson Experiment\\Experiment 4 - Neutral Style\\Variability Analysis\\Panda_robot\\Trajectories"
trajectories = {}

# for n, (root, dirs, files) in enumerate(os.walk(trajectories_path)):
#     print(files)
#     if n>0:
#         for file in files:
#             if file == trajectories_names:
#                 print(dir_arr)
#                 print(n)
#                 trajectories[dir_arr[n-1]] = np.load(os.path.join(root,file))
#     else:
#         dir_arr = dirs

for n, (root, dirs, files) in enumerate(os.walk(trajectories_path)):
    for file in files:
        name = file.split('.')
        trajectory = np.load(os.path.join(root,file))
        trajectory[1,:,2] = np.where(trajectory[1,:,2]>0,trajectory[1,:,2] - 6.28, trajectory[1,:,2])
        trajectories[name[0]] = np.concatenate((trajectory[0],trajectory[1]),axis=1)

original_trajectory = trajectories["trajectory_org"]
del trajectories["trajectory_org"]
print(np.shape(trajectories["trajectory_100"]))

# original_trajectory = original_trajectory.to_numpy()
# zeros = np.zeros((33,6))
# original_trajectory = np.concatenate((original_trajectory,zeros),axis=0)
shape = np.shape(original_trajectory)
for key,value in trajectories.items():
        rmse = np.sqrt(((original_trajectory - value)**2).mean(axis=0))
        print(f"Key {key} RMSE value {rmse}")


fig, axs = plt.subplots(2,3)
samples = np.linspace(0, shape[0], shape[0])
for key,value in trajectories.items():
    for n,ax in enumerate(axs.flatten()):
        # if key=="Output_seed_10" or key=="Output_seed_73":
        ax.plot(samples,value[:,n], label=key)
        # ax.legend()

for n,ax in enumerate(axs.flatten()):
    ax.plot(samples,original_trajectory[:,n], label="original motion")
    ax.legend()

for n,ax in enumerate(axs.flat):
    if n < 3:
        ax.set(ylabel='V (m/s)', xlabel='Sample')
    else:
        ax.set(ylabel='AV (rad/s)', xlabel='Sample')
fig.suptitle('End Effector Twist - Different Human Inputs')

for ax, col in zip(axs[0], ["X", "Y", "Z"]):
            ax.set_title(col, fontsize=18)
plt.show()

