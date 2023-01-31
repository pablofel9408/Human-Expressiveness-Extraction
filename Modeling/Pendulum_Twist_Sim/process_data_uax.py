import numpy as np 
from scipy import integrate
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd

from construct_laban_qualities import Laban_Dict

def integrate_tensor(input_tensor, dx= 0.000016):
    output_arr = []
    for cord in range(np.shape(input_tensor)[1]):
        integration_arr = integrate.cumulative_trapezoid(input_tensor[:,cord],dx=dx)
        integration_arr = signal.detrend(integration_arr)
        integration_arr = np.insert(integration_arr,-1,integration_arr[-1])
        output_arr.append(integration_arr)
    return np.asarray(output_arr).transpose(1,0)

def filt(trajectory):
    b, a = signal.butter(3, 0.0001)
    for cord in range(np.shape(trajectory)[1]):
        trajectory[:,cord] = signal.filtfilt(b, a, trajectory[:,cord])
    return trajectory

def calc_laban(trajectory, flag=False):
    b, a = signal.butter(3, 0.0001)
    for cord in range(np.shape(trajectory)[1]):
        trajectory[:,cord] = signal.filtfilt(b, a, trajectory[:,cord])
    laban_obj = Laban_Dict(trajectory)
    df, signals_dict = laban_obj.start_process(human=flag)

    return df, signals_dict

trajectory_same = np.load("C:\\Users\\posorio\\Downloads\\5_dbpendulum_fsm\\5_dbpendulum_fsm\\human_same_input_samples.npy")
trajectory_robot_same = pd.read_csv("C:\\Users\\posorio\\Downloads\\5_dbpendulum_fsm\\5_dbpendulum_fsm\\pendulum_twist.csv")
trajectory_network_same = np.load("C:\\Users\\posorio\\Downloads\\5_dbpendulum_fsm\\5_dbpendulum_fsm\\network_output_double_pendulum_same_human_input.npy")
trajectory_robot_same = trajectory_robot_same.to_numpy()
trajectory_human_old_vel = np.load("C:\\Users\\posorio\\Downloads\\5_dbpendulum_fsm\\5_dbpendulum_fsm\\human_same_input_samples_vel.npy")

print("--------Pendulum Laban Qualities------")
df, signals_dict_rob = calc_laban(trajectory_robot_same)
print("--------Human Laban Qualities------")
df, signals_dict = calc_laban(trajectory_same,flag=True)
print("--------Network Output Laban Qualities------")
calc_laban(trajectory_network_same)

# plt.figure()
# plt.plot(signals_dict["acc"][0])
trajectory_same = integrate_tensor(trajectory_same)
np.save("C:\\Users\\posorio\\Downloads\\5_dbpendulum_fsm\\5_dbpendulum_fsm\human_test.npy", trajectory_same)
plt.figure()
plt.plot(signals_dict["vel"][0])
plt.figure()
plt.plot(trajectory_same)
plt.show()
# plt.figure()
# plt.plot(signals_dict_rob["acc"][0])
# plt.figure()
# plt.plot(signals_dict_rob["vel"][0])
# plt.figure()
# plt.plot(signals_dict_rob["pos"][0])
# plt.show()


# trajectory_same[:,2] = trajectory_same[:,2]/1000

# trajectory_same = integrate_tensor(trajectory_same)
# trajectory_random = integrate_tensor(trajectory_random)
# plt.figure()
# plt.plot(trajectory_same)
# plt.show()

# np.save("C:\\Users\\posorio\\Downloads\\5_dbpendulum_fsm\\5_dbpendulum_fsm\human_same_input_samples.npy", trajectory_same)
# np.save("C:\\Users\\posorio\\Downloads\\5_dbpendulum_fsm\\5_dbpendulum_fsm\\human_random_input_samples.npy", trajectory_random)