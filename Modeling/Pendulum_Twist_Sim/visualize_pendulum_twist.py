import os
import pathlib

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

import torch 
import random 
import pickle

import utilities
from GAN_Translation.General_Functions.START_gan_proc import ModelingGAN
from Robot_Feature_Extraction.Preprocessing.preprocess_robot_datasets import PreprocessRobotData
from Robot_Feature_Extraction.Preprocessing.process_human_data import PreprocessHumanData

from sklearn.preprocessing import MinMaxScaler
import scipy
from scipy import interpolate
from scipy import integrate
from scipy import signal

seed = 10
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

dir_path = pathlib.Path(__file__).parent / "GAN_Translation\\Config Files"
dir_path_simulation = pathlib.Path(__file__).parent / "Translation_Process_Simulation\\Config Files"
filepath_dataset_hum_cst = os.path.join(dir_path, "human_dataset_config.json")
filepath_dataset_rob_cst = os.path.join(dir_path, "robot_dataset_config.json")
filepath_vae_hum_cst = os.path.join(dir_path, "VAE_human_model_system_config.json")
filepath_vae_rob_cst = os.path.join(dir_path, "VAE_robot_model_system_config.json")
filepath_model_cst = os.path.join(dir_path, "model_system_config.json")
filepath_sim_cst = os.path.join(dir_path_simulation, "simulation_config.json")

dataset_constants_robot = utilities.load_raw_constants(filepath_dataset_rob_cst)
dataset_constants_human = utilities.load_raw_constants(filepath_dataset_hum_cst)
vae_constants_human = utilities.load_raw_constants(filepath_vae_hum_cst)
vae_constants_robot = utilities.load_raw_constants(filepath_vae_rob_cst)
model_constants = utilities.load_raw_constants(filepath_model_cst)
simulation_constants = utilities.load_raw_constants(filepath_sim_cst)
model_constants["model_config"]["generator"]["human_vae"] = vae_constants_human
model_constants["model_config"]["generator"]["robot_vae"] = vae_constants_robot
model_constants["task_name"] = "GAN"

prepros_obj_rob = PreprocessRobotData(dataset_constants_robot)
prepros_obj_rob.start_preprocessing(tag='train')
prepros_obj_hum = PreprocessHumanData(dataset_constants_human)
prepros_obj_hum.load_recursive_pt(dataset_constants_human["dataset_paths"])

dataset_rob = prepros_obj_rob.return_data_()
dataset_hum = prepros_obj_hum.return_data_()
dataset_tags_hum = prepros_obj_hum.return_data_tags()

if dataset_constants_human["expressive_data"]:
    dataset_hum = (dataset_hum,dataset_tags_hum)

dataset = (dataset_rob,dataset_hum)

print(np.shape(dataset_hum["train"]))
dirpath = "C:\\Users\\posorio\\Documents\\Expressive movement\\Data Engineering\\Main Scripts\\Data acquisition\\Human Expressive Data\\Data  Engineering Emotion Dataset\\Data  Engineering Emotion Dataset\\Dataset\\Scalers\\scaler_dataset_human.pickle"
with open(dirpath, 'rb') as handle:
    b = pickle.load(handle)

unscaled_dataset_hum = np.copy(dataset_hum["train"])
for n ,(key, value) in enumerate(b.items()):
    unscaled_dataset_hum[:,:,n] = value.inverse_transform(unscaled_dataset_hum[:,:,n])

trajectory = pd.read_csv("C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\pendulum_twist.csv")
trajectory = trajectory.to_numpy()
aux_zeros = np.zeros((17,6))
trajectory = np.concatenate((trajectory,aux_zeros),axis=0)

fig5, axs5 = plt.subplots(2, 3)
timeline = np.linspace(0,len(trajectory),len(trajectory))
for n,ax in enumerate(axs5.flatten()):
    ax.plot(timeline,trajectory[:,n],linewidth=2.0)
    if n < 3:
        ax.set(ylabel='Linear velocity (m/s)', xlabel='Sample')
    else:
        ax.set(ylabel='Angular velocity (rad/s)', xlabel='Sample')
        
fig5.suptitle('Kinematic Twist Double Pendulum', fontsize=20)
plt.show()

trajectory_aux = np.zeros((720,6))
for j in range(0,np.shape(trajectory)[1]):
    time = np.linspace(0, 5, len(trajectory))
    t, c, k = interpolate.splrep(time, trajectory[:,j], k=3)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)

    data_time = np.linspace(0, 5, 720)
    trajectory_aux[:,j] = spline(data_time)

print(np.shape(trajectory))
print(type(dataset_hum["train"]))
trajectory = np.copy(trajectory_aux)
trajectory_og = np.copy(trajectory)

scaler = MinMaxScaler()
scaler.fit(trajectory)
trajectory = scaler.transform(trajectory)

fig5, axs5 = plt.subplots(2, 3)
timeline = np.linspace(0,len(trajectory),len(trajectory))
for n,ax in enumerate(axs5.flatten()):
    ax.plot(timeline,trajectory[:,n],linewidth=2.0)
    if n < 3:
        ax.set(ylabel='Linear velocity (m/s)', xlabel='Sample')
    else:
        ax.set(ylabel='Angular velocity (rad/s)', xlabel='Sample')
        
fig5.suptitle('Kinematic Twist Double Pendulum', fontsize=20)
plt.show()

shape = np.shape(trajectory)
trajectory = np.reshape(trajectory,(shape[0]//60,60,shape[1]))
print(np.shape(trajectory))
random_human_indices = [random.randint(0,np.shape(dataset_hum["train"])[0]-1) for _ in range(np.shape(trajectory)[0])]
random_human_indices_2 = [random.randint(0,np.shape(dataset_hum["train"])[0]-1)]*np.shape(trajectory)[0]
print(random_human_indices)
print(random_human_indices_2)
print(random_human_indices == random_human_indices_2)
# random_human_indices = [0,300,1000,2200, 500]
human_samples_2 = dataset_hum["train"][random_human_indices]
human_samples = dataset_hum["train"][random_human_indices_2]
# with open('single_human_samples.npy', 'wb') as f:
#     np.save(f, human_samples)
print(len(random_human_indices))

modeling_proc_obj = ModelingGAN(model_constants)
modeling_proc_obj.set_input_data(dataset, expressive=dataset_constants_human["expressive_data"])
epoch, trainLoss, valLoss, model = modeling_proc_obj.load_model(evalFlag=True)
model.eval()

pendulum_trajectory = torch.tensor(trajectory).to(dev)
pendulum_trajectory_pm = pendulum_trajectory.permute(0,2,1)

human_input = torch.tensor(human_samples_2).to(dev)
human_input_pm = human_input.permute(0,2,1)
out_hat_twist_2, z_human, z_robot, mu, log_var, \
    z_human_hat, z_robot_hat, mu_hat,log_var_hat, \
        cont_att, loc_att_human, loc_att_robot = model(pendulum_trajectory_pm.float(), human_input_pm.float())

human_input = torch.tensor(human_samples).to(dev)
human_input_pm = human_input.permute(0,2,1)
out_hat_twist, z_human, z_robot, mu, log_var, \
    z_human_hat, z_robot_hat, mu_hat,log_var_hat, \
        cont_att, loc_att_human, loc_att_robot = model(pendulum_trajectory_pm.float(), human_input_pm.float())

out_hat_twist_2 = out_hat_twist_2.detach().clone().cpu().numpy()
out_hat_twist = out_hat_twist.detach().clone().cpu().numpy()
print("Mean Squared Error :", ((out_hat_twist - out_hat_twist_2)**2).mean())
pendulum_trajectory = pendulum_trajectory.detach().clone().cpu().numpy()
human_input = human_input.detach().clone().cpu().numpy()

for k in range(0,np.shape(out_hat_twist)[0]):
    out_hat_twist[k,:,:] = utilities.filter_signal_2(out_hat_twist[k])
out_hat_twist_og = np.copy(out_hat_twist)

for k in range(0,np.shape(out_hat_twist_2)[0]):
    out_hat_twist_2[k,:,:] = utilities.filter_signal_2(out_hat_twist_2[k])
out_hat_twist_og_2 = np.copy(out_hat_twist_2)

fig5, axs5 = plt.subplots(6, 3)
fig, axs = plt.subplots(6, 3)
fig4, axs4 = plt.subplots(6, 3)
fig3, axs3 = plt.subplots(6, 3)
fig2, axs2 = plt.subplots(6, 3)
figs = [fig5,fig,fig4,fig3,fig2]
axs = [axs5,axs,axs4,axs3,axs2]
timeline = np.linspace(0,np.shape(pendulum_trajectory)[1],np.shape(pendulum_trajectory)[1])

cols = ["Human Motion", "Robot Motion", "Network's Output"]
for ax, col in zip(axs5[0], cols):
    ax.set_title(col)

for j, (figr,ax) in enumerate(zip(figs,axs)):
    
    index_oi = j
    out_hat_twist_2 = out_hat_twist[index_oi]
    pendulum_trajectory_2 = pendulum_trajectory[index_oi]
    human_input_2 = human_input[index_oi]
    ax[0, 0].plot(timeline, human_input_2[:,0],linewidth=2.0)
    ax[0, 1].plot(timeline, pendulum_trajectory_2[:,0],linewidth=2.0)
    ax[0, 2].plot(timeline, out_hat_twist_2[0,:],linewidth=2.0)

    ax[1, 0].plot(timeline, human_input_2[:,1],linewidth=2.0)
    ax[1, 1].plot(timeline, pendulum_trajectory_2[:,1],linewidth=2.0)
    ax[1, 2].plot(timeline, out_hat_twist_2[1,:],linewidth=2.0)

    ax[2, 0].plot(timeline, human_input_2[:,2],linewidth=2.0)
    ax[2, 1].plot(timeline, pendulum_trajectory_2[:,2],linewidth=2.0)
    ax[2, 2].plot(timeline, out_hat_twist_2[2,:],linewidth=2.0)

    ax[3, 0].plot(timeline, human_input_2[:,3],linewidth=2.0)
    ax[3, 1].plot(timeline, pendulum_trajectory_2[:,3],linewidth=2.0)
    ax[3, 2].plot(timeline, out_hat_twist_2[3,:],linewidth=2.0)

    ax[4, 0].plot(timeline, human_input_2[:,4],linewidth=2.0)
    ax[4, 1].plot(timeline, pendulum_trajectory_2[:,4],linewidth=2.0)
    ax[4, 2].plot(timeline, out_hat_twist_2[4,:],linewidth=2.0)

    ax[5, 0].plot(timeline, human_input_2[:,4],linewidth=2.0)
    ax[5, 1].plot(timeline, pendulum_trajectory_2[:,5],linewidth=2.0)
    ax[5, 2].plot(timeline, out_hat_twist_2[5,:],linewidth=2.0)

    for n,axd in enumerate(ax.flatten()):
        if n in [0,3,6]:
            axd.set(ylabel='LA (m/s^2)', xlabel='Sample')
        elif n < 9:
            axd.set(ylabel='LV (m/s)', xlabel='Sample')
        else:
            axd.set(ylabel='AV (rad/s)', xlabel='Sample')
            
    figr.suptitle('Kinematic Twist Double Pendulum', fontsize=20)

    for ax, col in zip(ax[0], cols):
        ax.set_title(col)

plt.show()

fig5, axs5 = plt.subplots(1, 5)
timeline = np.linspace(0,np.shape(pendulum_trajectory)[1],np.shape(pendulum_trajectory)[1])
for n,ax in enumerate(axs5.flatten()):
    ax.plot(timeline,out_hat_twist[n,0,:],linewidth=2.0)

output_x = np.array([], dtype=np.int64).reshape(0,6)
out_hat_twist_og = np.transpose(out_hat_twist_og,(0,2,1))
for i in range(np.shape(out_hat_twist_og)[0]):
    output_x = np.vstack([output_x, out_hat_twist_og[i,:,:]])

output_y = np.array([], dtype=np.int64).reshape(0,6)
out_hat_twist_og_2 = np.transpose(out_hat_twist_og_2,(0,2,1))
for i in range(np.shape(out_hat_twist_og_2)[0]):
    output_y = np.vstack([output_y, out_hat_twist_og_2[i,:,:]])

print("Output X shape: ",np.shape(output_x))
# out_hat_twist_og = np.transpose(out_hat_twist_og,(0,2,1))
# shape = np.shape(out_hat_twist_og)
# print(shape)
# out_hat_twist = np.reshape(out_hat_twist_og,(shape[0]*shape[1],shape[2]))
# sos = scipy.signal.butter(2, 5, 'lp', fs=60,  output='sos')
# for ax in range(np.shape(out_hat_twist)[1]):
#     out_hat_twist[:,ax] = scipy.signal.sosfiltfilt(sos,out_hat_twist[:,ax])
out_hat_twist = np.copy(output_x)
out_hat_twist = scaler.inverse_transform(out_hat_twist)
out_hat_twist_2 = scaler.inverse_transform(output_y)
timeline = np.linspace(0,np.shape(out_hat_twist)[0],np.shape(out_hat_twist)[0])
fig5, axs5 = plt.subplots(2, 3)
for n,ax in enumerate(axs5.flatten()):
    ax.plot(timeline,out_hat_twist[:,n],linewidth=2.0, color='g', label="Same Human Movement")
    ax.plot(timeline,out_hat_twist_2[:,n],linewidth=2.0, color='r', label="Random Human Movements")
    ax.plot(timeline,trajectory_og[:,n],linewidth=1.0, color="b", label="Original Motion")
    if n < 3:
        ax.set(ylabel='Linear velocity (m/s)', xlabel='Sample')
    else:
        ax.set(ylabel='Angular velocity (rad/s)', xlabel='Sample')
    ax.legend()     
fig5.suptitle("Kinematic Twist Double Pendulum - Network's Output", fontsize=20)

trajectory_aux = np.zeros((499680,6))
for j in range(0,np.shape(out_hat_twist)[1]):
    time = np.linspace(0, 5, 720)
    t, c, k = interpolate.splrep(time, out_hat_twist[:,j], k=3)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)

    data_time = np.linspace(0, 5, 499680)
    trajectory_aux[:,j] = spline(data_time)
np.save("network_output_double_pendulum_same_human_input.npy",trajectory_aux)

trajectory_aux = np.zeros((499680,6))
for j in range(0,np.shape(out_hat_twist_2)[1]):
    time = np.linspace(0, 5, 720)
    t, c, k = interpolate.splrep(time, out_hat_twist_2[:,j], k=3)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)

    data_time = np.linspace(0, 5, 499680)
    trajectory_aux[:,j] = spline(data_time)
np.save("network_output_double_pendulum_random_human_input_samples.npy",trajectory_aux)

fig6, axs6 = plt.subplots(2, 3)
for n,ax in enumerate(axs6.flatten()):
    ax.plot(timeline,trajectory_og[:,n],linewidth=2.0)
    if n < 3:
        ax.set(ylabel='Linear velocity (m/s)', xlabel='Sample')
    else:
        ax.set(ylabel='Angular velocity (rad/s)', xlabel='Sample')
        
fig6.suptitle('Kinematic Twist Double Pendulum', fontsize=20)

plt.show()

unscaled_dataset_hum_1 = unscaled_dataset_hum[random_human_indices]
unscaled_dataset_hum_samp = np.array([], dtype=np.int64).reshape(0,6)
for i in range(np.shape(unscaled_dataset_hum_1)[0]):
    unscaled_dataset_hum_samp = np.vstack([unscaled_dataset_hum_samp, unscaled_dataset_hum_1[i,:,:]])

trajectory_aux = np.zeros((499680,6))
for j in range(0,np.shape(unscaled_dataset_hum_samp)[1]):
    time = np.linspace(0, 5, 720)
    t, c, k = interpolate.splrep(time, unscaled_dataset_hum_samp[:,j], k=3)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)

    data_time = np.linspace(0, 5, 499680)
    trajectory_aux[:,j] = spline(data_time)
np.save("human_random_input_samples.npy",trajectory_aux)

fig5, axs5 = plt.subplots(2, 3)
timeline = np.linspace(0,len(unscaled_dataset_hum_samp),len(unscaled_dataset_hum_samp))
for n,ax in enumerate(axs5.flatten()):
    ax.plot(timeline,unscaled_dataset_hum_samp[:,n],linewidth=2.0)
    if n < 3:
        ax.set(ylabel='Linear acceleration (m/s^2)', xlabel='Sample')
    else:
        ax.set(ylabel='Angular velocity (rad/s)', xlabel='Sample')
        
fig5.suptitle('IMU Human', fontsize=20)
plt.show()

unscaled_dataset_hum_2 = unscaled_dataset_hum[random_human_indices_2]
unscaled_dataset_hum_samp_2 = np.array([], dtype=np.int64).reshape(0,6)
for i in range(np.shape(unscaled_dataset_hum_2)[0]):
    unscaled_dataset_hum_samp_2 = np.vstack([unscaled_dataset_hum_samp_2, unscaled_dataset_hum_2[i,:,:]])

trajectory_aux = np.zeros((499680,6))
for j in range(0,np.shape(unscaled_dataset_hum_samp_2)[1]):
    time = np.linspace(0, 5, 720)
    t, c, k = interpolate.splrep(time, unscaled_dataset_hum_samp_2[:,j], k=3)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)

    data_time = np.linspace(0, 5, 499680)
    trajectory_aux[:,j] = spline(data_time)
np.save("human_same_input_samples.npy",trajectory_aux)

fig5, axs5 = plt.subplots(2, 3)
timeline = np.linspace(0,len(unscaled_dataset_hum_samp_2),len(unscaled_dataset_hum_samp_2))
for n,ax in enumerate(axs5.flatten()):
    ax.plot(timeline,unscaled_dataset_hum_samp_2[:,n],linewidth=2.0)
    if n < 3:
        ax.set(ylabel='Linear acceleration (m/s^2)', xlabel='Sample')
    else:
        ax.set(ylabel='Angular velocity (rad/s)', xlabel='Sample')
        
fig5.suptitle('IMU Human', fontsize=20)
plt.show()