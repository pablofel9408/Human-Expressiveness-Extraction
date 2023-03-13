import os
import math 
import pickle
import random

import numpy as np 
import pandas as pd 

from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
from scipy import integrate, signal

import torch
from .utilities_mobile import filter_signal_2,find,has_numbers, filter_signal, close_script
from .constants_mecademic import *

class ProcessTrajectory():
    def __init__(self, constants, model,scalers_path, emotion_tag="TRE",participant_tag="NABA",lambda_val=1) -> None:
        self.dataset_constants_human = constants[0]
        self.dataset_constants_robot = constants[1]
        self.model_constants = constants[2]

        self.emotion_tag = emotion_tag
        self.participant_tag = participant_tag
        self.lambda_val = lambda_val

        self.model = model

        self.scalers_path = scalers_path
        self.scalers = []
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def set_paths(self, paths):
        self.paths = paths

    def load_data_scalers(self):

        filepath = find("*.pickle", self.scalers_path)
        for file in filepath:
            if has_numbers(file):
                with open(file, "rb") as input_file:
                        e = pickle.load(input_file)
                self.scalers.append(e)

    def derivate_tensor(self,input_tensor, dx=1000):

        x = np.linspace(0, np.shape(input_tensor)[0]/dx, np.shape(input_tensor)[0])
        # print(x[1]-x[0])
        # print(np.shape(input_tensor)[0]/100000)
        output_arr = []
        for cord in range(np.shape(input_tensor)[1]):
            diff_arr = np.diff(input_tensor[:,cord],n=1)/np.diff(x)
            diff_arr = np.insert(diff_arr,-1,diff_arr[-1])
            output_arr.append(diff_arr)
        return np.asarray(output_arr).transpose(1,0)

    def integrate_tensor(self,input_tensor, dx=1000):
        output_arr = []
        x = np.linspace(0, np.shape(input_tensor)[0]/dx, np.shape(input_tensor)[0])
        for cord in range(np.shape(input_tensor)[1]):
            integration_arr = integrate.cumulative_trapezoid(input_tensor[:,cord], x)
            integration_arr = signal.detrend(integration_arr)
            integration_arr = np.insert(integration_arr,-1,integration_arr[-1])
            output_arr.append(integration_arr)
        return np.asarray(output_arr).transpose(1,0)

    def process_trajectory(self, trajectory_path):
        trajectory = pd.read_csv(trajectory_path)
        trajectory = trajectory.drop(["Unnamed: 0"],axis=1)
        print(trajectory.columns)
        trajectory = trajectory.to_numpy()
        return trajectory
    
    def fix_shape(self,trajectory):
        missing_zeros = math.ceil(np.shape(trajectory)[0]/60)*60 - np.shape(trajectory)[0]
        aux_zeros = np.zeros((missing_zeros,6))
        trajectory = np.concatenate((trajectory,aux_zeros),axis=0)
        return trajectory

    def interpolate_samples(self, trajectory, sample_num=720, end_time=5):
        trajectory_aux = np.zeros((sample_num,6))
        for j in range(0,np.shape(trajectory)[1]):
            time = np.linspace(0, end_time, len(trajectory))
            t, c, k = interpolate.splrep(time, trajectory[:,j], k=3)
            spline = interpolate.BSpline(t, c, k, extrapolate=False)

            data_time = np.linspace(0, end_time, sample_num)
            trajectory_aux[:,j] = spline(data_time)
        return trajectory_aux

    def load_data(self, dataset):
        self.human_dataset = dataset[1]
        self.human_dataset_tags = dataset[2]
        self.robot_dataset = dataset[0]

        if self.model_constants["neutral_style"]:
            self.neutral_style = dataset[3]

    def unscaled_data_hum(self, dataset_hum, tag="train"):
        unscaled_dataset_hum = dataset_hum[tag][0]
        for n ,(key, value) in enumerate(self.scaler.items()):
            unscaled_dataset_hum[:,:,n] = value.inverse_transform(unscaled_dataset_hum[:,:,n])
        return unscaled_dataset_hum

    def generate_human_train_samples(self, trajectory, tag="train"):
        human_dataset_unscaled = self.unscaled_data_hum(self.human_dataset)

        data_neutral_style = np.asarray([self.neutral_style[tag][act] for act in self.human_dataset_tags[tag]["act"].values()])
        names = set(self.human_dataset_tags[tag]["emo"].values())
        names_act = set(self.human_dataset_tags[tag]["act"].values())
        print(set(self.human_dataset_tags[tag]["act"].values()))
        print(names)
        # use a list comprehension, iterating through keys and checking the values match each n
        d = {}
        d_act = {}
        for n in names:
            d[n] = [k for k in self.human_dataset_tags[tag]["emo"].keys() if self.human_dataset_tags[tag]["emo"][k] == n]

        for n in names_act:    
            d_act[n] = [k for k in self.human_dataset_tags[tag]["act"].keys() if self.human_dataset_tags[tag]["act"][k] == n]
        
        idx_list = list(set(d[self.emotion_tag])&set(d_act[self.participant_tag]))
        
        # random_human_indices = [random.randint(0,np.shape(self.human_dataset[tag][0])[0]-1) for _ in range(np.shape(trajectory)[0])]
        # same_human_indices = [random.randint(0,np.shape(self.human_dataset[tag][0])[0]-1)]*np.shape(trajectory)[0]
        # same_human_indices = [73]*np.shape(trajectory)[0]
        random_human_indices = np.random.choice(idx_list,np.shape(trajectory)[0]) 
        same_human_indices = [np.random.choice(idx_list,1)[0]]*np.shape(trajectory)[0]
        print(same_human_indices)
        print("random:",random_human_indices)
        
        human_samples_random = self.human_dataset[tag][0][random_human_indices]
        human_samples_same = self.human_dataset[tag][0][same_human_indices]

        human_samples_random_unscaled = human_dataset_unscaled[random_human_indices]
        human_samples_same_unscaled = human_dataset_unscaled[same_human_indices]


        if self.model_constants["neutral_style"]:
            data_neutral_style = np.asarray([self.neutral_style[tag][act] for act in self.human_dataset_tags[tag]["act"].values()])
            data_neutral_style_random = data_neutral_style[random_human_indices]
            data_neutral_style_same = data_neutral_style[same_human_indices]
            return human_samples_random, human_samples_same, human_samples_random_unscaled, \
                        human_samples_same_unscaled, data_neutral_style_random, data_neutral_style_same,\
                            random_human_indices, same_human_indices         
        else:
            return human_samples_random, human_samples_same,\
                 human_samples_random_unscaled, human_samples_same_unscaled,\
                   random_human_indices, same_human_indices
    
    def generate_network_output(self, trajectory, human_sample, neutral_style=None):
        print(np.shape(trajectory))
        robot_trajectory = torch.tensor(trajectory).to(self.device)
        robot_trajectory_pm = robot_trajectory.permute(0,2,1)

        human_input = torch.tensor(human_sample).to(self.device)
        human_input_pm = human_input.permute(0,2,1)

        if not self.model_constants["neutral_style"]:
            out_hat_twist, z_human, z_robot, mu, log_var, \
                z_human_hat, z_robot_hat, mu_hat,log_var_hat, \
                    cont_att, loc_att_human, loc_att_robot = self.model(robot_trajectory_pm.float(), human_input_pm.float())
        else:
            neutral_style = torch.tensor(neutral_style).to(self.device)
            out_hat_twist, z_human, z_robot, mu, log_var, \
                z_human_hat, z_robot_hat, mu_hat,log_var_hat, \
                    cont_att, loc_att_human, loc_att_robot = self.model(robot_trajectory_pm.float(), human_input_pm.float(),
                                                                        neutral_style)
        
        return out_hat_twist.detach().clone().cpu().numpy()
    
    def filter_signal(self, network_output):
        for k in range(0,np.shape(network_output)[0]):
            network_output[k,:,:] = filter_signal_2(network_output[k])
        return network_output

    def reshape_data(self, input_array):
        output_x = np.array([], dtype=np.int64).reshape(0,6)
        for i in range(np.shape(input_array)[0]):
            output_x = np.vstack([output_x, input_array[i,:,:]])
        return output_x

    def save_array(self, filepath, input_array):
        np.save(filepath,input_array)

    def reshape_array(self, trajectory, samples_num=60):
        shape = np.shape(trajectory)
        print((shape[0]//samples_num,samples_num,shape[1]))
        return np.reshape(trajectory,(shape[0]//samples_num,samples_num,shape[1]))

    def robot_inverse_scale(self,trajectory):
        for i in range(np.shape(trajectory)[1]):
            trajectory[:,i,:] = self.scalers[i].inverse_transform(trajectory[:,i,:])
        return trajectory

    def robot_scale(self,trajectory):
        for i in range(np.shape(trajectory)[1]):
            trajectory[:,i,:] = self.scalers[i].transform(trajectory[:,i,:])
        return trajectory

    def scale_pendulum(self, trajectory):
        self.scaler_pendulum = MinMaxScaler()
        self.scaler_pendulum.fit(trajectory)
        trajectory = self.scaler_pendulum.transform(trajectory)
        return trajectory

    def load_human_scalers(self, dirpath=None):
        if dirpath is None:
            dirpath = os.path.join(self.dataset_constants_human["scalers_path"], "scaler_dataset_human.pickle")

        with open(dirpath, 'rb') as handle:
            scaler = pickle.load(handle)
        self.scaler = scaler

    def generate_plot(self, trajectory, x_label="Sample",
                        y_label1="Linear velocity (m/s)", 
                        y_label2="Angular velocity (rad/s)",
                        title="Kinematic Twist"):

        fig5, axs5 = plt.subplots(2, 3)
        timeline = np.linspace(0,len(trajectory),len(trajectory))
        for n,ax in enumerate(axs5.flatten()):
            ax.plot(timeline,trajectory[:,n],linewidth=2.0)
            if n < 3:
                ax.set(ylabel=y_label1, xlabel=x_label)
            else:
                ax.set(ylabel=y_label2, xlabel=x_label)
                
        fig5.suptitle(title, fontsize=20)
        plt.show()

    def generate_plot_3d_matrix(self, trajectory, x_label="Sample",
                        y_label1="Linear velocity (m/s)", 
                        y_label2="Angular velocity (rad/s)",
                        title="Kinematic Twist"):

        fig5, axs5 = plt.subplots(2, 3)
        timeline = np.linspace(0,np.shape(trajectory)[2],np.shape(trajectory)[2])
        for n,ax in enumerate(axs5.flatten()):
            ax.plot(timeline,trajectory[0,n,:],linewidth=2.0)
            if n < 3:
                ax.set(ylabel=y_label1, xlabel=x_label)
            else:
                ax.set(ylabel=y_label2, xlabel=x_label)
                
        fig5.suptitle(title, fontsize=20)
        plt.show()

    def plot_trajectory(self, pendulum_trajectory, network_output, human_input):
        fig5, axs5 = plt.subplots(6, 3)
        fig, axs = plt.subplots(6, 3)
        figs = [fig5,fig]
        axs = [axs5,axs]
        timeline = np.linspace(0,np.shape(pendulum_trajectory)[1],np.shape(pendulum_trajectory)[1])

        cols = ["Human Motion", "Robot Motion", "Network's Output"]
        for ax, col in zip(axs5[0], cols):
            ax.set_title(col)

        for j, (figr,ax) in enumerate(zip(figs,axs)):
            
            index_oi = j
            out_hat_twist_2 = network_output[index_oi]
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
                    
            figr.suptitle('Kinematic Twist Meca500', fontsize=20)

            for ax, col in zip(ax[0], cols):
                ax.set_title(col)

        plt.show()

    def visualize_network_outputs(self, network_output_same, network_output_random, trajectory):
        timeline = np.linspace(0,np.shape(trajectory)[0],np.shape(trajectory)[0])
        fig5, axs5 = plt.subplots(2, 3)
        for n,ax in enumerate(axs5.flatten()):
            ax.plot(timeline,network_output_same[:,n],linewidth=2.0, color='g', label="Same Human Movement")
            ax.plot(timeline,network_output_random[:,n],linewidth=2.0, color='r', label="Random Human Movements")
            ax.plot(timeline,trajectory[:,n],linewidth=1.0, color="b", label="Original Motion")
            if n < 3:
                ax.set(ylabel='Linear velocity (m/s)', xlabel='Sample')
            else:
                ax.set(ylabel='Angular velocity (rad/s)', xlabel='Sample')
            ax.legend()     
        fig5.suptitle("Kinematic Twist Meca500 - Network's Output", fontsize=20)

        plt.show()

    def start_generation(self, trajectory_path, save=False, cutoff_freq=0.8, fs=60):
        
        self.load_human_scalers()
        trajectory_orign = self.process_trajectory(trajectory_path)
        print(np.shape(trajectory_orign))
        trajectory_orign = filter_signal(trajectory_orign, cutoff_freq=cutoff_freq, fs=fs)
        self.generate_plot(trajectory_orign)

        trajectory_orign_deriv = self.derivate_tensor(trajectory_orign)
        print(np.shape(trajectory_orign_deriv))
        trajectory_orign_deriv = self.fix_shape(trajectory_orign_deriv)
        self.generate_plot(trajectory_orign_deriv)
        
        trajectory = self.interpolate_samples(trajectory_orign_deriv, sample_num=120)
        print(np.shape(trajectory))
        trajectory_scaled = self.scale_pendulum(trajectory)
        self.generate_plot(trajectory_scaled)
        trajectory_scaled = self.reshape_array(trajectory_scaled)

        if not self.model_constants["neutral_style"]:
            human_samples_random, human_samples_same, \
                human_samples_random_unscaled, human_samples_same_unscaled,\
                    random_human_indices, same_human_indices = self.generate_human_train_samples(trajectory_scaled)
            network_output_random = self.generate_network_output(trajectory_scaled,human_samples_random)
            network_output_same = self.generate_network_output(trajectory_scaled,human_samples_same)
        else:
            human_samples_random, human_samples_same, human_samples_random_unscaled, \
                human_samples_same_unscaled, data_neutral_style_random, data_neutral_style_same,\
                    random_human_indices, same_human_indices  = self.generate_human_train_samples(trajectory_scaled)
            network_output_random = self.generate_network_output(trajectory_scaled,human_samples_random,data_neutral_style_random)
            network_output_same = self.generate_network_output(trajectory_scaled,human_samples_same,data_neutral_style_same)

        network_output_random_filt = self.filter_signal(network_output_random)
        network_output_random_filt = np.transpose(network_output_random_filt,(0,2,1))
        network_output_random_filt = self.reshape_data(network_output_random_filt)
        network_output_random_unscaled = self.scaler_pendulum.inverse_transform(network_output_random_filt)

        network_output_same_filt = self.filter_signal(network_output_same)
        network_output_same_filt = np.transpose(network_output_same_filt,(0,2,1))
        network_output_same_filt = self.reshape_data(network_output_same_filt)
        network_output_same_unscaled = self.scaler_pendulum.inverse_transform(network_output_same_filt)

        self.plot_trajectory(trajectory_scaled, network_output_same, human_samples_same)
        self.visualize_network_outputs(network_output_same_unscaled,network_output_random_unscaled,trajectory)

        network_output_random_unscaled = self.interpolate_samples(network_output_random_unscaled, sample_num=np.shape(trajectory_orign)[0])
        network_output_same_unscaled = self.interpolate_samples(network_output_same_unscaled, sample_num=np.shape(trajectory_orign)[0])

        network_output_random_unscaled_position = self.integrate_tensor(network_output_random_unscaled)
        print(np.shape(network_output_random_unscaled_position))
        self.generate_plot(network_output_random_unscaled_position)

        network_output_same_unscaled_position = self.integrate_tensor(network_output_same_unscaled)
        print(np.shape(network_output_same_unscaled_position))
        self.generate_plot(network_output_same_unscaled_position)
        
        human_samples_random_unscaled = self.reshape_data(human_samples_random_unscaled)
        human_samples_random_unscaled = self.interpolate_samples(human_samples_random_unscaled, sample_num=np.shape(trajectory_orign)[0])

        human_samples_same_unscaled = self.reshape_data(human_samples_same_unscaled)
        human_samples_same_unscaled = self.interpolate_samples(human_samples_same_unscaled, sample_num=np.shape(trajectory_orign)[0])
        
        if save:
            trajectory_network_same_filepath_twist_emo = os.path.join(dirpath, trajectory_network_twist_same_filepath)
            trajectory_network_random_filepath_twist_emo = os.path.join(dirpath, trajectory_network_twist_random_filepath)

            trajectory_network_same_filepath_position_emo = os.path.join(dirpath, trajectory_network_position_same_filepath)
            trajectory_network_random_filepath_position_emo = os.path.join(dirpath, trajectory_network_position_random_filepath)
            
            random_trajs_paths_twist = trajectory_network_same_filepath_twist_emo+"_"+str(self.lambda_val)+"_"+self.participant_tag+"_"+self.emotion_tag+".npy"
            same_trajs_paths_twist = trajectory_network_random_filepath_twist_emo+"_"+str(self.lambda_val)+"_"+self.participant_tag+"_"+self.emotion_tag+".npy"
            
            random_trajs_paths_pos = trajectory_network_same_filepath_position_emo+"_"+str(self.lambda_val)+"_"+self.participant_tag+"_"+self.emotion_tag+".npy"
            same_trajs_paths_pos = trajectory_network_random_filepath_position_emo+"_"+str(self.lambda_val)+"_"+self.participant_tag+"_"+self.emotion_tag+".npy"
            
            self.save_array(random_trajs_paths_twist,network_output_random_unscaled)
            self.save_array(same_trajs_paths_twist,network_output_same_unscaled)

            self.save_array(random_trajs_paths_pos,network_output_random_unscaled_position)
            self.save_array(same_trajs_paths_pos,network_output_same_unscaled_position)

        return random_human_indices, same_human_indices
        