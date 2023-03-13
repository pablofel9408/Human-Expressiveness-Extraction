import os

import pickle
import numpy as np
from numpy.linalg import norm
import pandas as pd

import random
import time

import matplotlib.pyplot as plt
from scipy import integrate
from scipy import signal
from scipy.stats import pearsonr

import swift
import roboticstoolbox as rp
import spatialgeometry as sg
import spatialmath as sm
import spatialmath.base as tr

import torch

from tslearn.preprocessing import TimeSeriesScalerMinMax

import utilities

class Simulation_Methods_Multi():
    def __init__(self, constants, model, scalers_path, emotion_tag="TRE",participant_tag="NABA",lambda_val=1) -> None:

        self.dx = 0.016
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.cst = constants
        self.scalers_path = scalers_path
        self.generation_model = model
        self.human_data = None
        self.robot_data = None
        self.scalers = []

        self.emotion_tag = emotion_tag
        self.participant_tag = participant_tag
        self.lambda_val = lambda_val
        
        if model:
            self.translation_model = model.to(self.device)

    def set_model(self, model):
        self.translation_model = model.to(self.device)

    def set_environment(self, no_env=False):

        if self.cst["robot"]=="manipulator":

            # Create a Panda robot object
            self.panda_orign = rp.models.Panda()

            # Set joint angles to ready configuration
            self.panda_orign.q = self.panda_orign.qr

             # Create a Panda robot object
            self.panda_network_out = rp.models.Panda()

            # Set joint angles to ready configuration
            self.panda_network_out.q = self.panda_network_out.qr

            if not no_env:
                self.env = swift.Swift()
                self.env.launch(realtime=True)

                # Add the Panda to the simulator
                self.env.add(self.panda_orign)
                self.env.add(self.panda_network_out)

    def set_input_data(self,input_data, neutral_data=None, tag="val", neutral=False):
        self.human_data = input_data[1][tag][0]
        self.robot_data = input_data[0][tag]
        self.dataset_tags = input_data[2][tag]

        if neutral:
            self.neutral_style = neutral_data[tag]

    def start_simulation(self, neutral=False, no_env=False, calc_similarity=False):
        
        self.load_data_scalers()
        dataset_idx = self.generate_poses(neutral)
        # self.env_add_traj_map()
         
        traj_plan_poses_orign_dict, traj_plan_poses_gan_dict = self.control_loop(real=True, no_env=no_env)

        output_vel_real = traj_plan_poses_orign_dict["output_vel"]
        output_qd_real = traj_plan_poses_orign_dict["output_qd"]
        output_q_real = traj_plan_poses_orign_dict["output_q"]
        output_pose_real = traj_plan_poses_orign_dict["output_pose"]
        output_roation_real = traj_plan_poses_orign_dict["output_rotation"]
        error_real = traj_plan_poses_orign_dict["error"]

        output_vel = traj_plan_poses_gan_dict["output_vel"]
        output_qd = traj_plan_poses_gan_dict["output_qd"]
        output_q = traj_plan_poses_gan_dict["output_q"]
        output_pose = traj_plan_poses_gan_dict["output_pose"]
        output_roation = traj_plan_poses_gan_dict["output_rotation"]
        error = traj_plan_poses_gan_dict["error"]

        save_file_tag = self.participant_tag+"_"+self.emotion_tag+"_"+str(self.lambda_val)+"_seed_"+dataset_idx+".npy"
        filepath = "C:\\Users\\posorio\\OneDrive - 国立研究開発法人産業技術総合研究所\\Documents\\Expressive movement\\Modeling\\Translation_Process_Simulation\\Data"
        np.save(os.path.join(filepath,"robot_motion_twist_"+save_file_tag),output_vel_real)
        np.save(os.path.join(filepath,"robot_motion_joint_vel_"+save_file_tag),output_qd_real)
        np.save(os.path.join(filepath,"robot_motion_joint_"+save_file_tag),output_q_real)
        np.save(os.path.join(filepath,"robot_motion_pose_"+save_file_tag),output_pose_real)
        np.save(os.path.join(filepath,"robot_motion_rotation_"+save_file_tag),output_roation_real)

        np.save(os.path.join(filepath,"network_output_twist_"+save_file_tag),output_vel)
        np.save(os.path.join(filepath,"network_output_joint_vel_"+save_file_tag),output_qd)
        np.save(os.path.join(filepath,"network_output_joint_"+save_file_tag),output_q)
        np.save(os.path.join(filepath,"network_output_pose_"+save_file_tag),output_pose)
        np.save(os.path.join(filepath,"network_output_rotation_"+save_file_tag),output_roation)

        if self.cst["visualize"]:
            # self.visualize_trajectories(output_vel_real, output_qd_real, 
            #                             output_q_real, output_pose_real, error_real, label="Real")

            self.visualize_trajectories((output_vel,output_vel_real), (output_qd, output_qd_real), 
                                        (output_q, output_q_real), label="Network Output ")

            self.visualize_translated_traj((output_pose_real, output_roation_real), 
                                            (output_pose, output_roation), label="Both Trajectories")

        if calc_similarity:
            cosine = np.sum(output_vel_real*output_vel, axis=0)/(norm(output_vel_real, axis=0)*norm(output_vel, axis=0))
            print(cosine)

            for i in range(np.shape(output_vel)[1]):
                print(pearsonr(output_vel_real[:,i],output_vel[:,i]))

        if self.cst["save_sim"]:
            self.save_data()

    def load_data_scalers(self):

        filepath = utilities.find("*.pickle", self.scalers_path)
        for file in filepath:
            if utilities.has_numbers(file):
                with open(file, "rb") as input_file:
                        e = pickle.load(input_file)
                self.scalers.append(e)

    def visualize_translated_traj(self, output_pose1, output_pose2, label=""):

        # dirpath = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Translation_Dataset_Analysis\\Plots\\Comparisson Experiment\\Experiment 4 - Neutral Style\\Variability Analysis\\Panda_robot\\Trajectories"
        # np.save(os.path.join(dirpath,"trajectory_700.npy"), output_pose2)
        # np.save(os.path.join(dirpath,"trajectory_org"".npy"),output_pose1)

        fig0 = plt.figure()
        ax0 = plt.axes(projection ='3d')
        ax0.plot3D(output_pose1[0][:,0], output_pose1[0][:,1], 
                    output_pose1[0][:,2], 'green',linewidth=2.0, label="Robot Motion")
        ax0.plot3D(output_pose2[0][:,0], output_pose2[0][:,1], 
                    output_pose2[0][:,2], 'red',linewidth=2.0, label="Network Output")
        # ax0.set(ylabel='meters', xlabel='meter', zlabel='meter')
        fig0.suptitle('End Effector Position', fontsize=20)
        fig0.legend(loc="center right")

        sample = np.linspace(0,len(output_pose1[1]),len(output_pose1[1]))
        figA, axsA = plt.subplots(1, 3)
        axsA[0].plot(sample, output_pose1[1][:,0],linewidth=2.0, label="Robot Motion")
        axsA[0].plot(sample, output_pose2[1][:,0],linewidth=2.0, label="Network Output")
        axsA[0].legend(loc="upper right")
        axsA[1].plot(sample, output_pose1[1][:,1],linewidth=2.0, label="Robot Motion")
        axsA[1].plot(sample, output_pose2[1][:,1],linewidth=2.0, label="Network Output")
        axsA[1].legend(loc="upper right")
        axsA[2].plot(sample, output_pose1[1][:,2], linewidth=2.0, label="Robot Motion")
        axsA[2].plot(sample, output_pose2[1][:,2],linewidth=2.0, label="Network Output")
        axsA[2].legend(loc="upper right")
        for n,ax in enumerate(axsA.flat):
            ax.set(ylabel='Angular velocity (rad/s)', xlabel='Samples')
        figA.suptitle('End Effector Twist Error')
        figA.suptitle('End Effector Rotation', fontsize=20)

        plt.show()

        if self.cst["save_sim"]:
            fig0.savefig(label + ' ' + self.cst["simulation_label"] + ' End Effector Position.png')

    def visualize_trajectories(self, output_vel, output_qd, output_q,label=""):

        # sample = np.linspace(0,len(error),len(error))
        # fig1, axs1 = plt.subplots(3, 2)
        # axs1[0, 0].plot(sample, error[:,0])
        # axs1[0, 1].plot(sample, error[:,3])
        # axs1[1, 0].plot(sample, error[:,1])
        # axs1[1, 1].plot(sample, error[:,4])
        # axs1[2, 0].plot(sample, error[:,2])
        # axs1[2, 1].plot(sample, error[:,5])
        # fig1.suptitle('End Effector Twist Error')

        sample = np.linspace(0,len(output_qd[0]),len(output_qd[0]))
        fig2, axs2 = plt.subplots(4, 2)
        axs2[0, 0].plot(sample, output_qd[0][:,0],'red',linewidth=2.0, label="Network Output")
        axs2[0, 0].plot(sample, output_qd[1][:,0],'green',linewidth=2.0, label="Robot Motion")
        axs2[0, 0].legend(loc="upper right")

        axs2[0, 1].plot(sample, output_qd[0][:,1],'red',linewidth=2.0, label="Network Output")
        axs2[0, 1].plot(sample, output_qd[1][:,1],'green',linewidth=2.0, label="Robot Motion")
        axs2[0, 1].legend(loc="upper right")

        axs2[1, 0].plot(sample, output_qd[0][:,2],'red',linewidth=2.0, label="Network Output")
        axs2[1, 0].plot(sample, output_qd[1][:,2],'green',linewidth=2.0, label="Robot Motion")
        axs2[1, 0].legend(loc="upper right")

        axs2[1, 1].plot(sample, output_qd[0][:,3],'red',linewidth=2.0, label="Network Output")
        axs2[1, 1].plot(sample, output_qd[1][:,3],'green',linewidth=2.0, label="Robot Motion")
        axs2[1, 1].legend(loc="upper right")

        axs2[2, 0].plot(sample, output_qd[0][:,4],'red',linewidth=2.0, label="Network Output")
        axs2[2, 0].plot(sample, output_qd[1][:,4],'green',linewidth=2.0, label="Robot Motion")
        axs2[2, 0].legend(loc="upper right")

        axs2[2, 1].plot(sample, output_qd[0][:,5],'red',linewidth=2.0, label="Network Output")
        axs2[2, 1].plot(sample, output_qd[1][:,5],'green',linewidth=2.0, label="Robot Motion")
        axs2[2, 1].legend(loc="upper right")

        axs2[3, 0].plot(sample, output_qd[0][:,6],'red',linewidth=2.0, label="Network Output")
        axs2[3, 0].plot(sample, output_qd[1][:,6],'green',linewidth=2.0, label="Robot Motion")
        axs2[3, 0].legend(loc="upper right")

        axs2[0, 0].title.set_text('Joint 1 Velocities')
        axs2[0, 1].title.set_text('Joint 2 Velocities')
        axs2[1, 0].title.set_text('Joint 3 Velocities')
        axs2[1, 1].title.set_text('Joint 4 Velocities')
        axs2[2, 0].title.set_text('Joint 5 Velocities')
        axs2[2, 1].title.set_text('Joint 6 Velocities')
        axs2[3, 0].title.set_text('Joint 7 Velocities')

        for n,ax in enumerate(axs2.flat):
            ax.set(ylabel='Angular velocity (rad/s)', xlabel='Samples')
        
        fig2.suptitle('Joints Tracking Velocities', fontsize=20)

        sample = np.linspace(0,len(output_q[0]),len(output_q[0]))
        fig3, axs3 = plt.subplots(4, 2)
        axs3[0, 0].plot(sample, output_q[0][:,0],'red',linewidth=2.0, label="Network Output") 
        axs3[0, 0].plot(sample,output_q[1][:,0],'green',linewidth=2.0, label="Robot Motion")
        axs3[0, 0].legend(loc="upper right")
        
        axs3[0, 1].plot(sample, output_q[0][:,1],'red',linewidth=2.0, label="Network Output")
        axs3[0, 1].plot(sample, output_q[1][:,1],'green',linewidth=2.0, label="Robot Motion")
        axs3[0, 1].legend(loc="upper right")

        axs3[1, 0].plot(sample, output_q[0][:,2],'red',linewidth=2.0, label="Network Output")
        axs3[1, 0].plot(sample,output_q[1][:,2],'green',linewidth=2.0, label="Robot Motion")
        axs3[1, 0].legend(loc="upper right")
        
        axs3[1, 1].plot(sample, output_q[0][:,3],'red',linewidth=2.0, label="Network Output")
        axs3[1, 1].plot(sample,output_q[1][:,3],'green',linewidth=2.0, label="Robot Motion")
        axs3[1, 1].legend(loc="upper right")

        axs3[2, 0].plot(sample, output_q[0][:,4],'red',linewidth=2.0, label="Network Output")
        axs3[2, 0].plot(sample,output_q[1][:,4],'green',linewidth=2.0, label="Robot Motion")
        axs3[2, 0].legend(loc="upper right")

        axs3[2, 1].plot(sample, output_q[0][:,5],'red',linewidth=2.0, label="Network Output")
        axs3[2, 1].plot(sample, output_q[1][:,5],'green',linewidth=2.0, label="Robot Motion")
        axs3[2, 1].legend(loc="upper right")

        axs3[3, 0].plot(sample, output_q[0][:,6],'red' ,linewidth=2.0, label="Network Output")
        axs3[3, 0].plot(sample, output_q[1][:,6],'green',linewidth=2.0, label="Robot Motion")
        axs3[3, 0].legend(loc="upper right")

        axs3[0, 0].title.set_text('Joint 1 Position')
        axs3[0, 1].title.set_text('Joint 2 Position')
        axs3[1, 0].title.set_text('Joint 3 Position')
        axs3[1, 1].title.set_text('Joint 4 Position')
        axs3[2, 0].title.set_text('Joint 5 Position')
        axs3[2, 1].title.set_text('Joint 6 Position')
        axs3[3, 0].title.set_text('Joint 7 Position')

        for n,ax in enumerate(axs3.flat):
            ax.set(ylabel='Position (rad)', xlabel='Samples')
        fig3.suptitle('Joints Tracking Position', fontsize=20)

        sample = np.linspace(0,len(output_vel[0]),len(output_vel[0]))
        fig4, axs4 = plt.subplots(3, 2)
        axs4[0, 0].plot(sample, output_vel[0][:,0],'red',linewidth=2.0, label="Network Output")
        axs4[0, 0].plot(sample, output_vel[1][:,0],'green',linewidth=2.0, label="Robot Motion")
        axs4[0, 0].legend(loc="upper right")
        
        axs4[0, 1].plot(sample, output_vel[0][:,3],'red', linewidth=2.0, label="Network Output")
        axs4[0, 1].plot(sample, output_vel[1][:,3],'green',linewidth=2.0, label="Robot Motion")
        axs4[0, 1].legend(loc="upper right")

        axs4[1, 0].plot(sample, output_vel[0][:,1],'red',linewidth=2.0, label="Network Output")
        axs4[1, 0].plot(sample, output_vel[1][:,1],'green',linewidth=2.0, label="Robot Motion")
        axs4[1, 0].legend(loc="upper right")
        
        axs4[1, 1].plot(sample, output_vel[0][:,4],'red',linewidth=2.0, label="Network Output")
        axs4[1, 1].plot(sample, output_vel[1][:,4],'green',linewidth=2.0, label="Robot Motion")
        axs4[1, 1].legend(loc="upper right")
        
        axs4[2, 0].plot(sample, output_vel[0][:,2],'red',linewidth=2.0, label="Network Output")
        axs4[2, 0].plot(sample, output_vel[1][:,2],'green',linewidth=2.0, label="Robot Motion")
        axs4[2, 0].legend(loc="upper right")

        axs4[2, 1].plot(sample, output_vel[0][:,5],'red',linewidth=2.0, label="Network Output")
        axs4[2, 1].plot(sample, output_vel[1][:,5],'green',linewidth=2.0, label="Robot Motion")
        axs4[2, 1].legend(loc="upper right")

        axs4[0, 0].title.set_text('Linear Velocity X-Axis')
        axs4[0, 1].title.set_text('Angular Velocity X-Axis')
        axs4[1, 0].title.set_text('Linear Velocity Y-Axis')
        axs4[1, 1].title.set_text('Angular Velocity Y-Axis')
        axs4[2, 0].title.set_text('Linear Velocity Z-Axis')
        axs4[2, 1].title.set_text('Angular Velocity Z-Axis')

        for n,ax in enumerate(axs4.flat):
            if (n % 2 == 0):
                ax.set(ylabel='Velocity (m/s)', xlabel='Sample')
            else:
                ax.set(ylabel='Angular Velocity (rad/s)', xlabel='Sample')
        fig4.suptitle('End Effector Twist', fontsize=20)

        sample = np.linspace(0,len(output_vel[0]),len(output_vel[0]))
        fig5, axs5 = plt.subplots(6, 3)

        axs5[0,0].set_title("Robot  Motion")
        axs5[0,1].set_title("Human Motion")
        axs5[0,2].set_title("Network Output")

        axs5[0, 0].plot(sample, self.robot_sample_scaled[:,0],linewidth=2.0)
        axs5[0, 1].plot(sample, self.human_sample_scaled[:,0],linewidth=2.0)
        axs5[0, 2].plot(sample, self.gen_output_sample[0,:],linewidth=2.0)

        axs5[1, 0].plot(sample, self.robot_sample_scaled[:,1],linewidth=2.0)
        axs5[1, 1].plot(sample, self.human_sample_scaled[:,1],linewidth=2.0)
        axs5[1, 2].plot(sample, self.gen_output_sample[1,:],linewidth=2.0)

        axs5[2, 0].plot(sample, self.robot_sample_scaled[:,2],linewidth=2.0)
        axs5[2, 1].plot(sample, self.human_sample_scaled[:,2],linewidth=2.0)
        axs5[2, 2].plot(sample, self.gen_output_sample[2,:],linewidth=2.0)

        axs5[3, 0].plot(sample, self.robot_sample_scaled[:,3],linewidth=2.0)
        axs5[3, 1].plot(sample, self.human_sample_scaled[:,3],linewidth=2.0)
        axs5[3, 2].plot(sample, self.gen_output_sample[3,:],linewidth=2.0)

        axs5[4, 0].plot(sample, self.robot_sample_scaled[:,4],linewidth=2.0)
        axs5[4, 1].plot(sample, self.human_sample_scaled[:,4],linewidth=2.0)
        axs5[4, 2].plot(sample, self.gen_output_sample[4,:],linewidth=2.0)

        axs5[5, 0].plot(sample, self.robot_sample_scaled[:,5],linewidth=2.0)
        axs5[5, 1].plot(sample, self.human_sample_scaled[:,5],linewidth=2.0)
        axs5[5, 2].plot(sample, self.gen_output_sample[5,:],linewidth=2.0)

        for n,ax in enumerate(axs5.flat):

            if n in [1,4,7]:
                ax.set(ylabel='A (m/s^2)', xlabel='Sample')
            elif n < 9:
                ax.set(ylabel='V (m/s)', xlabel='Sample')
            else:
                ax.set(ylabel='AV (rad/s)', xlabel='Sample')

        # fig4.suptitle('End Effector Twist', fontsize=20)

        if self.cst["save_sim"]:
            # fig1.savefig(label + ' ' + self.cst["simulation_label"] + ' End Effector Twist Error.png')
            fig2.savefig(label + ' ' + self.cst["simulation_label"] + ' Joints Tracking Velocities.png')
            fig3.savefig(label + ' ' + self.cst["simulation_label"] + ' Joints Tracking Position.png')
            fig4.savefig(label + ' ' + self.cst["simulation_label"] + ' End Effector Twist.png')
            fig5.savefig(label + ' ' + self.cst["simulation_label"] + ' Movement Visualization.png')
    
    def generate_poses(self, neutral=False):
        # random.seed(100)
        # random.randint(0,self.robot_data.size()[0])
        robot_traj_sample = self.robot_data[772].unsqueeze(0)

        names = set(self.dataset_tags["emo"].values())
        names_act = set(self.dataset_tags["act"].values())
        print(set(self.dataset_tags["act"].values()))
        print(names)
        # use a list comprehension, iterating through keys and checking the values match each n
        d = {}
        d_act = {}
        for n in names:
            d[n] = [k for k in self.dataset_tags["emo"].keys() if self.dataset_tags["emo"][k] == n]

        for n in names_act:    
            d_act[n] = [k for k in self.dataset_tags["act"].keys() if self.dataset_tags["act"][k] == n]
        
        idx_list = list(set(d[self.emotion_tag])&set(d_act[self.participant_tag]))
        k = np.random.choice(idx_list,1)[0]
        print("Index of interest: ",k)

        #[0,0,50,50,100,100, 700, 700]
        if neutral:
            self.human_data = torch.tensor(self.human_data)
        human_traj_sample = self.human_data[k].unsqueeze(0)
        self.robot_sample_scaled = robot_traj_sample.clone().cpu().squeeze(0).numpy()
        self.human_sample_scaled = human_traj_sample.clone().cpu().squeeze(0).numpy()
        print(f"Human Dataset index: {random.randint(0,self.human_data.size()[0])}")
        print(np.shape(self.human_sample_scaled))
        print(np.shape(self.robot_sample_scaled))

        seq_robot = robot_traj_sample.to(self.device)
        seq_robot = seq_robot.permute(0,2,1)
        
        seq_human = human_traj_sample.to(self.device)
        seq_human = seq_human.permute(0,2,1)

        if not neutral:
            out_hat_twist, z_human, z_robot, mu, log_var, \
                z_human_hat, z_robot_hat, mu_hat,log_var_hat, \
                    cont_att, loc_att_human, loc_att_robot = self.translation_model(seq_robot.float(), seq_human.float())
        else:
            data_neutral_style = np.asarray([self.neutral_style[act] for act in self.dataset_tags["act"].values()])
            data_neutral_style = torch.tensor(data_neutral_style).to(self.device)
            data_neutral_style = data_neutral_style[k].unsqueeze(0)
            out_hat_twist, z_human, z_robot, mu, log_var, \
                z_human_hat, z_robot_hat, mu_hat,log_var_hat, \
                    cont_att, loc_att_human, loc_att_robot = self.translation_model(seq_robot.float(), seq_human.float(),
                                                                                    data_neutral_style)

        self.traj_plan_poses_gan = out_hat_twist.detach().clone().cpu().numpy()
        self.gen_output_sample =  np.squeeze(out_hat_twist.detach().clone().cpu().numpy(),axis=0)
        print("Filter input shape: ",np.shape(self.gen_output_sample))
        self.gen_output_sample = utilities.filter_signal_2(self.gen_output_sample)
        for i in range(np.shape(self.traj_plan_poses_gan)[1]):
            self.traj_plan_poses_gan[:,i,:] = self.scalers[i].inverse_transform(self.traj_plan_poses_gan[:,i,:])
        self.traj_plan_poses_gan = np.squeeze(self.traj_plan_poses_gan,axis=0)
        self.traj_plan_poses_gan = utilities.filter_signal_2(self.traj_plan_poses_gan)

        self.traj_plan_poses_real = seq_robot.detach().clone().cpu().numpy()
        for i in range(np.shape(self.traj_plan_poses_real)[1]):
            self.traj_plan_poses_real[:,i,:] = self.scalers[i].inverse_transform(self.traj_plan_poses_real[:,i,:])
        self.traj_plan_poses_real = np.squeeze(self.traj_plan_poses_real,axis=0)

        sample = np.linspace(0,len( self.traj_plan_poses_real[0,:]),len( self.traj_plan_poses_real[0,:]))
        fig5, axs5 = plt.subplots(6, 3)

        seq_human = seq_human.clone().cpu().squeeze(0).numpy()
        axs5[0, 0].plot(sample, self.traj_plan_poses_gan[0,:],linewidth=2.0)
        axs5[0, 1].plot(sample, self.traj_plan_poses_real[0,:],linewidth=2.0)
        axs5[0, 2].plot(sample, seq_human[0,:],linewidth=2.0)

        axs5[1, 0].plot(sample, self.traj_plan_poses_gan[1,:],linewidth=2.0)
        axs5[1, 1].plot(sample, self.traj_plan_poses_real[1,:],linewidth=2.0)
        axs5[1, 2].plot(sample, seq_human[1,:],linewidth=2.0)

        axs5[2, 0].plot(sample, self.traj_plan_poses_gan[2,:],linewidth=2.0)
        axs5[2, 1].plot(sample, self.traj_plan_poses_real[2,:],linewidth=2.0)
        axs5[2, 2].plot(sample, seq_human[2,:],linewidth=2.0)

        axs5[3, 0].plot(sample, self.traj_plan_poses_gan[3,:],linewidth=2.0)
        axs5[3, 1].plot(sample, self.traj_plan_poses_real[3,:],linewidth=2.0)
        axs5[3, 2].plot(sample, seq_human[3,:],linewidth=2.0)

        axs5[4, 0].plot(sample, self.traj_plan_poses_gan[4,:],linewidth=2.0)
        axs5[4, 1].plot(sample, self.traj_plan_poses_real[4,:],linewidth=2.0)
        axs5[4, 2].plot(sample, seq_human[4,:],linewidth=2.0)

        axs5[5, 0].plot(sample, self.traj_plan_poses_gan[5,:],linewidth=2.0)
        axs5[5, 1].plot(sample, self.traj_plan_poses_real[5,:],linewidth=2.0)
        axs5[5, 2].plot(sample, seq_human[5,:],linewidth=2.0)

        plt.show()

        return str(k)
    
    def derivate_tensor(self,input_tensor, dx=0.016):
        output_arr = []
        x = np.linspace(0, 1, np.shape(input_tensor)[0])
        for cord in range(np.shape(input_tensor)[1]):
            diff_arr = np.diff(input_tensor[:,cord],n=1)/np.diff(x)
            diff_arr = np.insert(diff_arr,-1,diff_arr[-1])
            output_arr.append(diff_arr)
        return np.asarray(output_arr).transpose(1,0)
    
    def integrate_tensor(self,input_tensor, dx=0.017):
        output_arr = []
        x = np.linspace(0, 1, np.shape(input_tensor)[0])
        for cord in range(np.shape(input_tensor)[1]):
            integration_arr = integrate.cumulative_trapezoid(input_tensor[:,cord],x)
            integration_arr = signal.detrend(integration_arr)
            integration_arr = np.insert(integration_arr,-1,integration_arr[-1])
            output_arr.append(integration_arr)
        return np.asarray(output_arr).transpose(1,0)
     
    def env_add_traj_map(self):
        for n,i in enumerate(self.traj_plan_poses):
            target = sg.Sphere(radius=0.01, pose=i)
            self.env.add(target)

    def calculate_path_simulation(self, robot, output_measurements, index):
        # The pose of the Panda's end-effector
        Te = robot.fkine(robot.q)
        output_measurements["output_pose"] = np.vstack((output_measurements["output_pose"], np.array(Te)[:3,3]))
        Rot = np.array(Te)[:3,:3]
        output_measurements["output_rotation"] = np.vstack((output_measurements["output_rotation"], tr.tr2eul(Rot)))
        
        # The manipulator Jacobian in the end-effecotr frame
        Je = robot.jacobe(robot.q)
        v_current = Je @ output_measurements["output_qd"][index]

        # Calulate the required end-effector spatial velocity for the robot
        # to approach the goal. Gain is set to 1.0
        # v, _, error = rp.p_servo(Te, Tep, self.task_constants['gain'])
        # v = tr.tr2delta(np.array(Te), np.array(Tep))
        # v_desired[2:] = v[2:]
        v = output_measurements["trajectory"][:,index]
        output_measurements["error"] = np.vstack((output_measurements["error"],v))

        # Gain term (lambda) for control minimisation
        Y = 1

        # Solve for the joint velocities dq
        qd = np.linalg.pinv(Je) @ (Y * (v))

        # if not neutral:
        #     q = self.panda.q[:] + 0.016*0.03* qd
        # else:
        #     q = self.panda.q[:] + 0.016*0.1* qd
        q = robot.q[:] + 0.016*0.03* qd
        # q = qd + q
        output_measurements["output_vel"] = np.vstack((output_measurements["output_vel"],np.array(v)))

        # Apply the joint velocities to the Panda
        # self.panda.qd[:] = qd[:]
        robot.q[:] = q[:]
        output_measurements["output_q"] = np.vstack((output_measurements["output_q"],np.array(q)))
        output_measurements["output_qd"] = np.vstack((output_measurements["output_qd"],np.array(qd))) 

        return output_measurements
    
    def set_trajectories_dicts(self, trajectory):

        output_vel= np.array([], dtype=np.float64).reshape(0,6)
        output_qd=np.array([], dtype=np.float64).reshape(0,7)
        output_q=np.array([], dtype=np.float64).reshape(0,7)
        output_pose = np.array([], dtype=np.float64).reshape(0,3)
        output_roation = np.array([], dtype=np.float64).reshape(0,3)
        error = np.array([], dtype=np.float64).reshape(0,6)

        output_q = np.vstack((output_q,np.zeros((7))))
        output_qd = np.vstack((output_qd,np.ones((7))))

        trajectory_dict = {"output_vel":output_vel, "output_qd":output_qd,
                            "output_q":output_q,"output_pose":output_pose,
                            "output_rotation":output_roation, "error":error,
                            "trajectory":trajectory}

        return trajectory_dict

    def control_loop(self, real=False, no_env=False):
        
        self.set_environment(no_env=no_env)
        # time.sleep(30)
        time.sleep(3)

        traj_plan_poses_orign_dict = self.set_trajectories_dicts(self.traj_plan_poses_real)
        traj_plan_poses_gan_dict = self.set_trajectories_dicts(self.traj_plan_poses_gan)

        for t in range(np.shape(self.traj_plan_poses_real)[1]):

            traj_plan_poses_orign_dict = self.calculate_path_simulation(self.panda_orign, 
                                                                        traj_plan_poses_orign_dict, t)
            traj_plan_poses_gan_dict = self.calculate_path_simulation(self.panda_network_out, 
                                                                        traj_plan_poses_gan_dict, t)
            
            if not no_env:
                # Step the simulator by 50 ms
                self.env.step(0.016)
                time.sleep(0.016)
        
        if not no_env:
            time.sleep(3)
            self.env.close()
        
        return traj_plan_poses_orign_dict, traj_plan_poses_gan_dict

    def save_data(self):
        pass