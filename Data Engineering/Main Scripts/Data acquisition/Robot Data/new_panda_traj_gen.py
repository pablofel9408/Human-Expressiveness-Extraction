import os
import sys
import math 
import json 
import copy 

import numpy as np
import numpy.ma as ma
import pandas
import pathlib
import matplotlib.pyplot as plt

import swift
import roboticstoolbox as rp
import spatialgeometry as sg
import spatialmath as sm
import spatialmath.base as tr

import scipy.signal as sig
import scipy.interpolate as interpolate
import utils

#TODO: rewrite traj generation

class ManTrajGent():
    def __init__(self, task, sample_id, rotation_motion) -> None:

        self.task = task
        self.sample_id = sample_id
        self.rotation_motion = rotation_motion[0]
        self.rotation_motion_val = rotation_motion[1]
        
        if (self.task != "load") and (self.sample_id != 'auto'):
            self.env = swift.Swift()
            self.env.launch(realtime=True)

            # Create a Panda robot object
            self.panda = rp.models.Panda()

            # Set joint angles to ready configuration
            self.panda.q = self.panda.qr

            # Add the Panda to the simulator
            self.env.add(self.panda)

            self.dir_path = pathlib.Path(__file__).parent.resolve()
            self.filepath = utils.find("*.json",self.dir_path)
            
            self.task_constants = utils.load_constants(self.filepath[0],self.task)
            self.t = np.linspace(0.0, self.task_constants['task_time'], int(self.task_constants['interp_time']/self.task_constants['dt']))

            if self.task != 'automatic':
                self.choose_task()

    def run_automatic_genration(self):
        
        freq = [100,200,300]
        new_samp_num = [100,200,300]
        r = [0.05,0.1,0.2]
        self.dir_path = pathlib.Path(__file__).parent.resolve()
        self.filepath = utils.find("*.json",self.dir_path)
        self.task_constants = utils.load_constants(self.filepath[0],self.task)
        count = 0
        for samp in new_samp_num:
            self.task_constants['new_samp_num'] = samp
            for rad in r:
                self.sample_id = str(count)
                self.task_constants['a'] = rad
                self.env = swift.Swift()
                self.env.launch(realtime=True)

                # Create a Panda robot object
                self.panda = rp.models.Panda()

                # Set joint angles to ready configuration
                self.panda.q = self.panda.qr

                # Add the Panda to the simulator
                self.env.add(self.panda)
                self.t = np.linspace(0.0, self.task_constants['task_time'], int(self.task_constants['interp_time']/self.task_constants['dt']))
                self.choose_task()
                self.run_control_traj()
                self.env.close()
                count+=1


    
    def choose_task(self):
        tasks_dict = {
            'circle': self.draw_circle,
            'infinity': self.draw_infinity,
            'line2DXY': self.draw_line_xy,
            'line2DYZ': self.draw_line_yz,
            'line2DY': self.draw_line_y,
            'line2DZ': self.draw_line_z,
            'line2DX': self.draw_line_x,
            'zigzag': self.draw_zig_zag,
            'new_traj': self.new_traj
        }
        print(self.task)
        tasks_dict.get(self.task, lambda: 'Invalid')()
        self.transform_poses()

    def draw_line_xy(self):
        self.y = self.t * 0.3
        self.x = self.task_constants['m']*self.y + self.task_constants['b']
        self.z = [0.2]*len(self.y)

    def draw_line_yz(self):
        self.y = self.t * 0.3
        self.z = self.task_constants['m']*self.y + self.task_constants['b']
        self.x = [self.task_constants['x']]*len(self.y)

    def draw_line_y(self):
        self.y = self.t + self.task_constants['b']
        self.x = [self.task_constants['x']]*len(self.y)
        self.z = [self.task_constants['z']]*len(self.y)
    
    def draw_line_x(self):
        self.x = self.t + self.task_constants['b']
        self.y = [self.task_constants['y']]*len(self.x)
        self.z = [self.task_constants['z']]*len(self.x)

    def draw_line_z(self):
        self.z = self.t *0.3 + self.task_constants['b']
        self.y = [0.4]*len(self.z)
        self.x = [0.3]*len(self.z)

    def resample_by_interpolation(self,signal, input_fs, output_fs):

        scale = output_fs / input_fs
        # calculate new length of sample
        n = round(len(signal) * scale)

        # use linear interpolation
        # endpoint keyword means than linspace doesn't go all the way to 1.0
        # If it did, there are some off-by-one errors
        # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
        # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
        # Both are OK, but since resampling will often involve
        # exact ratios (i.e. for 44100 to 22050 or vice versa)
        # using endpoint=False gets less noise in the resampled sound
        resampled_signal = np.interp(
            np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
            np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
            signal,  # known data points
        )
        return resampled_signal

    def draw_circle(self):

        self.t = np.linspace(0.0, self.task_constants['task_time'], int(self.task_constants['interp_time']/self.task_constants['dt']))
        Te = self.panda.fkine(self.panda.q)
        self.cricle_center = sm.SE3(np.column_stack((np.array(Te)[:,:3],np.array([0.25,0.25,self.task_constants['z'],1]))))
        # self.cricle_center = sm.SE3(np.column_stack((np.array(Te)[:,:3],np.array([0.25,0.25,0.6,1]))))

        self.x = self.task_constants['r']*np.cos(2*math.pi*self.task_constants['f']*self.t)
        self.y = self.task_constants['r']*np.sin(2*math.pi*self.task_constants['f']*self.t)

        # self.x_vel = -2*math.pi*self.task_constants['f']*self.task_constants['r']*np.sin(2*math.pi*self.task_constants['f']*self.t)
        # self.y_vel = 2*math.pi*self.task_constants['f']*self.task_constants['r']*np.cos(2*math.pi*self.task_constants['f']*self.t)

        # plt.plot(self.t,self.x)
        # plt.plot(self.t,self.y)
        # plt.show()
        
        if self.task_constants['resample']:
            self.t = np.linspace(0,self.task_constants['new_time'],self.task_constants['new_samp_num'])
            self.x = self.resample_by_interpolation(self.x,self.task_constants['interp_time']/self.task_constants['dt'],self.task_constants['new_samp_num'])
            self.y = self.resample_by_interpolation(self.y,self.task_constants['interp_time']/self.task_constants['dt'],self.task_constants['new_samp_num'])

        self.z = [self.task_constants['z']]*len(self.y)

        # plt.plot(self.t,self.x)
        # plt.plot(self.t,self.y)
        # plt.show()

    def draw_infinity(self):
        Te = self.panda.fkine(self.panda.q)
        self.cricle_center = sm.SE3(np.column_stack((np.array(Te)[:,:3],np.array([0.25,0.25,self.task_constants['z'],1]))))
        self.x = self.task_constants['r']*np.cos(2*math.pi*self.task_constants['f']*self.t)
        print(len(self.t))
        print(len(self.x))
        self.y = self.task_constants['r']*np.sin(2*2*math.pi*self.task_constants['f']*self.t)

        if self.task_constants['resample']:
            self.t = np.linspace(0,self.task_constants['new_time'],self.task_constants['new_samp_num'])
            self.x = self.resample_by_interpolation(self.x,self.task_constants['interp_time']/self.task_constants['dt'],self.task_constants['new_samp_num'])
            self.y = self.resample_by_interpolation(self.y,self.task_constants['interp_time']/self.task_constants['dt'],self.task_constants['new_samp_num'])

        self.z = [self.task_constants['z']]*len(self.y)

    def draw_zig_zag(self):
        self.x = self.t * 0.3 + self.task_constants['b']
        self.z = (4*self.task_constants['a']/self.task_constants['p']) * (np.abs(((self.x-self.task_constants['p']/4) % 
                                                                                self.task_constants['p'])-self.task_constants['p']/2)) + 2*self.task_constants['a'] 

        if self.task_constants['resample']:
            self.t = np.linspace(0,self.task_constants['new_time'],self.task_constants['new_samp_num'])
            self.x = self.resample_by_interpolation(self.x,self.task_constants['interp_time']/self.task_constants['dt'],self.task_constants['new_samp_num'])
            self.z = self.resample_by_interpolation(self.z,self.task_constants['interp_time']/self.task_constants['dt'],self.task_constants['new_samp_num'])

        self.y =  [0.3]*len(self.x) #self.t * 0.3 + self.task_constants['b']

    def new_traj(self):
        if self.task_constants['resample']:
            self.t = np.linspace(0,self.task_constants['new_time'],self.task_constants['new_samp_num'])

        self.draw_line_yz()
        Te = self.panda.fkine(self.panda.q)
        R = np.array(Te)

    def compute_traj(self,t=100):
        self.return_traj  = rp.ctraj(self.pose_end,self.pose_init,t=len(self.t))
        self.return_traj = [i for i in self.return_traj]

    def transform_poses(self):
        self.traj_plan_poses = []
        Te = self.panda.fkine(self.panda.q)
        R = np.array(Te)
        angles = tr.tr2rpy(np.array(Te),check=False)

        for i in zip(self.x,self.y,self.z):

            # if self.rotation_motion is not None:
            #     R = tr.rpy2tr(angles + self.rotation_motion_val)
            #     angles = tr.tr2rpy(np.array(R),check=False)

            if self.task == 'circle' or self.task == 'infinity':
                pose = sm.SE3(i[0],i[1],0)
                self.cricle_center = np.array(self.cricle_center)
                self.cricle_center[:,:3] = R[:,:3]
                self.cricle_center = sm.SE3(self.cricle_center)
                self.traj_plan_poses.append(self.cricle_center @ pose)
            else:
                pose = sm.SE3(np.column_stack((R[:,:3],np.array([i[0],i[1],i[2],1]))))
                self.traj_plan_poses.append(pose)

        self.pose_init = self.traj_plan_poses[0]
        self.pose_end = Te
        self.compute_traj()
        [self.traj_plan_poses.insert(0,n_pose) for n_pose in self.return_traj[::-1]]

        if self.task == 'new_traj':

            self.pose_init = sm.SE3(np.column_stack((R[:,:3],np.array([self.x[0],self.y[0],self.z[0],1]))))
            self.pose_end = sm.SE3(np.column_stack((R[:,:3],np.array([self.x[-1],self.y[-1],self.z[-1],1]))))
            self.compute_traj()
            [self.traj_plan_poses.append(n_pose) for n_pose in self.return_traj]

            self.pose_end = sm.SE3(np.column_stack((R[:,:3],np.array([self.x[0],self.y[0],self.z[0],1]))))
            self.draw_line_xy()
            self.pose_init = sm.SE3(np.column_stack((R[:,:3],np.array([self.x[0],self.y[0],self.z[0],1]))))
            self.compute_traj()
            [self.traj_plan_poses.append(n_pose) for n_pose in self.return_traj]
            for i in zip(self.x,self.y,self.z):
                pose = sm.SE3(np.column_stack((R[:,:3],np.array([i[0],i[1],i[2],1]))))
                self.traj_plan_poses.append(pose)

            self.pose_end = sm.SE3(np.column_stack((R[:,:3],np.array([self.x[-1],self.y[-1],self.z[-1],1]))))          
            self.draw_line_z()
            self.pose_init = sm.SE3(np.column_stack((R[:,:3],np.array([self.x[0],self.y[0],self.z[0],1]))))
            self.compute_traj()
            [self.traj_plan_poses.append(n_pose) for n_pose in self.return_traj]
            for i in zip(self.x,self.y,self.z):
                pose = sm.SE3(np.column_stack((R[:,:3],np.array([i[0],i[1],i[2],1]))))
                self.traj_plan_poses.append(pose)

            self.pose_end = sm.SE3(np.column_stack((R[:,:3],np.array([self.x[-1],self.y[-1],self.z[-1],1]))))          
            self.draw_circle()
            pose = sm.SE3(self.x[0],self.y[0],0)
            self.cricle_center = np.array(self.cricle_center)
            self.cricle_center[:,:3] = R[:,:3]
            self.cricle_center = sm.SE3(self.cricle_center)
            self.pose_init = self.cricle_center @ pose
            self.compute_traj()
            [self.traj_plan_poses.append(n_pose) for n_pose in self.return_traj]
            for i in zip(self.x,self.y,self.z):
                if self.rotation_motion is not None:
                    R = tr.rpy2tr(angles + self.rotation_motion_val)
                    angles = tr.tr2rpy(np.array(R),check=False)

                pose = sm.SE3(i[0],i[1],0)
                self.cricle_center = np.array(self.cricle_center)
                self.cricle_center[:,:3] = R[:,:3]
                self.cricle_center = sm.SE3(self.cricle_center)
                self.traj_plan_poses.append(self.cricle_center @ pose)

            self.pose_end = copy.deepcopy(self.traj_plan_poses[-1])
            self.compute_traj(t=100)
            [self.traj_plan_poses.append(n_pose) for n_pose in self.return_traj]

            self.pose_end = copy.deepcopy(self.traj_plan_poses[-1])
            self.pose_init = Te
            self.compute_traj()
            [self.traj_plan_poses.append(n_pose) for n_pose in self.return_traj]

        
    def plot_traj3D(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter (self.x,self.y,self.z)
        plt.show()

    def env_add_traj_map(self):
        for n,i in enumerate(self.traj_plan_poses):
            target = sg.Sphere(radius=0.01, pose=i)
            self.env.add(target)

    def run_control_traj(self):

        self.env_add_traj_map()

        # # Number of joint in the panda which we are controlling
        n = 7
        self.output_vel= np.array([], dtype=np.float64).reshape(0,6)
        self.output_qd=np.array([], dtype=np.float64).reshape(0,7)
        self.output_q=np.array([], dtype=np.float64).reshape(0,7)
        self.output_pose = np.array([], dtype=np.float64).reshape(0,3)
        self.error = np.array([], dtype=np.float64).reshape(0,6)

        count=0
        for t,pose in enumerate(self.traj_plan_poses):
            # Set the desired end-effector pose
            Tep = pose # panda.fkine(panda.q) * 
            # v_desired = np.array([self.x_vel[t],self.y_vel[t],0,0,0,0])


            # The pose of the Panda's end-effector
            Te = self.panda.fkine(self.panda.q)
            self.output_pose = np.vstack((self.output_pose,np.array(Te)[:3,3]))

            # The manipulator Jacobian in the end-effecotr frame
            Je = self.panda.jacobe(self.panda.q)

            # Calulate the required end-effector spatial velocity for the robot
            # to approach the goal. Gain is set to 1.0
            # v, _, error = rp.p_servo(Te, Tep, self.task_constants['gain'])
            v = tr.tr2delta(np.array(Te), np.array(Tep))
            # v_desired[2:] = v[2:]
            self.error = np.vstack((self.error,v))

            # Gain term (lambda) for control minimisation
            Y = 1

            # Solve for the joint velocities dq
            qd = np.linalg.pinv(Je) @ (Y * (v))
            q = self.panda.q[:] + 0.01*60*qd
            self.output_vel = np.vstack((self.output_vel,np.array(v)))

            # Apply the joint velocities to the Panda
            # self.panda.qd[:] = qd[:]
            self.panda.q[:] = q[:]
            self.output_q = np.vstack((self.output_q,np.array(q)))
            self.output_qd = np.vstack((self.output_qd,np.array(qd)))

            # Step the simulator by 50 ms
            self.env.step(0.001)


        print(len(self.output_vel))
        self.plot_twist(plot_all=False)
        plt.show()

        validate_flag = utils.query_yes_no("Save trajectory", default="no")
        if validate_flag:
            self.save_trajecotry()

    def save_trajecotry(self):
        with open(os.path.join(self.task_constants['dataset_path_twist'], 'twist_' +  self.task + '_' + self.rotation_motion + '_' +
                                str(self.rotation_motion_val)  + '_' + self.sample_id + '.npy'), 'wb') as f:                 
            np.save(f, self.output_vel)

        with open(os.path.join(self.task_constants['dataset_path_joint'], 'joint_vel_' + self.task + '_' + self.rotation_motion + '_' +
                                str(self.rotation_motion_val)  + '_' + self.sample_id + '.npy'), 'wb') as f:                 
            np.save(f, self.output_qd)

        with open(os.path.join(self.task_constants['dataset_path_json'], 'constants' + self.task + '_' + self.rotation_motion + '_' +
                                str(self.rotation_motion_val) + '_' + self.sample_id +'.json'), 'w') as json_file:
            json.dump(self.task_constants,json_file)

    def plot_twist(self, plot_all=True):
        # sample = np.linspace(0,len(self.output_pose)*0.01,len(self.output_pose))
        # sample1 = np.linspace(0,len(self.output_pose)*0.01,len(self.traj_plan_poses))

        # figa, axs = plt.subplots(3,3)
        # print(np.shape(self.output_pose))
        # aux = self.output_pose[self.output_pose[:,0] < 0.38,0]
        # axs[0, 0].plot(sample,self.output_pose[:,0])
        # axs[1, 0].plot(sample1,np.array(self.traj_plan_poses)[:,0,3])
        # axs[2, 0].plot(sample,self.output_pose[:,0]-np.array(self.traj_plan_poses)[:,0,3])

        # axs[0, 1].plot(sample,self.output_pose[:,1])
        # axs[1, 1].plot(sample1,np.array(self.traj_plan_poses)[:,1,3])
        # axs[2, 1].plot(sample,self.output_pose[:,1]-np.array(self.traj_plan_poses)[:,1,3])

        # axs[0, 2].plot(sample,self.output_pose[:,2])
        # axs[1, 2].plot(sample1,np.array(self.traj_plan_poses)[:,2,3])
        # axs[2, 2].plot(sample,self.output_pose[:,2]-np.array(self.traj_plan_poses)[:,2,3])

        # axs[0, 0].title.set_text('Control Output End Effector Position - X')
        # axs[1, 0].title.set_text('Reference End Effector Position - X')
        # axs[2, 0].title.set_text('Differnece Between Position - X')

        # axs[0, 1].title.set_text('Control Output End Effector Position - Y')
        # axs[1, 1].title.set_text('Reference End Effector Position - Y')
        # axs[2, 1].title.set_text('Differnece Between Position - Y')

        # axs[0, 2].title.set_text('Control Output End Effector Position - Z')
        # axs[1, 2].title.set_text('Reference End Effector Position - Z')
        # axs[2, 2].title.set_text('Differnece Between Position - Z')
        
        # for n,ax in enumerate(axs.flat):
        #     ax.set(ylabel='Position (m)', xlabel='Time (s)')
        # figa.suptitle('Task Space Position Tracking - Reference vs Output', fontsize=20)

        if plot_all:
            samp = np.linspace(0,len(self.error),len(self.error))
            fig, axs = plt.subplots(3, 2)
            axs[0, 1].plot(sample, self.error[:,1])
            axs[1, 0].plot(sample, self.error[:,2])
            axs[1, 1].plot(sample, self.error[:,3])
            axs[2, 0].plot(sample, self.error[:,4])
            axs[2, 1].plot(sample, self.error[:,5])
            fig.suptitle('End Effector Twist Error')

            sample = np.linspace(0,len(self.output_pose)*0.01,len(self.output_qd))
            fig, axs = plt.subplots(4, 2)
            axs[0, 0].plot(sample, self.output_qd[:,0])
            axs[0, 1].plot(sample, self.output_qd[:,1])
            axs[1, 0].plot(sample, self.output_qd[:,2])
            axs[1, 1].plot(sample, self.output_qd[:,3])
            axs[2, 0].plot(sample, self.output_qd[:,4])
            axs[2, 1].plot(sample, self.output_qd[:,5])
            axs[3, 0].plot(sample, self.output_qd[:,6])

            axs[0, 0].title.set_text('Joint 1 Velocities')
            axs[0, 1].title.set_text('Joint 2 Velocities')
            axs[1, 0].title.set_text('Joint 3 Velocities')
            axs[1, 1].title.set_text('Joint 4 Velocities')
            axs[2, 0].title.set_text('Joint 5 Velocities')
            axs[2, 1].title.set_text('Joint 6 Velocities')
            axs[3, 0].title.set_text('Joint 7 Velocities')

            for n,ax in enumerate(axs.flat):
                ax.set(ylabel='Angular velocity (rad/s)', xlabel='Time (s)')
            fig.suptitle('Joints Tracking Velocities', fontsize=20)

            sample = np.linspace(0,len(self.output_pose)*0.01,len(self.output_q))
            fig, axs = plt.subplots(4, 2)
            axs[0, 0].plot(sample, self.output_q[:,0])
            axs[0, 1].plot(sample, self.output_q[:,1])
            axs[1, 0].plot(sample, self.output_q[:,2])
            axs[1, 1].plot(sample, self.output_q[:,3])
            axs[2, 0].plot(sample, self.output_q[:,4])
            axs[2, 1].plot(sample, self.output_q[:,5])
            axs[3, 0].plot(sample, self.output_q[:,6])

            axs[0, 0].title.set_text('Joint 1 Position')
            axs[0, 1].title.set_text('Joint 2 Position')
            axs[1, 0].title.set_text('Joint 3 Position')
            axs[1, 1].title.set_text('Joint 4 Position')
            axs[2, 0].title.set_text('Joint 5 Position')
            axs[2, 1].title.set_text('Joint 6 Position')
            axs[3, 0].title.set_text('Joint 7 Position')

            for n,ax in enumerate(axs.flat):
                ax.set(ylabel='Position (rad)', xlabel='Time (s)')
            fig.suptitle('Joints Tracking Position', fontsize=20)

        sample = np.linspace(0,len(self.output_vel)*0.01,len(self.output_vel))
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].plot(sample, self.output_vel[:,0])
        axs[0, 1].plot(sample, self.output_vel[:,3])
        axs[1, 0].plot(sample, self.output_vel[:,1])
        axs[1, 1].plot(sample, self.output_vel[:,4])
        axs[2, 0].plot(sample, self.output_vel[:,2])
        axs[2, 1].plot(sample, self.output_vel[:,5])

        axs[0, 0].title.set_text('Linear Velocity X-Axis')
        axs[0, 1].title.set_text('Angular Velocity X-Axis')
        axs[1, 0].title.set_text('Linear Velocity Y-Axis')
        axs[1, 1].title.set_text('Angular Velocity Y-Axis')
        axs[2, 0].title.set_text('Linear Velocity Z-Axis')
        axs[2, 1].title.set_text('Angular Velocity Z-Axis')

        for n,ax in enumerate(axs.flat):
            if (n % 2 == 0):
                ax.set(ylabel='Velocity (m/s)', xlabel='Time (s)')
            else:
                ax.set(ylabel='Angular Velocity (rad/s)', xlabel='Time (s)')

        fig.suptitle('End Effector Twist', fontsize=20)

    def load_data(self, filepath, plot=False):

        for file in filepath:
            with open(file, 'rb') as f:                 
                self.output_vel = np.load(f)
                if plot:
                    self.plot_twist(plot_all=False)
        plt.show()


def main():

    rotation_vel = ('no_rotation',[0.0,0.0,0.0])
    sample_id = 0
    if len(sys.argv) > 3:
        task_name = sys.argv[1]
        sample_id = sys.argv[2]
        rotation_vel = (sys.argv[3], np.array(list(map(float,sys.argv[4].split(',')))))

    elif len(sys.argv) > 1:
        task_name = sys.argv[1]
        sample_id = sys.argv[2]
    else:
        task_name = 'circle'

    file_oi = [r'C:\Users\posorio\Documents\Expressive movement\Data Engineering\Datasets\Robot\Panda (manipulator)\Twist\Infinity\twist_infinity_no_rotation_[0.0, 0.0, 0.0]_0.npy',
                r'C:\Users\posorio\Documents\Expressive movement\Data Engineering\Datasets\Robot\Panda (manipulator)\Twist\\Infinity\twist_infinity_no_rotation_[0.0, 0.0, 0.0]_3.npy']
    
    traj_gen_obj = ManTrajGent(task_name,sample_id,rotation_vel)
    if task_name == 'load':
        traj_gen_obj.load_data(file_oi,plot=True)
    else:
        # traj_gen_obj.plot_traj3D()
        if sample_id == 'auto':
            traj_gen_obj.run_automatic_genration()
        else:
            traj_gen_obj.run_control_traj()



if __name__ == '__main__':
    main()