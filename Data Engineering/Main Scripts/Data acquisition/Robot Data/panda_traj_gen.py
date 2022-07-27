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

import utils

#TODO: rewrite traj generation

class ManTrajGent():
    def __init__(self, task, sample_id, rotation_motion) -> None:

        self.task = task
        self.sample_id = sample_id
        self.rotation_motion = rotation_motion[0]
        self.rotation_motion_val = rotation_motion[1]
        
        if self.task != "load":
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

    def draw_circle(self):
        Te = self.panda.fkine(self.panda.q)
        # self.t = np.linspace(0.0, self.task_constants['task_time'], int(1/self.task_constants['dt']))
        self.cricle_center = sm.SE3(np.column_stack((np.array(Te)[:,:3],np.array([0.25,0.25,self.task_constants['z'],1]))))
        self.x = self.task_constants['r']*np.cos(2*math.pi*self.task_constants['f']*self.t)
        self.y = self.task_constants['r']*np.sin(2*math.pi*self.task_constants['f']*self.t)
        print(len(self.x))
        self.z = [self.task_constants['z']]*len(self.y)
        self.x_vel = np.append(np.diff(self.x,axis=0),0)
        self.y_vel = np.append(np.diff(self.y,axis=0),0)
        plt.plot(self.t,self.x_vel)
        plt.plot(self.t,self.y_vel)
        plt.show() 

        plt.plot(self.t,self.x)
        plt.plot(self.t,self.y)
        plt.show()

    def draw_infinity(self):
        Te = self.panda.fkine(self.panda.q)
        self.cricle_center = sm.SE3(np.column_stack((np.array(Te)[:,:3],np.array([0.25,0.25,self.task_constants['z'],1]))))
        self.x = self.task_constants['r']*np.cos(2*math.pi*self.task_constants['f']*self.t)
        print(len(self.t))
        print(len(self.x))
        self.y = self.task_constants['r']*np.sin(2*2*math.pi*self.task_constants['f']*self.t)
        self.z = [self.task_constants['z']]*len(self.y)

    def draw_zig_zag(self):
        self.x = self.t * 0.3 + self.task_constants['b']
        self.y =  [0.3]*len(self.x) #self.t * 0.3 + self.task_constants['b']
        self.z = (4*self.task_constants['a']/self.task_constants['p']) * (np.abs(((self.x-self.task_constants['p']/4) % 
                                                                                self.task_constants['p'])-self.task_constants['p']/2)) + 2*self.task_constants['a'] 

    def new_traj(self):
        self.draw_line_yz()
        Te = self.panda.fkine(self.panda.q)
        R = np.array(Te)

    def compute_traj(self,t=50):
        self.return_traj  = rp.ctraj(self.pose_end,self.pose_init,t=t)
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

        if self.task == 'new_traj':
            self.pose_init = sm.SE3(np.column_stack((R[:,:3],np.array([self.x[0],self.y[0],self.z[0],1]))))
            self.pose_end = Te
            self.compute_traj()
            [self.traj_plan_poses.insert(0,n_pose) for n_pose in self.return_traj[::-1]]

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
        self.output_pose = np.array([], dtype=np.float64).reshape(0,3)
        self.error = np.array([], dtype=np.float64).reshape(0,6)
        for pose in self.traj_plan_poses:

            arrived = False

            # Set the desired end-effector pose
            Tep = pose # panda.fkine(panda.q) * 
            while not arrived:

                # The pose of the Panda's end-effector
                Te = self.panda.fkine(self.panda.q)
                self.output_pose = np.vstack((self.output_pose,np.array(Te)[:3,3]))

                # The manipulator Jacobian in the end-effecotr frame
                Je = self.panda.jacobe(self.panda.q)

                # Calulate the required end-effector spatial velocity for the robot
                # to approach the goal. Gain is set to 1.0
                v, arrived, error = rp.p_servo(Te, Tep, self.task_constants['gain'], threshold=0.001)
                self.error = np.vstack((self.error,error))

                # self.output_vel = np.vstack((self.output_vel,np.array(v)))

                # Gain term (lambda) for control minimisation
                Y = 0.001

                # The manipulability Jacobian
                Jm = self.panda.jacobm(self.panda.q)

                # # Project the gradient of manipulability into the null-space of the
                # # differential kinematics
                null = (
                    np.eye(n) - np.linalg.pinv(Je) @ Je
                ) @ Jm

                # Solve for the joint velocities dq
                qd = np.linalg.pinv(Je) @ v + 1 / Y * null.flatten()
                # qd = Y * np.linalg.pinv(Je) @ v
                self.output_qd = np.vstack((self.output_qd,np.array(qd)))

                # Apply the joint velocities to the Panda
                self.panda.qd[:] = qd[:]

                # Step the simulator by 50 ms
                self.env.step(0.001)
                self.output_vel = np.vstack((self.output_vel,np.array(v)))

        print(len(self.output_vel))
        self.plot_twist()
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

    def plot_twist(self):
        sample = np.linspace(0,len(self.output_pose)*0.001,len(self.output_pose))
        sample1 = np.linspace(0,1,len(self.traj_plan_poses))

        figa, (ax1,ax2) = plt.subplots(2)
        print(np.shape(self.output_pose))
        aux = self.output_pose[self.output_pose[:,0] < 0.38,0]
        ax1.plot(sample[self.output_pose[:,0] < 0.38],aux)
        ax2.plot(sample1,np.array(self.traj_plan_poses)[:,0,3])
        figa.suptitle('End Effector Task Space Position')

        samp = np.linspace(0,len(self.error),len(self.error))
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].plot(sample, self.error[:,0])
        axs[0, 1].plot(sample, self.error[:,1])
        axs[1, 0].plot(sample, self.error[:,2])
        axs[1, 1].plot(sample, self.error[:,3])
        axs[2, 0].plot(sample, self.error[:,4])
        axs[2, 1].plot(sample, self.error[:,5])
        fig.suptitle('End Effector Twist Error')

        sample = np.linspace(0,len(self.output_qd),len(self.output_qd))
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

        sample = np.linspace(0,len(self.output_vel),len(self.output_vel))
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
                ax.set(ylabel='Velocity m/s', xlabel='Sample')
            else:
                ax.set(ylabel='Angular Velocity rad/s', xlabel='Sample')


        fig.suptitle('End Effector Twist')

    def load_data(self, filepath, plot=False):

        for file in filepath:
            with open(file, 'rb') as f:                 
                self.output_vel = np.load(f)
                if plot:
                    self.plot_twist()
        plt.show()


def main():

    rotation_vel = ('no_rotation',[0.0,0.0,0.0])
    if len(sys.argv) > 3:
        task_name = sys.argv[1]
        sample_id = sys.argv[2]
        rotation_vel = (sys.argv[3], np.array(list(map(float,sys.argv[4].split(',')))))

    elif len(sys.argv) > 1:
        task_name = sys.argv[1]
        sample_id = sys.argv[2]
    else:
        task_name = 'circle'

    file_oi = [r'C:\Users\posorio\Documents\Expressive movement\Data Engineering\Datasets\Robot\Panda (manipulator)\Twist\twist_line2DYZ_no_rotation_[0.0, 0.0, 0.0]_0.npy',
                r'C:\Users\posorio\Documents\Expressive movement\Data Engineering\Datasets\Robot\Panda (manipulator)\Twist\twist_line2DYZ_no_rotation_[0.0, 0.0, 0.0]_1.npy']
    
    traj_gen_obj = ManTrajGent(task_name,sample_id,rotation_vel)
    if task_name == 'load':
        traj_gen_obj.load_data(file_oi,plot=True)
    else:
        traj_gen_obj.plot_traj3D()
        traj_gen_obj.run_control_traj()



if __name__ == '__main__':
    main()