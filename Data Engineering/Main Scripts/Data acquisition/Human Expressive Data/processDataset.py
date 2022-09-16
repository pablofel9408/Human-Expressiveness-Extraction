from math import degrees
import sys

import numpy as np
import matplotlib.pyplot as plt
import utilities

from scipy import interpolate
from scipy.spatial.transform import Rotation as R

class preprocessEmotionData():
    def __init__(self, constants) -> None:
        
        self.cst = constants
        self.mocap_dataset = None
        self.max_sign_len = 0
        self.resample_offset = 7

    def load_data(self, mocap_dataset):
        self.mocap_dataset = mocap_dataset
        self.reformat_input()

    def reformat_input(self):
        for val in self.cst['candidate']:
            spec_keys = [val+'_'+mark for mark in self.cst['new_markers']]
            for traj in self.mocap_dataset[val].keys():
                for marker in spec_keys:
                    for key in self.cst['output_signals']:
                        if key == 'pos':
                            pos = self.mocap_dataset[val][traj][marker]
                            self.max_sign_len = len(pos) if len(pos) > self.max_sign_len else self.max_sign_len
                            self.mocap_dataset[val][traj][marker] =  None
                            self.mocap_dataset[val][traj][marker] = {key:pos}
                        else:
                            self.mocap_dataset[val][traj][marker][key] = None

    def resample_data(self, samples, k_spline_degree, end_time, resample_len):
        time = np.linspace(0, end_time, len(samples))
        t, c, k = interpolate.splrep(time, samples, k=k_spline_degree)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        data_time = np.linspace(0, end_time, resample_len)
        output_spline_data = spline(data_time)

        return output_spline_data, data_time
    
    def calc_derivative(self, input_arr, time_interval):

        input_size = np.shape(input_arr)
        output_derv = np.zeros((input_size[0],input_size[1]))
        for axis in range(input_size[1]):
            deriv = np.diff(input_arr[:,axis]) / np.diff(time_interval)
            deriv = np.append(deriv,deriv[-1])
            output_derv[:,axis] = deriv
      
        return output_derv

    def calc_ang_velocity(self,input_rot, last_time):
        input_rot = np.reshape(input_rot,(len(input_rot),3,3))
        euler = R.from_matrix(input_rot).as_euler('XYZ',degrees=False)
        aux_arr = np.zeros((self.max_sign_len+self.resample_offset,3))
        for i in range(0,3):
            aux_arr[:,i], resampled_time = self.resample_data(euler[:,i], 
                                    self.cst['spline_degree'], last_time, 
                                    self.max_sign_len+self.resample_offset)
        
        return self.calc_derivative(aux_arr, resampled_time)
        

    def plot_coordinates_sing_ex(self, candidate_val='EMLA', traj_val='traj_0', marker='EMLA_LW'):

        for key in self.cst['output_signals']:
            if key in ["exp_weight", "exp_laban_time", 
                        "exp_flow", "exp_bounding_vol_box","exp_space"]:
                continue

            if isinstance(self.mocap_dataset[candidate_val][traj_val][marker][key],tuple):
                array_oi = self.mocap_dataset[candidate_val][traj_val][marker][key][0]
                print(self.mocap_dataset[candidate_val][traj_val][marker][key][1])
            else:
                array_oi = self.mocap_dataset[candidate_val][traj_val][marker][key]
            array_oi_size = np.shape(array_oi)
            print(array_oi_size)

            try:
                fig, axs = plt.subplots(1, array_oi_size[1])
            except:
                fig, axs = plt.subplots(1,1)

            map_dict_titles = {'pos':'Position', 'vel':'Velocity', 
                                'acc':'Acceleration', 'ang_vel':'Angular Velocity', 
                                'jerk': 'Jerk', 'exp_weight': 'Laban Weight','exp_laban_time': 'Laban Time',
                                'exp_flow': 'Laban Flow','exp_space': 'Laban Space',
                                'exp_bounding_vol_box': 'Volume Bounding Box'}
            fig.suptitle('Mocap Data ' + map_dict_titles[key])

            map_dict = {0:'X', 1:'Y', 2:'Z'}
            map_dict_y_axis= {'pos':'Position in meters(m)', 'vel':'Velocity in meters per second(m/s)', 
                                'acc':'Acceleration in meters per second squared(m/s^2)', 'jerk': 'Jerk(m/s^3)',
                                'ang_vel':'Angular Velocity(rad/s)', 'exp_weight': 'Laban Weight',
                                'exp_laban_time': 'Laban Time','exp_flow': 'Laban Flow',
                                'exp_space': 'Laban Space', 'exp_bounding_vol_box': 'Volume Bounding Box'}
            time = np.linspace(0,array_oi_size[0],array_oi_size[0])

            try:
                for num, axes in enumerate(axs.flat):
                    axes.plot(time, array_oi[:,num])
                    axes.set_title(map_dict[num] + '- Axis')
                    axes.set(xlabel='Sample number', ylabel=map_dict_y_axis[key])
            except:
                axs.plot(time, array_oi)
                axs.set_title(map_dict_titles[key])
                axs.set(xlabel='Sample number', ylabel=map_dict_y_axis[key])

        plt.show() 

    def start_preprocess(self):

        for val in self.cst['candidate']:
            spec_keys = [val+'_'+mark for mark in self.cst['new_markers']]
            for traj in self.mocap_dataset[val].keys():
                for marker in spec_keys:
                    for key in self.cst['output_signals']:
                        if key == 'pos':
                            
                            aux_arr = np.zeros((self.max_sign_len+self.resample_offset,3))
                            for i in range(0,3):
                                aux_arr[:,i], resampled_time = self.resample_data(self.mocap_dataset[val][traj][marker][key][:,i], 
                                                        self.cst['spline_degree'], self.mocap_dataset[val][traj]['Time'][-1], 
                                                        self.max_sign_len+self.resample_offset)
                            self.mocap_dataset[val][traj]['Time'] = resampled_time
                            self.mocap_dataset[val][traj][marker][key] = aux_arr
                                
                        elif key == 'vel':
                            self.mocap_dataset[val][traj][marker][key] = self.calc_derivative(self.mocap_dataset[val][traj][marker]['pos'],
                                                                                                self.mocap_dataset[val][traj]['Time'])
                            self.mocap_dataset[val][traj][marker][key] = utilities.filter_signal(self.mocap_dataset[val][traj][marker][key], 
                                                                                                cutoff_freq=5, filt_type='lp')
                        elif key == 'acc':
                            self.mocap_dataset[val][traj][marker][key] = self.calc_derivative(self.mocap_dataset[val][traj][marker]['vel'],
                                                                                                self.mocap_dataset[val][traj]['Time'])
                            self.mocap_dataset[val][traj][marker][key] = utilities.filter_signal(self.mocap_dataset[val][traj][marker][key], 
                                                                                                cutoff_freq=5, filt_type='lp')
                                                                                              
                        elif key == 'jerk':
                            self.mocap_dataset[val][traj][marker][key] = self.calc_derivative(self.mocap_dataset[val][traj][marker]['acc'],
                                                                                                self.mocap_dataset[val][traj]['Time'])  

                        elif key == 'ang_vel':
                            if 'LW' in marker:
                                self.mocap_dataset[val][traj][marker][key] = self.calc_ang_velocity(self.mocap_dataset[val][traj]['wrist_rot_LW'][:,1:],
                                                                                self.mocap_dataset[val][traj]['wrist_rot_LW'][-1,0])
                            else:
                                self.mocap_dataset[val][traj][marker][key] = self.calc_ang_velocity(self.mocap_dataset[val][traj]['wrist_rot_RW'][:,1:],
                                                                                self.mocap_dataset[val][traj]['wrist_rot_RW'][-1,0])

    def return_processed_data(self):
        return self.mocap_dataset
                                        