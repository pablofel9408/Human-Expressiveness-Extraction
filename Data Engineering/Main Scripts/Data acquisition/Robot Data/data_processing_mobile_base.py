from ctypes import util
from operator import index
import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
from torch import NoneType

import utils

"""
Additional datasets
-----------------------------------------------------------------------------------------------------------------------
"office": "C:\\Users\\posorio\\Documents\\Expressive movement\\Data Engineering\\Datasets\\Robot\\Mobile base\\Office",
"cafe": "C:\\Users\\posorio\\Documents\\Expressive movement\\Data Engineering\\Datasets\\Robot\\Mobile base\\Cafe",
"market": "C:\\Users\\posorio\\Documents\\Expressive movement\\Data Engineering\\Datasets\\Robot\\Mobile base",
"home": "C:\\Users\\posorio\\Documents\\Expressive movement\\Data Engineering\\Datasets\\Robot\\Mobile base",
----------------------------------------------------------------------------------------------------------------------
"""

class MobileBaseDataEngineering():
    def __init__(self, constants_filepath, trajectory='all') -> None:
        self.constants = utils.load_raw_constants(constants_filepath)
        self.traj = trajectory
        self.cols_oi = ["v_x", "v_y", "v_z", "v_ang_x", "v_ang_y", "v_ang_z"]
        self.new_cols_names = ["Time","v_x", "v_y", "v_z", "v_ang_x", "v_ang_y", "v_ang_z"]
        self.raw_cols_labels = ["Time","x", "y", "z", "theta_x", "theta_y", "theta_z", 
                                "w", "v_x", "v_y", "v_z", "v_ang_x", "v_ang_y", "v_ang_z"]
    
    def load_data(self):
        if self.traj == 'all':
            self.traj_data = {}
            for key,value in self.constants['filepaths'].items():
                data_files = utils.find(self.constants["base_name"]+"*",value)
                self.traj_data[key] = self.handle_mult_files(data_files)
        else:
            data_files = utils.find(self.constants["base_name"]+"*",self.constants['filepaths'][self.traj])
            self.traj_data = self.handle_mult_files(data_files)

    def handle_mult_files(self, data_files):
        if len(data_files)>1:
            aux_traj_data = pd.DataFrame()
            for n,filename in enumerate(data_files):
                print(n)
                if ".txt" in filename:
                    aux = pd.read_csv(filename, sep=" ",header=None)
                    aux = self.process_data(aux)
                    aux_traj_data = pd.concat([aux_traj_data,aux],ignore_index=True, axis=0)
                else:
                    aux = pd.read_csv(filename, header=None)  
                    aux = self.process_data(aux)

                    if (n%2)!=0:
                        time = aux[[0]]
                        aux_imu = aux_imu[[4,5,6,7,8,9]]
                        aux = aux[[1,2,3]]
                        aux_imu = aux_imu.iloc[:min(aux.shape[0],aux_imu.shape[0])]
                        aux = aux.iloc[:min(aux.shape[0],aux_imu.shape[0])]
                        user_acc = self.compensate_imu_readings(aux_imu,aux)
                        print(np.shape(user_acc))
                        aux_imu[[4,5,6]] = user_acc[:,1:]
                        aux_imu[0] = time
                        aux_imu = aux_imu[[0,4,5,6,7,8,9]]
                        aux_traj_data = pd.concat([aux_traj_data,aux_imu],ignore_index=True, axis=0)
                    else:
                        aux = aux[[0,4,5,6,7,8,9]]
                        aux_imu = aux.copy()
        else:
            aux_traj_data = pd.read_csv(os.path.join(self.constants['filepaths'][self.traj],"odom.txt"), sep=" ",header=None)
            aux_traj_data = self.process_data(aux_traj_data)

        print(aux_traj_data.columns)
        if aux_traj_data.shape[1] > 10:
            aux_traj_data.columns = self.raw_cols_labels
        else:
            aux_traj_data.columns = self.new_cols_names

        if aux_traj_data.isnull().values.sum()>1:
            print(aux_traj_data.isnull().values.sum())
            print("Corrupted data, check it!!!!!!!!!!!")
            sys.exit()
        else:
            aux_traj_data.fillna(0)
        
        return aux_traj_data

    def compensate_imu_readings(self, acc, euler):
        acc = acc.to_numpy()
        euler = euler.to_numpy()
        q = utils.euler_to_quaternion(euler)
        return utils.gravity_compensation(acc[:,:3],q)

    def process_data(self, inputDf):
        inputDf.drop(index = inputDf.index[0], axis=0, inplace=True)
        inputDf = inputDf.apply(pd.to_numeric)
        inputDf.reset_index()
        return inputDf

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

    def process_sampling(self,value):
        value = value[self.cols_oi]
        new_value = pd.DataFrame()
        total_traj = pd.DataFrame()
        for col in self.cols_oi:
            value[col] -= value[col].mean()
            total_traj[col] = self.resample_by_interpolation(value[col],len(value[col]),self.constants['n_total']) 
            new_value[col] = self.resample_by_interpolation(value[col],len(value[col]),self.constants['n'])
            # total_traj[col] = signal.resample(value[col], self.constants['n_total']) 
            # new_value[col] = signal.resample(value[col], self.constants['n'])
        
        total_traj = pd.DataFrame(utils.filter_signal(total_traj.to_numpy(),cutoff_freq=3))
        new_value = pd.DataFrame(utils.filter_signal(new_value.to_numpy(),cutoff_freq=3))
        return (new_value, total_traj)

    def pre_process_data(self):
        self.raw_data = self.traj_data
        if self.traj == 'all':
            for key,value in self.traj_data.items():
                self.traj_data[key] = self.process_sampling(value)
        else:
            self.traj_data = self.process_sampling(self.traj_data) 

    def reshape_data(self,value):
        aux = []
        prev_cond = []
        for col in range(np.shape(value[0])[1]):
            new_sig = np.reshape(value[0].to_numpy()[:,col],(np.round(self.constants["n"]/self.constants["n_total"]).astype(int),
                                                                self.constants["n_total"]))
            cond = np.where(np.ptp(new_sig,axis=1)<0.1)
            print(cond)
            cond = np.delete(cond[0],np.where(cond[0]>10),axis=0)
            print(cond)
            if len(cond) < len(prev_cond):
                cond = prev_cond
            else:
                 prev_cond = cond
            if np.sum(cond) > 0:
                new_sig = np.delete(new_sig,cond,axis=0)
            aux.append(new_sig)
        array_len = min([len(i) for i in aux])
        aux = [arr[len(arr)-array_len:,:] for arr in aux]
        aux = np.transpose(np.asarray(aux),(1,2,0))
        return np.vstack((aux,np.expand_dims(value[1],axis=0)))

    def post_process_data(self):
        if self.traj == 'all':
            for key,value in self.traj_data.items():
                self.traj_data[key] = self.reshape_data(value)
        else:
            self.traj_data = self.reshape_data(self.traj_data)
            print(np.shape(self.traj_data))

    def save_dataset(self):
        if self.traj == 'all':
            for key,value in self.traj_data.items():
                utils.save_numpy(self.constants['dataset_savepath'], key, value)
        else:
            utils.save_numpy(self.constants['dataset_savepath'], self.traj, value)

    def get_dataset_stats(self):
        ys = np.array([])
        for arr in self.traj_data.values():
            ys = np.vstack([ys, arr]) if ys.size else arr
        output_arr = np.asarray(ys)
        output_dict = utils.get_cross_correlation(output_arr)
        utils.save_json(self.constants['dataset_stats_savepath'],output_dict, name='mobile_base_dataset_stats.json')

    def run_data_stats(self):
        self.load_data()
        self.pre_process_data()
        self.post_process_data()
        self.get_dataset_stats()

    def run_data_processing(self):
        self.load_data()
        self.pre_process_data()
        self.post_process_data()
        self.save_dataset()

    def run_initial_assesment(self):
        self.load_data()
        self.plot_traj()

    def run_process_plot(self):
        self.load_data()
        self.pre_process_data()
        self.post_process_data()
        self.plot_traj()
    
    def run_after_processing(self):
        self.load_data()
        self.pre_process_data()
        self.plot_traj()
        
    def set_axis(self,fig,axs):
        axs[0, 0].title.set_text('Angular Velocity X-Axis')
        axs[0, 1].title.set_text('Linear Velocity X-Axis')
        axs[1, 0].title.set_text('Angular Velocity Y-Axis')
        axs[1, 1].title.set_text('Linear Velocity Y-Axis')
        axs[2, 0].title.set_text('Angular Velocity Z-Axis')
        axs[2, 1].title.set_text('Linear Velocity Z-Axis')

        for n,ax in enumerate(axs.flat):
            if (n % 2 == 0):
                ax.set(ylabel='Velocity m/s', xlabel='Sample')
            else:
                ax.set(ylabel='Angular Velocity rad/s', xlabel='Sample')

        fig.suptitle('Mobile Base Twist')

    def set_axis_v2(self,fig,axs):
        axs[0, 0].title.set_text('Liner Velocity X-Axis')
        axs[0, 1].title.set_text('Linear Velocity Y-Axis')
        axs[1, 0].title.set_text('Linear Velocity Y-Axis')
        axs[1, 1].title.set_text('Angular Velocity Y-Axis')
        axs[2, 0].title.set_text('Angular Velocity Z-Axis')
        axs[2, 1].title.set_text('Angular Velocity Z-Axis')

        for n,ax in enumerate(axs.flat):
            if (n % 2 == 0):
                ax.set(ylabel='Velocity m/s', xlabel='Sample')
            else:
                ax.set(ylabel='Angular Velocity rad/s', xlabel='Sample')

        fig.suptitle('Mobile Base Twist')
       
    def plot_traj(self,key_name=None, index_oi=30):
        if key_name is None:
            key_name = 'sequence1'

        if isinstance(self.traj_data,tuple):
            for n, data in enumerate(self.traj_data):
                sample = np.linspace(0,len(data),len(data))
                fig, axs = plt.subplots(3, 2)
                for a,ax in enumerate(axs.flatten()):
                    ax.plot(sample,data.to_numpy()[:,a])
                self.set_axis_v2(fig,axs)

        elif isinstance(self.traj_data,dict):
            print(np.shape(self.traj_data[key_name]))
            if isinstance(self.traj_data[key_name],pd.DataFrame):
                data = self.traj_data[key_name].to_numpy()
            else:
                data = self.traj_data[key_name][index_oi]
            
            sample = np.linspace(0,len(data),len(data))
            fig, axs = plt.subplots(3, 2)
            axs[0, 0].plot(sample,data[:,0])
            axs[0, 1].plot(sample, data[:,3])
            axs[1, 0].plot(sample, data[:,1])
            axs[1, 1].plot(sample, data[:,4])
            axs[2, 0].plot(sample, data[:,2])
            axs[2, 1].plot(sample, data[:,5])

            self.set_axis(fig,axs)

        elif isinstance(self.traj_data,np.ndarray):
            data = self.traj_data[index_oi]
            sample = np.linspace(0,len(data),len(data))
            fig, axs = plt.subplots(3, 2)
            axs[0, 0].plot(sample,data[:,0])
            axs[0, 1].plot(sample, data[:,3])
            axs[1, 0].plot(sample, data[:,1])
            axs[1, 1].plot(sample, data[:,4])
            axs[2, 0].plot(sample, data[:,2])
            axs[2, 1].plot(sample, data[:,5])

            self.set_axis(fig,axs)

        elif isinstance(self.traj_data,pd.DataFrame):
            data = self.traj_data
            sample = np.linspace(0,len(data),len(data))
            fig, axs = plt.subplots(3, 2)
            axs[0, 0].plot(sample,data['v_x'])
            axs[0, 1].plot(sample, data['v_ang_x'])
            axs[1, 0].plot(sample, data['v_y'])
            axs[1, 1].plot(sample, data['v_ang_y'])
            axs[2, 0].plot(sample, data['v_z'])
            axs[2, 1].plot(sample, data['v_ang_z'])
    
            axs[0,0].set_ylim([data['v_x'].min(), data['v_x'].max()])
            axs[0,1].set_ylim([data['v_ang_x'].min(), data['v_ang_x'].max()])
            axs[1,0].set_ylim([data['v_y'].min(), data['v_y'].max()])
            axs[1,1].set_ylim([data['v_ang_y'].min(), data['v_ang_y'].max()])
            axs[2,0].set_ylim([data['v_z'].min(), data['v_z'].max()])
            axs[2,1].set_ylim([data['v_ang_z'].min(), data['v_ang_z'].max()])
            self.set_axis(fig,axs)

            # M = 5
            # yticks = matplotlib.ticker.MaxNLocator(M)
            # for i in axs.flatten():
            #     i.yaxis.set_major_locator(yticks)

        plt.show()

def main():

    process = 0
    if len(sys.argv) > 2:
        process = int(sys.argv[2])
        task_name = sys.argv[1]

    elif len(sys.argv) > 1:
        task_name = sys.argv[1]
    else:
        task_name = 'sequence1'

    json_filepath = r"C:\Users\posorio\Documents\Expressive movement\Data Engineering\Main Scripts\Data acquisition\Robot Data\constants files\mobile_base_constants.json"
    mb_obj = MobileBaseDataEngineering(json_filepath, trajectory=task_name)

    if process < 1:
        mb_obj.run_initial_assesment()
    elif process == 1:
        mb_obj.run_after_processing()
    elif process == 2:
        mb_obj.run_process_plot()
    elif process == 3:
        mb_obj.run_data_stats()
    else:
        mb_obj.run_data_processing()

if __name__=='__main__':
    main()