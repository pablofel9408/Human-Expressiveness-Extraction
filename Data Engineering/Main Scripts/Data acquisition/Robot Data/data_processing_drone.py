from operator import index
import sys
import os
from types import new_class

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import utils

class DroneDataEngineering():
    def __init__(self, constants_filepath, trajectory='all') -> None:
        self.constants = utils.load_raw_constants(constants_filepath)
        self.traj = trajectory
        self.cols_oi = ['x.1','x.2','y.1','y.2','z.1','z.2']
    
    def load_data(self):
        if self.traj == 'all':
            self.traj_data = {}
            for key,value in self.constants['filepaths'].items():
                self.traj_data[key] = pd.read_csv(os .path.join(value,"blackbird_slash_imu.csv"))
        else:
            self.traj_data = pd.read_csv(os .path.join(self.constants['filepaths'][self.traj],"blackbird_slash_imu.csv"))

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

        total_traj = pd.DataFrame(utils.filter_signal(total_traj.to_numpy(),cutoff_freq=2))
        new_value = pd.DataFrame(utils.filter_signal(new_value.to_numpy(),cutoff_freq=2))
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
            cond = np.delete(cond[0],np.where(cond[0]>10),axis=0)
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
        utils.save_json(self.constants['dataset_stats_savepath'],output_dict, name='drone_dataset_stats.json')

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

        fig.suptitle('Drone Twist')
       
    def plot_traj(self,key_name=None, index_oi=0):
        if key_name is None:
            key_name = 'mouse'

        if isinstance(self.traj_data,tuple):
            for n, data in enumerate(self.traj_data):
                sample = np.linspace(0,len(data),len(data))
                fig, axs = plt.subplots(3, 2)
                for a,ax in enumerate(axs.flatten()):
                    ax.plot(sample,data.to_numpy()[:,a])
                self.set_axis(fig,axs)

        elif isinstance(self.traj_data,dict):
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
            axs[0, 0].plot(sample,data['x.1'])
            axs[0, 1].plot(sample, data['x.2'])
            axs[1, 0].plot(sample, data['y.1'])
            axs[1, 1].plot(sample, data['y.2'])
            axs[2, 0].plot(sample, data['z.1'])
            axs[2, 1].plot(sample, data['z.2'])
            self.set_axis(fig,axs)

        plt.show()

def main():

    process = 0
    if len(sys.argv) > 2:
        process = int(sys.argv[2])
        task_name = sys.argv[1]

    elif len(sys.argv) > 1:
        task_name = sys.argv[1]
    else:
        task_name = 'half_moon'

    json_filepath = r"C:\Users\posorio\Documents\Expressive movement\Data Engineering\Main Scripts\Data acquisition\Robot Data\constants files\drone_constants.json"
    drone_obj = DroneDataEngineering(json_filepath, trajectory=task_name)

    if process < 1:
        drone_obj.run_initial_assesment()
    elif process == 1:
        drone_obj.run_after_processing()
    elif process == 2:
        drone_obj.run_process_plot()
    elif process == 3:
        drone_obj.run_data_stats()
    else:
        drone_obj.run_data_processing()

if __name__=='__main__':
    main()