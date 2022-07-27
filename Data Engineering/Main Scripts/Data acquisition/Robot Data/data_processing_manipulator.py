from operator import index
import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import utils

class ManipulatorDataEngineering():
    def __init__(self, constants_filepath, trajectory='all') -> None:
        self.constants = utils.load_raw_constants(constants_filepath)
        self.traj = trajectory
        self.mult_final_len = utils.multiple(20,self.constants["n_total"])

    def load_recursive_np(self, filepath):
        trajectories = []
        filepath = utils.find("*.npy",filepath)
        for file in filepath:
            with open(file, 'rb') as f:                 
                trajectories.append(np.load(f))
        return trajectories
    
    def load_data(self):
        if self.traj == 'all':
            self.traj_data = {}
            for key,value in self.constants['filepaths'].items():
                self.traj_data[key] = self.load_recursive_np(value)
        else:
            self.traj_data = self.load_recursive_np(self.constants['filepaths'][self.traj])

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

        new_traj = np.empty((0,self.constants["n_total"],6))
        print(np.shape(value))
        for i,traj in enumerate(value):
            aux = []
            # traj -= np.mean(traj,axis=0)
            # traj = utils.filter_signal(traj,cutoff_freq=2)
            shape_traj = np.shape(traj)
            len_mult = min(self.mult_final_len, key=lambda x:abs(x-shape_traj[0]))

            for n in range(shape_traj[1]):
                # traj_resample = signal.resample(traj[:len_mult,n], len_mult)
                traj_resample = self.resample_by_interpolation(traj[:len_mult,n], len(traj[:len_mult,n]),len_mult)
                if len(traj[:,n])>self.constants['n_total']:
                    if (shape_traj[0] - len_mult) > 10:
                        # trajN_resample = signal.resample(traj[len_mult:,n], self.constants['n_total'])
                        trajN_resample = self.resample_by_interpolation(traj[:len_mult,n],len(traj[:len_mult,n]), self.constants['n_total'])
                        traj_resample = np.hstack((traj_resample,trajN_resample))
                    traj_resample = np.reshape(traj_resample,(int(len(traj_resample)/self.constants["n_total"]),self.constants["n_total"]))
                    aux.append(traj_resample)
                else:
                    # print(np.shape(traj))
                    aux.append(self.resample_by_interpolation(traj[:,n], len(traj[:,n]), self.constants['n_total']))

            traj = np.asanyarray(aux)
            if len(np.shape(traj)) <= 2:
                traj = np.expand_dims(traj, axis=0)
                
            if np.shape(traj)[0] == 6:
                traj = np.transpose(traj,axes=[1,2,0])
            else:
                traj = np.transpose(traj,axes=[0,2,1])

            new_traj = np.append(new_traj,traj,axis=0)
        print('Old size: ',np.shape(new_traj))
        a1,a2 = np.unique(np.where(abs(np.mean(new_traj,axis=1))<5e-6),return_counts=True)
        aa = []
        for key,value in dict(zip(a1,a2)).items():
            if value > 1:
                aa.append(key)
        new_traj = np.delete(new_traj,aa,axis=0)
        print('New size: ',np.shape(new_traj))
        new_traj = utils.filter_signal(new_traj.transpose(0,2,1),cutoff_freq=3, fs=60).transpose(0,2,1)
        return new_traj

    def pre_process_data(self):
        self.raw_data = self.traj_data
        if self.traj == 'all':
            for key,value in self.traj_data.items():
                self.traj_data[key] = self.process_sampling(value)
        else:
            self.traj_data = self.process_sampling(self.traj_data) 

    def get_dataset_stats(self):
        ys = np.array([])
        for arr in self.traj_data.values():
            ys = np.vstack([ys, arr]) if ys.size else arr
        output_arr = np.asarray(ys)
        output_dict = utils.get_cross_correlation(output_arr)
        utils.save_json(self.constants['dataset_stats_savepath'],output_dict, name='manipulator_dataset_stats.json')

    def save_dataset(self):
        if self.traj == 'all':
            for key,value in self.traj_data.items():
                utils.save_numpy(self.constants['dataset_savepath'], key, value)
        else:
            utils.save_numpy(self.constants['dataset_savepath'], self.traj, self.traj_data)

    def run_data_processing(self):
        self.load_data()
        self.pre_process_data()
        self.save_dataset()

    def run_get_stats(self):
        self.load_data()
        self.pre_process_data()
        self.get_dataset_stats()

    def run_initial_assesment(self):
        self.load_data()
        self.plot_traj()

    def run_process_plot(self):
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

        fig.suptitle('End Effector Twist')
       
    def plot_traj(self,key_name=None, index_oi=24):
        if key_name is None:
            key_name = 'circle'

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

        elif isinstance(self.traj_data,np.ndarray) or isinstance(self.traj_data,list):
            for data in self.traj_data:
                # data = self.traj_data[i]
                sample = np.linspace(0,len(data),len(data))
                fig, axs = plt.subplots(3, 2)
                axs[0, 0].plot(sample,data[:,0])
                axs[0, 1].plot(sample, data[:,3])
                axs[1, 0].plot(sample, data[:,1])
                axs[1, 1].plot(sample, data[:,4])
                axs[2, 0].plot(sample, data[:,2])
                axs[2, 1].plot(sample, data[:,5])

                self.set_axis(fig,axs)

                plt.show()

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
        task_name = 'circle'

    json_filepath = r"C:\Users\posorio\Documents\Expressive movement\Data Engineering\Main Scripts\Data acquisition\Robot Data\constants files\manipulator_constants.json"
    man_obj = ManipulatorDataEngineering(json_filepath, trajectory=task_name)
    if process < 1:
        man_obj.run_initial_assesment()
    elif process == 1:
        man_obj.run_process_plot()
    elif process == 2:
        man_obj.run_get_stats()
    else:
        man_obj.run_data_processing()


if __name__=='__main__':
    main()