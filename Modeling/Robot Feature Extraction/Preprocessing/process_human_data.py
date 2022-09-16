import os 
import copy
import pickle

import random
import numpy as np
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.model_selection import train_test_split

import torch

from . import augmentation
import utilities

class PreprocessHumanData():
    def __init__(self, constants) -> None:

        self.cst=constants
        self.datasets={}
        self.norm_datasets={}
        self.scalers={}

        self.output_dict={i:None for i in ['train','val']} #test
        self.aug_labels=[]

    def load_recursive_pt(self, filepath):
        
        keyword = "_body_signals"
        if self.cst["normalization"]:
            keyword = "_scaled_body_signals"

        filepath = utilities.find("*.pt",filepath)
        for file in filepath:
            if "test" + keyword  in file:
                print("test" + keyword)
                aux = self.retrieve_data(file)
                aux1 = aux[:,:,6:9]
                aux2 = aux[:,:,-3:]
                self.output_dict["test"] = np.concatenate((aux1,aux2),axis=2)
            elif "train" + keyword in file:
                print("train" + keyword)
                aux = self.retrieve_data(file)
                aux1 = aux[:,:,6:9]
                aux2 = aux[:,:,-3:]
                self.output_dict["train"] = np.concatenate((aux1,aux2),axis=2)
            elif "val" + keyword in file:
                print("val" + keyword)
                aux = self.retrieve_data(file)
                aux1 = aux[:,:,6:9]
                aux2 = aux[:,:,-3:]
                self.output_dict["val"] = np.concatenate((aux1,aux2),axis=2)
        
        self.plot_data()

        dirpath = self.cst["scalers_path"]
        # if not os.path.exists(dirpath):
        #     os.makedirs(dirpath)
        # utilities.gen_archive(dirpath)

        scaler =  TimeSeriesScalerMinMax(value_range=(0,1))
        scaler.fit(self.output_dict['train'])
        for key,value in self.output_dict.items():
            self.output_dict[key]=scaler.transform(value)
        
        scaler_filename = os.path.join(dirpath, "scaler_dataset_human" + ".pickle")
        with open(scaler_filename, "wb") as output_file:
            pickle.dump(value, output_file)

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
        value = value.numpy()
        total_traj = np.zeros((len(value),self.cst["n"],np.shape(value)[2]))
        plt.plot(value[0])
        for n,ele in enumerate(value):
            for col in range(np.shape(ele)[1]):
                value[n,:,col] -= value[n,:,col].mean()
                total_traj[n,:,col] = self.resample_by_interpolation(ele[:,col],len(ele[:,col]),self.cst['n'])

        # total_traj = utilities.filter_signal(total_traj.to_numpy(),cutoff_freq=2)
        # plt.plot(total_traj[0])
        # plt.show()
        return total_traj

    def reshape_data(self,value):
        
        final_arr = np.array([], dtype=np.int64).reshape(0,60,15)
        for samp in range(np.shape(value)[0]):
            aux = []
            prev_cond = []
            for col in range(np.shape(value)[2]):
                new_sig = np.reshape(value[samp,:,col],(np.round(self.cst["n"]/self.cst["n_total"]).astype(int),
                                                                    self.cst["n_total"]))
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
            final_arr = np.vstack([final_arr,aux])
    
        return final_arr

    def plot_data(self):
        print(np.shape(self.output_dict["train"]))
        plt.plot(self.output_dict["train"][0])
        plt.show()

    def retrieve_data(self, file):
        value = torch.load(file)
        value = self.reshape_data(self.process_sampling(value))

        return value

    def return_data_(self):
        return self.output_dict

      