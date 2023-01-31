import os 
import copy
import pickle
import json

import random
from tkinter import constants
import numpy as np
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
        self.output_dict_tags = {}
        self.aug_labels=[]

    def load_recursive_pt(self, filepath):

        self.load_dataset_labels()
        
        keyword = "_body_signals"
        if self.cst["normalization"]:
            keyword = "_scaled_body_signals"

        filepath = utilities.find("*.pt",filepath)
        for file in filepath:
            
            if "test" + keyword  in file:
                print("test" + keyword)

                aux, emotion_tags, actor_tags = self.retrieve_data(file, self.output_dict_tags["test"])
                self.output_dict_tags["test"]["emo"] = emotion_tags
                self.output_dict_tags["test"]["act"] = actor_tags
                aux1 = aux[:,:,6:9]
                aux2 = aux[:,:,-3:]
                if self.cst["expressive_data"]:
                    qualities_test = self.calc_expressive_qualities(aux)
                self.output_dict["test"] = np.concatenate((aux1,aux2),axis=2)
                # plt.plot(self.output_dict["test"][0])
                # plt.show()

            elif "train" + keyword in file:
                print("train" + keyword)
                aux, emotion_tags, actor_tags = self.retrieve_data(file, self.output_dict_tags["train"])
                self.output_dict_tags["train"]["emo"] = emotion_tags
                self.output_dict_tags["train"]["act"] = actor_tags
                aux1 = aux[:,:,6:9]
                aux2 = aux[:,:,-3:]
                if self.cst["expressive_data"]:
                    qualities_train = self.calc_expressive_qualities(aux)
                self.output_dict["train"] = np.concatenate((aux1,aux2),axis=2)

            elif "val" + keyword in file:
                print("val" + keyword)
                aux, emotion_tags, actor_tags = self.retrieve_data(file, self.output_dict_tags["val"])
                self.output_dict_tags["val"]["emo"] = emotion_tags
                self.output_dict_tags["val"]["act"] = actor_tags
                aux1 = aux[:,:,6:9]
                aux2 = aux[:,:,-3:]
                if self.cst["expressive_data"]:
                    qualities_val = self.calc_expressive_qualities(aux)
                self.output_dict["val"] = np.concatenate((aux1,aux2),axis=2)
        
        # self.plot_data()

        dirpath = self.cst["scalers_path"]
        # if not os.path.exists(dirpath):
        #     os.makedirs(dirpath)
        # utilities.gen_archive(dirpath)

        # scalers = {i:MinMaxScaler() for i in range(np.shape(self.output_dict['train'])[2])}
        # for idx, value in enumerate(scalers.values()):
        #     value.fit(self.output_dict['train'][:,:,idx])
        #     # print(np.shape(self.output_dict['train'][0][:,:,idx]))
        #     self.output_dict['train'][:,:,idx] = value.transform(self.output_dict['train'][:,:,idx])
        #     self.output_dict['val'][:,:,idx] = value.transform(self.output_dict['val'][:,:,idx])
        #     self.output_dict['test'][:,:,idx] = value.transform(self.output_dict['test'][:,:,idx])

        # # scaler =  TimeSeriesScalerMinMax(value_range=(0,1))
        # # scaler.fit(self.output_dict['train'])
        # # for key,value in self.output_dict.items():
        # #     self.output_dict[key]=scaler.transform(value)
        
        # scaler_filename = os.path.join(dirpath, "scaler_dataset_human" + ".pickle")
        # with open(scaler_filename, "wb") as output_file:
        #     pickle.dump(scalers, output_file)

        # import sys
        # sys.exit()

        if self.cst["expressive_data"]:
            self.output_dict["val"] = (self.output_dict["val"], qualities_val)
            self.output_dict["test"] = (self.output_dict["test"], qualities_test)
            self.output_dict["train"] = (self.output_dict["train"], qualities_train)

        # if self.cst["expressive_data"]:
        #     self.output_dict["val"] = (new_raw_data, qualities_val)
        #     self.output_dict["test"] = (new_raw_data, qualities_test)
        #     self.output_dict["train"] = (new_raw_data, qualities_train)

    def load_dataset_labels(self):
        f = open(os.path.join(self.cst["dataset_paths"],'output_dict_tags.json'),)
        self.output_dict_tags = json.load(f)

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
        # plt.plot(value[0])
        for n,ele in enumerate(value):
            for col in range(np.shape(ele)[1]):
                # value[n,:,col] -= value[n,:,col].mean()
                total_traj[n,:,col] = self.resample_by_interpolation(ele[:,col],len(ele[:,col]),self.cst['n'])

        # total_traj = utilities.filter_signal(total_traj.to_numpy(),cutoff_freq=2)
        # plt.plot(total_traj[0])
        # plt.show()
        return total_traj

    def reshape_data(self,value, label_dict):
        
        final_arr = np.array([], dtype=np.int64).reshape(0,60,15)
        emotion_tags = []
        actor_tags = []
        for samp in range(np.shape(value)[0]):
            aux = []
            prev_cond = []
            tag_oi = label_dict["emo"][samp]
            act_oi = label_dict["act"][samp]
            for col in range(np.shape(value)[2]):
                new_sig = np.reshape(value[samp,:,col],(np.round(self.cst["n"]/self.cst["n_total"]).astype(int),
                                                                    self.cst["n_total"]))
                # cond = np.where(np.ptp(new_sig,axis=1)<0.1)
                # cond = np.delete(cond[0],np.where(cond[0]>10),axis=0)
                # if len(cond) < len(prev_cond):
                #     cond = prev_cond
                # else:
                #     prev_cond = cond
                # if np.sum(cond) > 0:
                #     new_sig = np.delete(new_sig,cond,axis=0)
                aux.append(new_sig)
            array_len = min([len(i) for i in aux])
            aux = [arr[len(arr)-array_len:,:] for arr in aux]
            aux = np.transpose(np.asarray(aux),(1,2,0))

            emotion_tags.append([tag_oi]*len(aux))
            actor_tags.append([act_oi]*len(aux))
            final_arr = np.vstack([final_arr,aux])

        emotion_tags = np.asarray([x for sublist in emotion_tags for x in sublist])
        actor_tags = np.asarray([x for sublist in actor_tags for x in sublist])
        print(np.shape(emotion_tags))
        print(np.shape(final_arr))
        # final_arr = value
        # emotion_tags = label_dict["emo"]
        # actor_tags = label_dict["act"]

        return final_arr, emotion_tags, actor_tags

    def calc_expressive_qualities(self, input_signals):
        weight = self.calc_laban_weight(input_signals[:,:,3:6])
        time = self.calc_laban_time(input_signals[:,:,6:9])
        flow = self.calc_laban_flow(input_signals[:,:,9:12])
        space = self.calc_laban_space(input_signals[:,:,:3])
        bound_vol = self.calc_vol_bound_box(input_signals[:,:,:3])

        output_qualities = np.stack((weight,time,flow,space,bound_vol),axis=1)
        return output_qualities

    def calc_laban_weight(self, input_sign, time_inter=10.0): 
        alpha_w = 1.0
        n = round(np.shape(input_sign)[1]/time_inter)
        weight_tot = []
        weight_sum = []
        for k,val in enumerate(input_sign):
            aux = []
            sub_arr = [val[i:i+n,:] for i in range(0,len(val), n)]
            for arr in sub_arr:
                aux.append(alpha_w * np.linalg.norm(arr)**2)
            weight_tot.append(aux)
            weight_sum.append(sum(aux))

        # weight = torch.cat((weight_tot,torch.unsqueeze(weight_sum,dim=1)),dim=1)
        weight = np.array(weight_sum)
        return weight

    def calc_laban_time(self, input_sign, time_inter=10.0): 
        alpha_w = 1.0
        n = round(np.shape(input_sign)[1]/time_inter)
        laban_time_tot = []
        laban_time_sum = []
        for k,val in enumerate(input_sign):
            aux = []
            sub_arr = [val[i:i+n,:] for i in range(0,len(val), n)]
            for arr in sub_arr:
                aux.append(((alpha_w * np.linalg.norm(arr))/n)**2)
            laban_time_tot.append(aux)
            laban_time_sum.append(sum(aux))
        
        laban_time = np.array(laban_time_sum)
        # laban_time = np.concatenate((laban_time_tot,np.expand_dims(laban_time_sum,axis=1)),axis=1)

        return laban_time

    def calc_laban_flow(self, input_sign, time_inter=10.0): 
        alpha_w = 1.0
        n = round(np.shape(input_sign)[1]/time_inter)
        laban_flow_tot = []
        laban_flow_sum = []
        for k,val in enumerate(input_sign):
            aux = []
            sub_arr = [val[i:i+n,:] for i in range(0,len(val), n)]
            for arr in sub_arr:
                aux.append(((alpha_w * np.linalg.norm(arr))/n)**2)
            laban_flow_tot.append(aux)
            laban_flow_sum.append(sum(aux))
        
        laban_flow = np.array(laban_flow_sum)
        # laban_flow = np.concatenate((laban_flow_tot,np.expand_dims(laban_flow_sum,axis=1)),axis=1)

        return laban_flow

    def calc_laban_space(self, input_sign, time_inter=10.0): 
        alpha_w = 1.0
        n = round(np.shape(input_sign)[1]/time_inter)
        laban_space_tot = []
        laban_space_sum = []
        for k,val in enumerate(input_sign):
            arr = [val[i:i+n,:] for i in range(0,len(val), n)]
            aux = []
            tot_norm = np.linalg.norm(arr[-1]-arr[0]) + 0.000001
            for num in range(1,len(arr)):
                aux.append(((alpha_w * np.linalg.norm(arr[num]-arr[num-1]))/ tot_norm)**2)
            laban_space_tot.append(aux)
            laban_space_sum.append(sum(aux))

        laban_space = np.array(laban_space_sum)
        # laban_space = np.concatenate((laban_space_tot,np.expand_dims(laban_space_sum,axis=1)),axis=1)

        return laban_space

    def calc_vol_bound_box(self, input_sign, time_inter=10.0): 
        n = round(np.shape(input_sign)[1]/time_inter)
        bound_vol_tot = []
        bound_vol_sum = []
        for num,val in enumerate(input_sign):
            aux = []
            aa = [val[i:i+n,:] for i in range(0,len(val), n)]
            for arr in aa:
                aux.append((max(arr[:,0]) * max(arr[:,1]) * max(arr[:,2]))**2)
            bound_vol_tot.append(aux)
            bound_vol_sum.append(sum(aux))

        bound_vol = np.array(bound_vol_sum)
        # bound_vol = np.concatenate((bound_vol_tot,np.expand_dims(bound_vol_sum,axis=1)),axis=1)

        return bound_vol

    def plot_data(self):
        for i in range(len(self.output_dict["train"][0])):
            if np.shape(self.output_dict["train"][0][i]) != (60,6):
                print(np.shape(self.output_dict["train"][0][i]))
                break
        # print(np.shape(self.output_dict["train"]))
        plt.plot(self.output_dict["train"][0][0])
        plt.show()

    def retrieve_data(self, file, label_dict):
        value = torch.load(file)
        print(value.size())
        value, emotion_tags, actor_tags = self.reshape_data(self.process_sampling(value), label_dict)
        return value, emotion_tags, actor_tags

    def return_data_(self):
        return self.output_dict

    def return_data_tags(self):
        return self.output_dict_tags

      