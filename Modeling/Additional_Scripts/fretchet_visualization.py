import matplotlib.pyplot as plt
# from .GAN_Translation.General_Functions.loss_functions_variations as losses
import torch 
import torch.nn as nn
import torch.linalg as LA
# import utilities
import numpy as np
import scipy.linalg as linalg

import os
import json

from scipy import integrate
import pickle
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.preprocessing import MinMaxScaler
import copy
import pandas as pd
import seaborn as sns


# input_tensor3 = torch.randn(4,6,10).numpy()
# input_tensor4 = torch.randn(4,6,10)
# input_tensor5 = torch.randn(4,1,2)
# input_tensor6 = torch.randn(4,1,2)
def reshape_dict(user_dict):

    new_dict = {key: {"batch"+str(k): {"feature"+str(j): None for j in range(len(value[0]) )} for k in range(len(value))} for key, value in user_dict.items()}
    for key, value in user_dict.items():
        for n, val in enumerate(value):
            for k, feat in enumerate(val):
                new_dict[key]["batch"+str(n)]["feature"+str(k)] = feat

    df = pd.DataFrame.from_dict({(i,j): new_dict[i][j] 
                           for i in new_dict.keys() 
                           for j in new_dict[i].keys()})
    # df = pd.DataFrame(new_dict)
    return df

def reshape_data(input_dict):
    output_arr= np.array([], dtype=np.float64).reshape(0,10,50)
    for key,value in input_dict.items():
        output_arr = np.concatenate((output_arr,np.expand_dims(value,axis=0)),axis=0)
    return output_arr.transpose((1,0,2))

def calculate_stats(input_dict):
    output_dict = {}
    n=0
    for key,value in input_dict.items():
        output_dict[key] = np.array(value).mean(axis=0)
        output_dict["var_batch"+str(n)] = np.array(value).var(axis=0)
        n+=1
    return output_dict

history_path = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\GAN_Translation\\History"
with open(os.path.join(history_path, "history_fretchet_human" +'.json'), 'rb') as json_file:
    fid_human_history = json.load(json_file)

with open(os.path.join(history_path, "history_fretchet_robot" +'.json'), 'rb') as json_file:
    fid_robot_history = json.load(json_file)

# fid_human_history = reshape_dict(fid_human_history)
fid_human_history = reshape_data(fid_human_history)
fid_robot_history = reshape_data(fid_robot_history)
epoch_num = np.linspace(0,100,100)
print(np.shape(fid_robot_history))
print("Fretchet Distance Last Epoch Robot Data: ",fid_robot_history[:,-1,:].mean(axis=0))
print("Fretchet Distance Last Epoch Human Data: ",fid_human_history[:,-1,:].mean(axis=0))
# for feature in range(np.shape(fid_human_history)[2]):
#     fig,axis = plt.subplots(2,5)
#     for n, ax in enumerate(axis.flatten()):
#         ax.plot(epoch_num, fid_robot_history[n,:,feature])
#     fig.suptitle('Features 1-10 Fretchet Distance', fontsize=16)
#     plt.show()
# fid_human_history = calculate_stats(fid_human_history)
# fid_robot_history = calculate_stats(fid_robot_history)

# df = pd.DataFrame(fid_human_history)
# df["variance"] = df.var(axis=1)
# df["mean"] = df.mean(axis=1)
# print(fid_human_history.loc['features0'])
# sns.displot(fid_human_history, x="flipper_length_mm", hue="species")
# df.to_csv(os.path.join(history_path,"fid_human_stats.csv"))
# df = pd.DataFrame(fid_robot_history)
# df["variance"] = df.var(axis=1)
# df["mean"] = df.mean(axis=1)
# df.to_csv(os.path.join(history_path,"fid_robot_stats.csv"))









