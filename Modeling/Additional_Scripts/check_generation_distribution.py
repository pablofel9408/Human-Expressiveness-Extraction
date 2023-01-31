import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.linalg as LA
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

def append_csv_lines(filename,tag):
    with open(filename, 'a') as f:
        f.write('\n')
        f.write(tag)
        f.write('\n')

sample_idx = 3
index_list = [[8151,1018,1018],[2386,298,298],[6176,772,772],[6176,772,772]]
index_list = index_list[sample_idx]
tags = ["train"]
for idx,tag in enumerate(tags):
    df = pd.read_csv("C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\laban_qualities_"+ tag + "_human.csv")
    df4 = pd.read_csv("C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\laban_qualities_"+ tag + "_robot.csv")
    df2 = pd.read_csv("C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\generated_single_sample_"+ tag + "_seed_" + str(index_list[idx]) + ".csv")
    df3 = pd.read_csv("C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Translation_Dataset_Analysis\\Data\\emotion_dataset_"+ tag+ ".csv")

    # indices = list(df3[df3["emo"]!="NEE"].index)
    # print(df.index)
    # print(len(indices))
    # df = df.loc[df.index[indices]]
    # df2 = df2.loc[df2.index[indices]]
    df3 = df3[df3["emo"]!="NEE"].reset_index(drop=True)

    cols = df.columns[1:]
    df.insert(0, 'class', 'human')
    df2.insert(0, 'class', 'robot')
    n = 0
    
    print(df4.loc[[index_list[idx]]])
    stats_csv_name = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Translation_Dataset_Analysis\\Stats\\Sample " + str(sample_idx+1)+"\\General Sample Stats "+ tag +".csv"
    df4.loc[[index_list[idx]]].to_csv(stats_csv_name)
    print("Human Data Mean--------------------")
    print(df[cols].mean())
    append_csv_lines(stats_csv_name, "Human Data Mean")
    df[cols].mean().to_csv(stats_csv_name, mode='a', index=True, header=False)
    print("Human Data Variance--------------------")
    print(df[cols].var())
    append_csv_lines(stats_csv_name, "Human Data Variance")
    df[cols].var().to_csv(stats_csv_name, mode='a', index=True, header=False)
    print("Generated Data Mean------------------")
    print(df2[cols].mean())
    append_csv_lines(stats_csv_name, "Generated Data Mean")
    df2[cols].mean().to_csv(stats_csv_name, mode='a', index=True, header=False)
    print("Generated Data Variance------------------")
    print(df2[cols].var())
    append_csv_lines(stats_csv_name, "Generated Data Variance")
    df2[cols].var().to_csv(stats_csv_name, mode='a', index=True, header=False)
    print("------------------------------------")
    # partial_df = list(df3[df3["emo"]=="TRE"].index)
    # partial_df = df.loc[df.index[partial_df]]
    # # print(partial_df)

    # print(df3["act"].unique())
    # for act in df3["act"].unique():
    #     print(act)
    #     indices = list(df3[df3["act"]==act].index)
    #     df_aux = df.loc[df.index[indices]]
    #     df2_aux = df2.loc[df2.index[indices]]
    #     df3_aux = df3[df3["act"]==act]

    fig, axes = plt.subplots(nrows=4, ncols=2,sharex=False, sharey=False)
    axes = axes.ravel()  # array to 1D
    axes[0].set_title("Human Data")
    axes[1].set_title("Network Output")
    for n,ax in enumerate(axes):
        if n % 2 == 0:
            sns.histplot(df, x=cols[n//2], hue=df3["emo"],ax=ax, bins=100)
        else:
            sns.histplot(df2, x=cols[n//2], hue=df3["emo"], ax=ax, bins=100)
        n+=1
    # fig.suptitle('Expressive Qualities Actor - ' + act + ' - Human Data vs Network Output, Dataset - ' + tag + "seed:" + str(index_list[idx]))
    fig.suptitle('Expressive Qualities Actor - Human Data vs Network Output, Dataset - ' + tag + "seed:" + str(index_list[idx]))
    # plt.ylim(0, 200)
    plt.show()









