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
from scipy.stats import gaussian_kde

def ks_statistic_kde(data1, data2, num_points=None):
    """Calculate the Kolmogorov-Smirnov statistic to compare two sets of data.
    Kernel Density Estimation is used to estimate the distribution of each set
    set of 1-dimensional data points. From this, the K-S statistic is
    calculated: the maximum absolute distance between the two cumulative
    distribution functions.
    Parameters:
        data1, data2: 1-dimensional lists or arrays of data points to compare.
        num_points: number of points to evaluate the density along.
    """
    xs, kd1, kd2 = _get_estimators_and_xs(data1, data2, num_points)
    with np.errstate(under='ignore'):
        cdf1 = np.array([kd1.integrate_box_1d(-np.inf, x) for x in xs])
        cdf2 = np.array([kd2.integrate_box_1d(-np.inf, x) for x in xs])
    return abs(cdf1 - cdf2).max()

def _get_kd_estimator_and_xs(data, num_points):
    """Get KDE estimator for a given dataset, and generate a good set of
    points to sample the density at."""
    data = np.asarray(data, dtype=np.float64)
    kd_estimator = gaussian_kde(data)
    data_samples = kd_estimator.resample(num_points//2)[0]
    xs = np.sort(data_samples)
    return kd_estimator, xs

def _get_estimators_and_xs(data1, data2, num_points):
    """Get KDE estimators for two different datasets and a set of points
    to evaluate both distributions on."""
    if num_points is None:
        num_points = min(5000, (len(data1) + len(data2))//2)
    kd1, xs1 = _get_kd_estimator_and_xs(data1, num_points//2)
    kd2, xs2 = _get_kd_estimator_and_xs(data2, num_points//2)
    xs = np.sort(np.concatenate([xs1, xs2]))
    return xs, kd1, kd2

def _get_point_estimates(data1, data2, num_points):
    """Get point estimates for KDE distributions for two different datasets.
    """
    xs, kd1, kd2 = _get_estimators_and_xs(data1, data2, num_points)
    with np.errstate(under='ignore'):
        p1 = kd1(xs)
        p2 = kd2(xs)
    return xs, p1, p2

def _kl_divergence(xs, p1, p2):
    """Calculate Kullback-Leibler divergence of p1 and p2, which are assumed to
    values of two different density functions at the given positions xs.
    Return divergence in nats."""
    with np.errstate(divide='ignore', invalid='ignore'):
        kl = p1 * (np.log(p1) - np.log(p2))
    kl[~np.isfinite(kl)] = 0 # small numbers in p1 or p2 can cause NaN/-inf, etc.
    return np.trapz(kl, x=xs) #integrate curve

def js_metric(data1, data2, num_points=None):
    """Calculate the Jensen-Shannon metric to compare two sets of data.
    Kernel Density Estimation is used to estimate the distribution of each set
    set of 1-dimensional data points. From this, the J-S metric (square root of
    J-S divergence) is calculated.
    Note: KDE will often underestimate the probability at the far tails of the
    distribution (outside of where supported by the data), which can lead to
    overestimates of K-L divergence (and hence J-S divergence) for highly
    non-overlapping datasets.
    Parameters:
        data1, data2: 1-dimensional lists or arrays of data points to compare.
        num_points: number of points to evaluate the density along.
    """
    xs, p1, p2 = _get_point_estimates(data1, data2, num_points)
    m = (p1 + p2)/2
    return ((_kl_divergence(xs, p1, m) + _kl_divergence(xs, p2, m))/2)**0.5

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

    from scipy.stats import kstest
    
    print(list(df.columns)[1:-1])
    for n,qualitie in enumerate(df.columns):
        if n > 0 and n < 5:
            aa = pd.concat([df[qualitie],df2[qualitie]],axis=1)
            ax = aa.plot.kde()
            print(aa.mean())
            print(aa.std())
            xs, p1, p2 = _get_point_estimates(df[qualitie].to_numpy(), df2[qualitie].to_numpy(), None)
            print(f"KL Divergence Laban Qualities {qualitie} Human - Network Output: {_kl_divergence(xs, p1, p2)}")
            xs, p1, p2 = _get_point_estimates(df4[qualitie].to_numpy(), df2[qualitie].to_numpy(), None)
            print(f"KL Divergence Laban Qualities {qualitie} Robot - Network Output: {_kl_divergence(xs, p1, p2)}")
            print(f"Kolmogoro-Sirnov Laban Qualities {qualitie} Human - Network Output: {ks_statistic_kde(df[qualitie].to_numpy(), df2[qualitie].to_numpy(), None)}")
            print(f"Kolmogoro-Sirnov Laban Qualities {qualitie} Robot - Network Output: {ks_statistic_kde(df4[qualitie].to_numpy(), df2[qualitie].to_numpy(), None)}")
            print(f"Jensen-Shannon Laban Qualities {qualitie} Human - Network Output: {js_metric(df[qualitie].to_numpy(), df2[qualitie].to_numpy(), None)}")
            print(f"Jensen-Shannon Laban Qualities {qualitie} Robot - Network Output: {js_metric(df4[qualitie].to_numpy(), df2[qualitie].to_numpy(), None)}")
            a = kstest(df2[qualitie].to_numpy(),df[qualitie].to_numpy())
            print(f"Jensen-Shannon Laban Qualities {qualitie} Human - Network Output: {a}")
            b = kstest(df2[qualitie].to_numpy(),df4[qualitie].to_numpy())
            print(f"Jensen-Shannon Laban Qualities {qualitie} Robot - Network Output: {b}")
            plt.show()

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









