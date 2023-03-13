import os
import sys
import re
import shutil
import fnmatch
import json
import random

import matplotlib.pyplot as plt
import seaborn as sns

import scipy
from scipy.stats import gaussian_kde
import torch.linalg as LA

def close_script():
    print("Exit singal sent, closing script.....")
    sys.exit()

def findDirs(path,pattern):
    resultList=[]
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            if name == pattern:
                resultList.append(os.path.join(root, name))
    return resultList

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        if "Archive" in dirs:
            dirs.remove("Archive")
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def load_raw_constants(filepath):
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data

def gen_archive(dirname):
    files = [os.path.join(dirname, f) for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
    if files:
        path = os.path.join(dirname,'Archive\\')
        if not os.path.exists(path):
            os.makedirs(path)
        if  not os.path.exists(os.path.join(path,'Version_0.0\\')):
            os.makedirs(os.path.join(path,'Version_0.0\\'))
            index_oi = 0.0
        else:
            list_subfolders_with_paths = [f.path for f in os.scandir(path) if f.is_dir()]
            list_subfolders_with_paths = [float(path_name.split('/')[-1].split('_')[-1]) for path_name in list_subfolders_with_paths]
            index_oi = max(list_subfolders_with_paths)
            index_oi += 0.1
            os.makedirs(os.path.join(path,'Version_'+str(index_oi)+'\\'))
        archive_path = os.path.join(path,'Version_'+str(index_oi)+'\\')
        files = [os.path.join(dirname, f) for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
        for f in files:
            shutil.copy(f, archive_path)
            os.remove(f)

def plot_kernels(tensor, num_cols=6):
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def filter_signal(x, cutoff_freq=0.8, fs=30.33, filt_type='lp'):
    sos = scipy.signal.butter(2, cutoff_freq, filt_type, fs=fs,  output='sos')
    for ax in range(np.shape(x)[1]):
        x[:,ax] = scipy.signal.sosfiltfilt(sos,x[:,ax])

    return x

def filter_signal_2(x, cutoff_freq=4, filt_type='lp'):
    sos = scipy.signal.butter(2, cutoff_freq, filt_type, fs=60,  output='sos')
    for ax in range(np.shape(x)[0]):
        x[ax,:] = scipy.signal.sosfiltfilt(sos,x[ax,:])

    return x

def groupby(a, b):
    # Get argsort indices, to be used to sort a and b in the next steps
    sidx = b.argsort(kind='mergesort')
    a_sorted = a[sidx]
    b_sorted = b[sidx]

    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(np.r_[True,b_sorted[1:] != b_sorted[:-1],True])

    # Split input array with those start, stop ones
    out = [a_sorted[i:j] for i,j in zip(cut_idx[:-1],cut_idx[1:])]
    return out

def pop_random(lst):
    idx = random.randrange(0, len(lst))
    return lst.pop(idx)

def get_pairs(lst):
    pairs = []
    while lst:
        rand1 = pop_random(lst)
        pairs.append(rand1)
    return pairs

def calc_expressive_qualities(input_signals, alpha=1.7):
    output_dict = {}
    output_dict["weight"] = calc_laban_weight(input_signals["vel"], alpha_w=alpha)[0]
    output_dict["time"] = calc_laban_time(input_signals["acc"], alpha_w=alpha)[0]
    output_dict["flow"] = calc_laban_flow(input_signals["jerk"], alpha_w=alpha)[0]
    output_dict["space"] = calc_laban_space(input_signals["pos"], alpha_w=alpha)[0]
    output_dict["bound_vol"] = calc_vol_bound_box(input_signals["pos"])[0]

    return output_dict

def calc_laban_weight(input_sign, time_inter=1.0,alpha_w = 1.7): 
    n = round(np.shape(input_sign)[1]/time_inter)
    weight_tot = []
    weight_sum = []
    for k,val in enumerate(input_sign):
        aux = []
        sub_arr = [val[i:i+n,:] for i in range(0,len(val), n)]
        for arr in sub_arr:
            # print(np.shape(arr))
            aux.append(max(alpha_w * np.linalg.norm(arr, ord=2, axis=-1)**2))
        weight_tot.append(aux)
        weight_sum.append(max(aux))

    # weight = torch.cat((weight_tot,torch.unsqueeze(weight_sum,dim=1)),dim=1)
    weight = np.array(weight_sum)
    return weight

def calc_laban_time(input_sign, time_inter=1.0,alpha_w = 1.7): 
    n = round(np.shape(input_sign)[1]/time_inter)
    laban_time_tot = []
    laban_time_sum = []
    for k,val in enumerate(input_sign):
        aux = []
        sub_arr = [val[i:i+n,:] for i in range(0,len(val), n)]
        for arr in sub_arr:
            aux.append(sum(((alpha_w * np.linalg.norm(arr, ord=2, axis=-1))/n)**2))
        laban_time_tot.append(aux)
        laban_time_sum.append(max(aux))
    
    laban_time = np.array(laban_time_sum)
    # laban_time = np.concatenate((laban_time_tot,np.expand_dims(laban_time_sum,axis=1)),axis=1)

    return laban_time

def calc_laban_flow(input_sign, time_inter=1.0,alpha_w = 1.7): 
    n = round(np.shape(input_sign)[1]/time_inter)
    laban_flow_tot = []
    laban_flow_sum = []
    for k,val in enumerate(input_sign):
        aux = []
        sub_arr = [val[i:i+n,:] for i in range(0,len(val), n)]
        for arr in sub_arr:
            aux.append(sum(((alpha_w * np.linalg.norm(arr, ord=2, axis=-1))/n)**2))
        laban_flow_tot.append(aux)
        laban_flow_sum.append(sum(aux))
    
    laban_flow = np.array(laban_flow_sum)
    # laban_flow = np.concatenate((laban_flow_tot,np.expand_dims(laban_flow_sum,axis=1)),axis=1)

    return laban_flow

def calc_laban_space(input_sign, time_inter=2.0,alpha_w = 1.7): 
    n = round(np.shape(input_sign)[1]/time_inter)
    laban_space_tot = []
    laban_space_sum = []
    for k,val in enumerate(input_sign):
        arr = [val[i:i+n,:] for i in range(0,len(val), n)]
        aux = []
        for num in range(1,len(arr)):
            aux.append(sum(((alpha_w *(np.linalg.norm(arr[num]-arr[num-1], ord=1, axis=-1))))**2))
        laban_space_tot.append(aux)
        laban_space_sum.append(sum(aux))

    laban_space = np.array(laban_space_sum)
    # laban_space = np.concatenate((laban_space_tot,np.expand_dims(laban_space_sum,axis=1)),axis=1)

    return laban_space

def calc_vol_bound_box(input_sign, time_inter=1.0): 
    n = round(np.shape(input_sign)[1]/time_inter)
    bound_vol_tot = []
    bound_vol_sum = []
    for num,val in enumerate(input_sign):
        aux = []
        aa = [val[i:i+n,:] for i in range(0,len(val), n)]
        for arr in aa:
            aux.append((max(np.abs(arr[:,0])) * max(np.abs(arr[:,1])) * max(np.abs(arr[:,2])))**2)
        bound_vol_tot.append(aux)
        bound_vol_sum.append(sum(aux))

    bound_vol = np.array(bound_vol_sum)
    return bound_vol
    # bound_vol = np.concatenate((bound_vol_tot,np.expand_dims(bound_vol_sum,axis=1)),axis=1)

def has_numbers(inputString):
    return bool(len(re.findall(r'\d', inputString))>1)


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


###########################
###Visualization Helpers###
###########################

def vis_latent_space(inputLat,inpt):
    
    inputLat = inputLat[0]
    inpt = inpt.cpu().detach().numpy()
    plot_kernels(inpt,num_cols=6)
    # aa =  ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice',
    #          'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulczynski1', 'mahalanobis',
    #           'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 
    #           'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
    
    sns.clustermap(data=inputLat.transpose(1,0).cpu().detach().numpy(), figsize=(100, 100))
    plt.show()
    # for i in range(inputLat.size(1)):
    #     plt.plot(inputLat[:,i].cpu().numpy())
    #     plt.show()

def vis_lat_var(inputLat):

    aa = ['antialiased', 'nearest', 'bilinear', 'bicubic', 'spline16', 
    'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 
    'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman']
    for i in aa:
        plt.imshow(inputLat[0].unsqueeze(0).cpu().detach().numpy(), interpolation=i)
        plt.show()

import numpy as np
def vis_latent_space_lstm(inputLat):

    for hid in inputLat:
        print(np.shape(hid))
        plt.imshow(hid[:,0,:].cpu().numpy())
        plt.show()
        # close_script()


import torch
def saliency(img, model):
    img = img[0]
    #we don't need gradients w.r.t. weights for a trained model
    for param in model.parameters():
        param.requires_grad = False
    
    #set model in eval mode
    model.eval()
    #transoform input PIL image to torch.Tensor and normalize
    # input = transform(img)
    # input.unsqueeze_(0)

    #we want to calculate gradient of higest score w.r.t. input
    #so set requires_grad to True for input 
    img.requires_grad = True
    #forward pass to calculate predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preds,_,_,_,_,_,_ = model(img.float().to(device))
    score, indices = torch.max(preds, 1)
    #backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()
    #get max along channel axis
    slc, _ = torch.max(torch.abs(img.grad[0]), dim=0)
    #normalize to [0..1]
    slc = (slc - slc.min())/(slc.max()-slc.min())

    #apply inverse transform on image
    # with torch.no_grad():
    #     input_img = inv_normalize(input[0])
    #plot image and its saleincy map
    # plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(img.detach().numpy(), (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(slc.numpy(), cmap=plt.cm.hot)
    plt.xticks([])
    plt.yticks([])
    plt.show()






