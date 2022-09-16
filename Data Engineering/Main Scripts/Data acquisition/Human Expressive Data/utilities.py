import scipy.signal
import numpy as np
import json
import os 
import shutil
import numba as nb

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
            list_subfolders_with_paths = [float(path_name.split('\\')[-1].split('_')[-1]) for path_name in list_subfolders_with_paths]
            index_oi = max(list_subfolders_with_paths)
            index_oi += 0.1
            os.makedirs(os.path.join(path,'Version_'+str(round(index_oi,2))+'\\'))

        archive_path = os.path.join(path,'Version_'+str(round(index_oi,2))+'\\')
        # archive_path = [os.path.join(archive_path, f) for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
        files = [os.path.join(dirname, f) for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f))]
        # print(files)
        # print(archive_path)
        # import sys
        # sys.exit()
        for n,f in enumerate(files):
            print("FILE: " + f)
            print("\t" + archive_path)
            shutil.copy(f, archive_path,follow_symlinks=True)
            # os.remove(f)

def load_json(config_path):

    with open(config_path) as json_file:
        data = json.load(json_file)
    
    return data

def save_json(config_path, data):

    with open(config_path,'w') as json_file:
        json.dump(data,json_file)

def filter_signal(x, cutoff_freq= 0.8, filt_type='lp'):
    sos = scipy.signal.butter(2, cutoff_freq, filt_type, fs=30.303,  output='sos')
    for ax in range(np.shape(x)[1]):
        x[:,ax] = scipy.signal.sosfiltfilt(sos,x[:,ax])

    return x

def get_cross_correlation(intput_signal):

    init_sig = intput_signal[0]
    final_corr_max = []
    final_corr_min = []
    for _,arr in enumerate(intput_signal[:320],1):
        corr = scipy.signal.correlate(init_sig, arr,mode='same')
        corr /= np.max(corr)
        final_corr_max.append(np.amax(corr,axis=0))
        final_corr_min.append(np.amin(corr,axis=0))

    return (np.mean(final_corr_max,axis=0),np.std(final_corr_max,axis=0)), \
                 (np.mean(final_corr_min,axis=0),np.std(final_corr_min,axis=0))

def close_script():
    import sys
    sys.exit()

@nb.njit(fastmath=True,parallel=True)
def calc_cros(vec_1,vec_2):
    res=np.empty((vec_1.shape[0],vec_2.shape[0],3),dtype=vec_1.dtype)
    for i in nb.prange(vec_1.shape[0]):
        for j in range(vec_2.shape[0]):
            res[i,j,0]=vec_1[i,1] * vec_2[j,2] - vec_1[i,2] * vec_2[j,1]
            res[i,j,1]=vec_1[i,2] * vec_2[j,0] - vec_1[i,0] * vec_2[j,2]
            res[i,j,2]=vec_1[i,0] * vec_2[j,1] - vec_1[i,1] * vec_2[j,0]
    
    return res