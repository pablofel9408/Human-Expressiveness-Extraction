import json
import os, fnmatch
import sys
import copy
import numpy as np
import scipy

def multiple(m, n):
    a = list(range(n, (m * n)+1, n))
    return a

def save_numpy(dirpath, filename, data):
    with open(os.path.join(dirpath, filename + '.npy'), 'wb') as f:                 
            np.save(f, data)

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def save_json(filepath,data,name='base.json'):
    with open(os.path.join(filepath,name), 'w') as fp:
        json.dump(data, fp)

def load_raw_constants(filepath):
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data

def load_constants(filepath,task):
    with open(filepath) as json_file:
        data = json.load(json_file)
    
    aux_dict = copy.deepcopy(data)
    data = data[task]
    data['dataset_path_twist'] = aux_dict['dataset_path_twist']
    data['dataset_path_joint'] = aux_dict['dataset_path_joint']
    data['dataset_path_json'] = aux_dict['dataset_path_json']
    return data

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[:,0], r[:,1], r[:,2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return np.column_stack((qw, qx, qy, qz))

def quaternion_multiply(quaternion1, quaternion0):
    q = np.copy(quaternion1)
    q[:,0] = -quaternion1[:,1] * quaternion0[:,1] - quaternion1[:,2] * quaternion0[:,2] - quaternion1[:,3] * quaternion0[:,3] + quaternion1[:,0] * quaternion0[:,1]
    q[:,1] = quaternion1[:,1] * quaternion0[:,0] + quaternion1[:,2] * quaternion0[:,3] - quaternion1[:,3] * quaternion0[:,2] + quaternion1[:,0] * quaternion0[:,1]
    q[:,2] = -quaternion1[:,1] * quaternion0[:,3] + quaternion1[:,2] * quaternion0[:,0] + quaternion1[:,3] * quaternion0[:,1] + quaternion1[:,0] * quaternion0[:,2]
    q[:,3] = quaternion1[:,1] * quaternion0[:,2] - quaternion1[:,2] * quaternion0[:,1] + quaternion1[:,3] * quaternion0[:,0] + quaternion1[:,0] * quaternion0[:,3]
    return q

def conjugate(q):
    q[:,1:] = -q[:,1:]
    return q

def rotate(a,q):
    a = np.column_stack((np.zeros((np.shape(a)[0],1)),a))
    q0_conj = conjugate(q)
    v_r = quaternion_multiply(a,q0_conj)
    v_r = quaternion_multiply(q,v_r)
    return v_r

def gravity_compensation(a,q):
    gravity = np.ones((np.shape(a)[0],np.shape(a)[1]+1))*[0,0, 0, -9.81]
    a_rotated = rotate(a, q)
    user_acceleration = a_rotated - gravity
    return user_acceleration

def filter_signal(x, cutoff_freq= 0.8, filt_type='lp', fs=100):
    sos = scipy.signal.butter(2, cutoff_freq, filt_type, fs=fs,  output='sos')
    for ax in range(np.shape(x)[1]):
        x[:,ax] = scipy.signal.sosfiltfilt(sos,x[:,ax])
    return x

def get_cross_correlation(intput_signal,cols_oi=["v_x", "v_y", "v_z", "v_ang_x", "v_ang_y", "v_ang_z"]):
    output_dict = {i:None for i in cols_oi}
    for n,key in enumerate(output_dict.keys()):
        print(key)
        init_sig = intput_signal[0,:,n]
        final_corr_max = []
        final_corr_min = []
        for _,arr in enumerate(intput_signal[:,:,n],1):
            corr = scipy.signal.correlate(init_sig, arr,mode='same')
            if abs(np.min(corr)) > abs(np.max(corr)):
                corr /= abs(np.min(corr))
            else:
                corr /= abs(np.max(corr))
            final_corr_max.append(np.amax(corr,axis=0))
            final_corr_min.append(np.amin(corr,axis=0))

        output_dict[key] = {'Max_corr': (np.mean(final_corr_max,axis=0), np.std(final_corr_max,axis=0)), \
                            'Min_corr': (np.mean(final_corr_min,axis=0), np.std(final_corr_min,axis=0)),
                            'Mean': np.mean(np.mean(intput_signal[:,:,n],axis=1)),
                            'Mean_Max_value': np.mean(np.max(intput_signal[:,:,n],axis=1)),
                            'Mean_Min_value': np.mean(np.min(intput_signal[:,:,n],axis=1)),
                            'Max_value': np.max(np.max(intput_signal[:,:,n],axis=1)),
                            'Min_value': np.min(np.min(intput_signal[:,:,n],axis=1))
                        }
        print(output_dict)
    return output_dict