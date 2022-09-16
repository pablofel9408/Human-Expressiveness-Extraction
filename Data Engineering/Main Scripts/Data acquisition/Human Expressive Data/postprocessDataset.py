"""
add data scaling based on the training 
"""
import os
import utilities
import augmentation
import copy
import random
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class postprocessEmotionData():
    def __init__(self, constants) -> None:
        
        self.cst = constants
        self.mocap_dataset = None 
        self.new_dataset = None
        self.dataset_matrices = {}
        self.max_time = 0

        self.exclude_tags = ['exp_weight', 'exp_laban_time', 'exp_flow',
                        'exp_bounding_vol_box', 'exp_space']

        coordinates = ['X','Y','Z']
        self.scalers = {}
        for sign_name in self.cst['output_signals']:
            if sign_name in self.exclude_tags:
                continue
            for coord in coordinates:
                self.scalers[sign_name+'_'+coord] = TimeSeriesScalerMinMax(value_range=(0,1))


        self.output_dict = {i:None for i in ['train','val', 'test']}

    def load_data(self, mocap_dataset):
        self.mocap_dataset = mocap_dataset

    def reformat_dataset(self):
        output_dict = {i:{'motion':[],'time':[]} if i not in self.exclude_tags else None for i in self.cst['output_signals']}
        for val in self.cst['candidate']:
            spec_keys = [val+'_'+mark for mark in self.cst['new_markers']]
            for traj in self.mocap_dataset[val].keys():
                for marker in spec_keys:
                    for key in self.cst['output_signals']:
                        if key in self.exclude_tags:
                            continue

                        output_dict[key]['motion'].append(self.mocap_dataset[val][traj][marker][key])
                        output_dict[key]['time'].append(self.mocap_dataset[val][traj]['Time'][-1])
        self.new_dataset = copy.deepcopy(output_dict)

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

        weight = np.concatenate((weight_tot,np.expand_dims(weight_sum,axis=1)),axis=1)

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

        laban_time = np.concatenate((laban_time_tot,np.expand_dims(laban_time_sum,axis=1)),axis=1)

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

        laban_flow = np.concatenate((laban_flow_tot,np.expand_dims(laban_flow_sum,axis=1)),axis=1)

        return laban_flow

    def calc_laban_space(self, input_sign, time_inter=10.0): 
        alpha_w = 1.0
        n = round(np.shape(input_sign)[1]/time_inter)
        laban_space_tot = []
        laban_space_sum = []
        for k,val in enumerate(input_sign):
            arr = [val[i:i+n,:] for i in range(0,len(val), n)]
            aux = []
            tot_norm = np.linalg.norm(arr[-1]-arr[0])
            for num in range(1,len(arr)):
                aux.append(((alpha_w * np.linalg.norm(arr[num]-arr[num-1]))/ tot_norm)**2)
            laban_space_tot.append(aux)
            laban_space_sum.append(sum(aux))

        laban_space = np.concatenate((laban_space_tot,np.expand_dims(laban_space_sum,axis=1)),axis=1)

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

        bound_vol = np.concatenate((bound_vol_tot,np.expand_dims(bound_vol_sum,axis=1)),axis=1)

        return bound_vol

    def proc_eng_features(self): 
    
        for key in self.cst['output_signals']:
            if key == 'exp_weight':
                self.new_dataset[key] = self.calc_laban_weight(self.new_dataset['vel']['motion'],
                                                                time_inter=self.cst['time_inter'])
                self.new_dataset[key] = np.max(self.new_dataset[key],axis=1)
                # self.new_dataset[key] = (self.new_dataset[key], max(self.new_dataset[key]))
            
            elif key == 'exp_laban_time':
                self.new_dataset[key] = self.calc_laban_time(self.new_dataset['acc']['motion'],
                                                                time_inter=self.cst['time_inter'])
                self.new_dataset[key] = np.sum(self.new_dataset[key],axis=1)
                # self.new_dataset[key] = (self.new_dataset[key], sum(self.new_dataset[key]))

            elif key == 'exp_flow':
                self.new_dataset[key] = self.calc_laban_flow(self.new_dataset['jerk']['motion'],
                                                                time_inter=self.cst['time_inter'])
                self.new_dataset[key] = np.sum(self.new_dataset[key],axis=1)
                # self.new_dataset[key] = (self.new_dataset[key], sum(self.new_dataset[key]))

            elif key == 'exp_space':
                self.new_dataset[key] = self.calc_laban_space(self.new_dataset['pos']['motion'],
                                                                time_inter=self.cst['time_inter'])
                self.new_dataset[key] = np.sum(self.new_dataset[key],axis=1)
                # self.new_dataset[key] = (self.new_dataset[key], sum(self.new_dataset[key]))
            
            elif key == 'exp_bounding_vol_box':
                self.new_dataset[key] = self.calc_vol_bound_box(self.new_dataset['pos']['motion'],
                                                                time_inter=self.cst['time_inter'])
                self.new_dataset[key] = np.sum(self.new_dataset[key],axis=1)
                # self.new_dataset[key] = (self.new_dataset[key], sum(self.new_dataset[key]))

    def data_augmentation(self):

        aux_arr = []
        labels = []
        for key in self.cst['output_signals']:
            if key in self.exclude_tags:
                continue
            
            # print(len(self.new_dataset[key]['motion']))
            aux_arr.append(self.new_dataset[key]['motion'])
            labels.append([key]*np.shape(self.new_dataset[key]['motion'])[0])

            new_signals_jit = augmentation.jitter(np.array(self.new_dataset[key]['motion']), sigma=0.002)
            new_signals_mw = augmentation.magnitude_warp(np.array(self.new_dataset[key]['motion']), sigma=0.8, knot=6) 
            new_signals_ww = augmentation.window_warp(np.array(self.new_dataset[key]['motion']), window_ratio=0.2)
            
            self.new_dataset[key]['motion'] = np.concatenate((self.new_dataset[key]['motion'],new_signals_jit,
                                                                new_signals_mw,new_signals_ww),axis=0)
            self.new_dataset[key]['time'] = self.new_dataset[key]['time']*5
        
        aux_arr = np.array(aux_arr).reshape((np.shape(aux_arr)[0]*np.shape(aux_arr)[1],380,3))
        labels = np.array(labels).flatten()
        new_signals_rw = augmentation.random_guided_warp(aux_arr,labels, slope_constraint='asymmetric')

        for key in self.cst['output_signals']:
            if key in self.exclude_tags:
                continue
  
            self.new_dataset[key]['motion'] = np.concatenate((self.new_dataset[key]['motion'],
                                                                new_signals_rw[np.where(labels==key)]),axis=0)
            self.new_dataset[key]['time'] = self.new_dataset[key]['time'] + self.new_dataset[key]['time'][:320]
        
        # x = np.linspace(0,self.new_dataset['pos']['time'][1500], len(self.new_dataset['pos']['motion'][1500]))
        # plt.plot(x,self.new_dataset['pos']['motion'][1500,:,0])
        # plt.plot(x,self.new_dataset['pos']['motion'][220,:,0])
        # plt.show()

        # utilities.close_script() 

    def reformat_dict_to_matrix(self):
        
        self.dataset_matrices['body_signals'] = np.zeros((np.shape(self.new_dataset['pos']['motion'])[0],
                                                            np.shape(self.new_dataset['pos']['motion'])[1],
                                                            np.shape(self.new_dataset['pos']['motion'])[2]*5))

        self.dataset_matrices['eng_feats'] = np.zeros((np.shape(self.new_dataset['exp_weight'])[0],5))

        self.dataset_matrices['time'] = self.new_dataset['pos']['time']
        body_sign_count = 3*len(set(self.cst['output_signals']) - set(self.exclude_tags))

        count = 0                                                   
        for key in self.cst['output_signals']:
            # if key == 'ang_vel':
            #     continue
            
            if key in self.exclude_tags:
                # motion = self.new_dataset[key]
                # if key == 'exp_space':
                #     print(np.shape(self.new_dataset[key]))
                #     utilities.close_script()
                # motion = np.concatenate((self.new_dataset[key],np.zeros((np.shape(self.new_dataset[key])[0],1))),axis=1)
                # aux_motion = motion.copy() if len(np.shape(motion))>1 else np.expand_dims(motion,axis=1)
                self.dataset_matrices['eng_feats'][:,count] = self.new_dataset[key]
                count+=1
            
            else:
                self.dataset_matrices['body_signals'][:,:,count:count+3] = self.new_dataset[key]['motion'].copy()
                count+=3
                if count >= body_sign_count:
                    count=0
         
    def split_dataset(self):

        # train is now 75% of the entire data set
        # the _junk suffix means that we drop that variable completely
        x_train, x_test, y_train, y_test = train_test_split(self.dataset_matrices['body_signals'],
                                                            self.dataset_matrices['eng_feats'], 
                                                            test_size=1 - self.cst['train_ratio'])

        # test is now 10% of the initial data set
        # validation is now 15% of the initial data set
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, shuffle=False,
                                                        test_size=self.cst['test_ratio']/(self.cst['test_ratio'] \
                                                            + self.cst['validation_ratio']))

        self.output_dict['train'] = [x_train, y_train]
        self.output_dict['val'] = [x_val, y_val]
        self.output_dict['test'] = [x_test, y_test]

    def scale_dataset(self): # -> scaling based on training data values
        
        self.output_dict['train_scaled'] = copy.deepcopy(self.output_dict['train'])
        self.output_dict['val_scaled'] = copy.deepcopy(self.output_dict['val'])
        self.output_dict['test_scaled'] = copy.deepcopy(self.output_dict['test'])

        dirpath = os.path.join(self.cst["dataset_dirpath"],'Scalers\\')
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        utilities.gen_archive(dirpath)

        scaler =  TimeSeriesScalerMinMax(value_range=(0,1))
        scaler.fit(self.output_dict['train'][0])
        self.output_dict['train_scaled'][0]=scaler.transform(self.output_dict['train'][0])
        self.output_dict['val_scaled'][0]=scaler.transform(self.output_dict['val'][0])
        self.output_dict['test_scaled'][0]=scaler.transform(self.output_dict['test'][0])

        # for idx, (key,value) in enumerate(self.scalers.items()):
        #     value.fit(self.output_dict['train'][0][:,:,idx])
        #     self.output_dict['train_scaled'][0][:,:,idx] = np.squeeze(value.transform(self.output_dict['train'][0][:,:,idx]),axis=2)
        #     self.output_dict['val_scaled'][0][:,:,idx] = np.squeeze(value.transform(self.output_dict['val'][0][:,:,idx]),axis=2)
        #     self.output_dict['test_scaled'][0][:,:,idx] = np.squeeze(value.transform(self.output_dict['test'][0][:,:,idx]),axis=2)
            
        scaler_filename = os.path.join(dirpath, "scaler_dataset_" + 'train' + ".pickle")
        with open(scaler_filename, "wb") as output_file:
            pickle.dump(scaler, output_file) 

        # scaler = joblib.load(scaler_filename)

    def plot_output_dict(self, plot_num=10):
        map_dict_titles = ['Position','Velocity','Acceleration', 'Jerk']
        map_dict = {0:'X', 1:'Y', 2:'Z'}
        map_dict_y_axis= ['Position in meters(m)', 'Velocity in meters per second(m/s)', 
                            'Acceleration in meters per second squared(m/s^2)','Jerk(m/s^3)']
        for _ in range(plot_num):
            dataset_len = [len(self.output_dict['train'][0]),len(self.output_dict['val'][0]),len(self.output_dict['test'][0])]
            p_list = [random.randrange(0,dataset_len[i]) for i in range(3)]
            for key,value in self.output_dict.items():
                print(key)
                if "train" in key:
                    p=p_list[0]
                elif "val" in key:
                    p=p_list[1]
                elif "test" in key:
                    p=p_list[2]
                # if not ("_scaled" in key):
                #     continue
                
                print(f'Dataset index selected randomly. Index Value: {p}')
                for k in range(0,np.shape(value[0])[2],3):

                    indx=0
                    if k > 2 and k < 5:
                        indx=1
                    elif (k > 5 and k < 8):
                        indx=2
                    elif k > 8:
                        indx=3

                    if indx==0:
                        fig, axs = plt.subplots(1, 3)
                        fig.suptitle('Scaled ' + key + ' Mocap Data' + map_dict_titles[indx])
                        time = np.linspace(0,len(value[0][0,:,0]),len(value[0][0,:,0]))
                        for num, axes in enumerate(axs.flat):
                            axes.plot(time, value[0][p,:,num+k])
                            axes.set_title(map_dict[num] + '- Axis')
                            axes.set(xlabel='Sample number', ylabel=map_dict_y_axis[indx])
                plt.show()

    def convert_array_to_tensor(self):
        
        for key in self.output_dict.keys():
            self.output_dict[key] = [torch.tensor(self.output_dict[key][0]),torch.tensor(self.output_dict[key][1])]

    def get_stats(self):
        if os.path.exists('dataset stats.txt'):
            os.remove('dataset stats.txt')
        
        axis = ['X','Y','Z']
        for key in self.cst['output_signals']: 
            # if key == 'ang_vel':
            #     continue
            
            with open('dataset stats.txt', 'a+') as f:
                if key not in self.exclude_tags: 
                    motion = self.new_dataset[key]['motion']
                    corr = utilities.get_cross_correlation(motion)
                    for coord in range(np.shape(self.new_dataset[key]['motion'])[2]):
                        f.write(f'-----{key} Signal----- \n')
                        f.write(f'-----Coordinate {axis[coord]}----- \n')
                        f.write(f'{key} Coordinate {axis[coord]} Mean Value: {np.mean(np.mean(motion[:,:,coord],axis=1))} \n')
                        f.write(f'{key} Coordinate {axis[coord]} Standard Deviation: {np.mean(np.std(motion[:,:,coord],axis=1))} \n')
                        f.write(f'{key} Coordinate {axis[coord]} Correlation Max Mean: {corr[0][0][coord]} \n')
                        f.write(f'{key} Coordinate {axis[coord]} Correlation Max Std: {corr[0][1][coord]} \n')
                        f.write(f'{key} Coordinate {axis[coord]} Correlation Min Mean: {corr[1][0][coord]} \n')
                        f.write(f'{key} Coordinate {axis[coord]} Correlation Min Std: {corr[1][1][coord]} \n')
                        f.write(f'{key} Coordinate {axis[coord]} Max Value: {np.mean(np.amax(motion[:,:,coord],axis=1))} \n')
                        f.write(f'{key} Coordinate {axis[coord]} Min Value: {np.mean(np.amin(motion[:,:,coord],axis=1))} \n')
                        f.write(f'\n')
                else:
                    motion = self.new_dataset[key]
                    f.write(f'-----{key} Signal----- \n')
                    f.write(f'{key} Mean Value: {np.mean(motion,axis=0)} \n')
                    f.write(f'{key} Standard Deviation: {np.std(motion,axis=0)} \n')
                    f.write(f'{key} Max Value: {np.amax(motion,axis=0)} \n')
                    f.write(f'{key} Min Value: {np.amin(motion,axis=0)} \n')
                    f.write(f'\n')

    def plot_dataset(self, plot_num=10):
        dataset_len = len(self.new_dataset['pos']['motion'])
        idx_used =[]
        for k in range(plot_num):

            p = random.randrange(0,dataset_len)
            if k==0:
                idx_used.append(p)
            else:
                while p in idx_used:
                    p = random.randrange(0,dataset_len) 
                idx_used.append(p)

            print(f'Dataset index selected randomly. Index Value: {p}')
            for key in self.cst['output_signals']: 
                
                if not isinstance(self.new_dataset[key],dict):
                    array_oi = self.new_dataset[key][p,:-1]
                    print(f'Feature {key} value: {self.new_dataset[key][p,-1]}')
                else:
                    array_oi = self.new_dataset[key]['motion'][p,:,:]
                array_oi_size = np.shape(array_oi)
                print(f'Signal {key} size: {array_oi_size}')

                if key=='pos':

                    try:
                        fig, axs = plt.subplots(1, array_oi_size[1])
                    except:
                        fig, axs = plt.subplots(1,1)

                    map_dict_titles = {'pos':'Position', 'vel':'Velocity', 
                                        'acc':'Acceleration', 'ang_vel':'Angular Velocity', 
                                        'jerk': 'Jerk', 'exp_weight': 'Laban Weight','exp_laban_time': 'Laban Time',
                                        'exp_flow': 'Laban Flow','exp_space': 'Laban Space',
                                        'exp_bounding_vol_box': 'Volume Bounding Box'}
                    fig.suptitle('Mocap Data ' + map_dict_titles[key])

                    map_dict = {0:'X', 1:'Y', 2:'Z'}
                    map_dict_y_axis= {'pos':'Position in meters(m)', 'vel':'Velocity in meters per second(m/s)', 
                                        'acc':'Acceleration in meters per second squared(m/s^2)', 'jerk': 'Jerk(m/s^3)',
                                        'ang_vel':'Angular Velocity(rad/s)', 'exp_weight': 'Laban Weight',
                                        'exp_laban_time': 'Laban Time','exp_flow': 'Laban Flow',
                                        'exp_space': 'Laban Space', 'exp_bounding_vol_box': 'Volume Bounding Box'}

                    try:
                        time = np.linspace(0,self.new_dataset[key]['time'][p],array_oi_size[0])
                        for num, axes in enumerate(axs.flat):
                            axes.plot(time, array_oi[:,num])
                            axes.set_title(map_dict[num] + '- Axis')
                            axes.set(xlabel='Sample number', ylabel=map_dict_y_axis[key])
                    except:
                        time = np.linspace(0,array_oi_size[0],array_oi_size[0])
                        axs.plot(time, array_oi)
                        axs.set_title(map_dict_titles[key])
                        axs.set(xlabel='Sample number', ylabel=map_dict_y_axis[key])

            plt.show()
            # input("Press Enter to continue...")

    def print_dataset_shapes(self):

        for key in self.cst['output_signals']:
            if key not in self.exclude_tags: 
                print(f"Key:{key} Motion, Shape:{np.shape(self.new_dataset[key]['motion'])}")
                print(f"Key:{key} Time, Shape:{np.shape(self.new_dataset[key]['time'])}")
            else: 
                print(f"Key:{key}, Shape:{np.shape(self.new_dataset[key])}")

    def save_dataset(self):
        save_tags = ['body_signals','eng_feat_expressive']

        dirpath = os.path.join(self.cst["dataset_dirpath"],'Data\\')
        # if not os.path.exists(dirpath):
        #     os.makedirs(dirpath)
        # utilities.gen_archive(dirpath)

        for key in self.output_dict.keys():
            for k, data_tensor in enumerate(self.output_dict[key]):
                print(data_tensor.size())
                torch.save(data_tensor,os.path.join(dirpath,'Tensor_human_'+key+'_'+save_tags[k]+'.pt'))

    def start_preprocess(self):

        self.reformat_dataset()

        if self.cst['set_data_aug']:
            self.data_augmentation()

        self.proc_eng_features()
        self.print_dataset_shapes()

        if self.cst['get_stats']:
            self.get_stats()

        if self.cst['plot_dataset']:
            self.plot_dataset(plot_num=self.cst['num_plot_samples'])

        self.reformat_dict_to_matrix()
        self.split_dataset()

        if self.cst['scale_dataset']:
            self.scale_dataset()
            # self.plot_output_dict(plot_num=self.cst['num_plot_samples'])

        if self.cst['convert_to_tensor']:
            self.convert_array_to_tensor()

        if self.cst['save_tensors']:
            self.save_dataset()

    def return_processed_data(self):
        return self.mocap_dataset
                                        