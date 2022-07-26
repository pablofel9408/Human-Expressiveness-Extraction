import os 
import copy
import pickle

import random
import numpy as np
import matplotlib.pyplot as plt
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.model_selection import train_test_split

import torch

import utilities

class PreprocessRobotData():
    def __init__(self, constants) -> None:

        self.cst=constants
        self.datasets={}
        self.norm_datasets={}
        self.scalers={}

        self.output_dict={i:None for i in ['train','test','val']}

    def load_recursive_np(self, filepath):
        trajectories = []
        filepath = utilities.find("*.npy",filepath)
        for file in filepath:
            with open(file, 'rb') as f:                 
                trajectories.append(np.load(f))
        return trajectories

    def load_files(self):
        path_list = utilities.findDirs(self.cst["dataset_paths"],
                                        self.cst["common_folder_name"])
        for dirpath in path_list:
            self.datasets[dirpath.split("\\")[8]] = self.load_recursive_np(dirpath)

    def reshape_data(self):
        return np.vstack([i for i in self.datasets_matrix.values()])

    def preprocess_data(self):
        
        self.datasets_matrix=copy.deepcopy(self.datasets)
        for name,value in self.datasets.items():
            self.datasets_matrix[name]=np.vstack(value)

        self.output_matrix = self.reshape_data()

    def split_dataset(self):
        # train is now 75% of the entire data set
        # the _junk suffix means that we drop that variable completely
        x_train, x_test, _, y_test = train_test_split(self.output_matrix,
                                                        self.output_matrix, shuffle=True,
                                                        test_size=1 - self.cst['train_ratio'])
        # test is now 10% of the initial data set
        # validation is now 15% of the initial data set
        x_val, x_test, _, y_test = train_test_split(x_test, y_test, shuffle=False,
                                                        test_size=self.cst['test_ratio']/(self.cst['test_ratio'] \
                                                            + self.cst['validation_ratio']))
        self.output_dict['train'] = x_train
        self.output_dict['val'] = x_val
        self.output_dict['test'] = x_test 

    def transform_to_tensor(self, input_dict:dict):

        for key,value in input_dict.items():
            input_dict[key] = torch.tensor(value)

        return input_dict

    def postprocess_data(self):

        self.split_dataset()
        
        if self.cst["save_dataset"]:
            self.save_dataset(tag="raw_data")

        if self.cst["normalization"]:
            dirpath = self.cst["scalers_path"]
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            utilities.gen_archive(dirpath)

            scaler =  TimeSeriesScalerMinMax(value_range=(0,1))
            scaler.fit(self.output_dict['train'])
            for key,value in self.output_dict.items():
                self.output_dict[key]=scaler.transform(value)
            
            scaler_filename = os.path.join(dirpath, "scaler_dataset_robot" + ".pickle")
            with open(scaler_filename, "wb") as output_file:
                pickle.dump(value, output_file)

            if self.cst["save_dataset"]:
                self.save_dataset(tag="after_processing")

        if self.cst["convert_to_tensor"]:
            self.transform_to_tensor(self.output_dict)

    def save_dataset(self, tag=""):

        if isinstance(self.output_dict['train'],np.ndarray):
            for key,value in self.output_dict.items():
                with open(os.path.join(self.cst['processed_data_path'], tag + '_' + key +  '.npy'), 'wb') as f:                 
                    np.save(f, value)

        elif isinstance(self.output_dict['train'],torch.Tensor):
            print('Not yet implemented')

        else:
            print('Data type do not follow expected input')

    def return_data_(self):
        return self.output_dict

    def visualize_data_random_seed(self, tag='train'):
        seed=random.randint(0,len(self.output_dict[tag]))
        print('Value of selected seed for plotting:', seed)

        sample = np.linspace(0,len(self.output_dict[tag][seed,:,0]),len(self.output_dict[tag][seed,:,0]))
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].plot(sample, self.output_dict[tag][seed,:,0])
        axs[0, 1].plot(sample, self.output_dict[tag][seed,:,3])
        axs[1, 0].plot(sample, self.output_dict[tag][seed,:,1])
        axs[1, 1].plot(sample, self.output_dict[tag][seed,:,4])
        axs[2, 0].plot(sample, self.output_dict[tag][seed,:,2])
        axs[2, 1].plot(sample, self.output_dict[tag][seed,:,5])

        axs[0, 0].title.set_text('Linear Velocity X-Axis')
        axs[0, 1].title.set_text('Angular Velocity X-Axis')
        axs[1, 0].title.set_text('Linear Velocity Y-Axis')
        axs[1, 1].title.set_text('Angular Velocity Y-Axis')
        axs[2, 0].title.set_text('Linear Velocity Z-Axis')
        axs[2, 1].title.set_text('Angular Velocity Z-Axis')

        for n,ax in enumerate(axs.flat):
            if (n % 2 == 0):
                ax.set(ylabel='Velocity (m/s)', xlabel='Time (s)')
            else:
                ax.set(ylabel='Angular Velocity (rad/s)', xlabel='Time (s)')

        fig.suptitle('Robot Architecture Input Twist', fontsize=20)
        plt.show()
    
    def start_preprocessing(self, tag=None):
        self.load_files()
        self.preprocess_data()
        self.postprocess_data()

        if (tag is not None) and (self.cst["visualize_data"]):
            self.visualize_data_random_seed(tag) 

        if self.cst["save_dataset"]:
            self.save_dataset()