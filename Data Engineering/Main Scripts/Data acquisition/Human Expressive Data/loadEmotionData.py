import os 
import sys
import re
import copy

import numpy as np
import matplotlib.pyplot as plt
from torch import float64 
from trc import TRCData

import utilities 

class loadEmotionData():
    def __init__(self, constants) -> None:
        self.cst = constants
        self.mocap_data = TRCData()
        self.candidates_dict = {i:{} for i in self.cst['candidate']} 

    def load_rotation(self, filename, label):
        data = np.genfromtxt(os.path.join(self.cst['rotation_path'],filename + '/' + label + '.dat'),
                     skip_header=0,
                     skip_footer=0,
                     names=True,
                     dtype=None,
                     delimiter='\t')
        return np.array(data.tolist())

    def load_dataset(self):
        count=0
        missing_files={}
        missing_files['names'] = []
        for num, file in enumerate(os.listdir(self.cst['dataset_path'])):
            
            try:
                name_oi = re.search('a_(.+?).trc', file).group(1)
            except:
                name_oi = re.search('(.+?).trc', file).group(1)

            aux_dict = {i: None for i in self.cst['trc_data_keys']} 
            if file == 'SALETRE03.4.trc':
                continue

            for i in self.cst['candidate']:
                if i in file:
                    index = i
                    break

            spec_keys = [index+'_'+mark for mark in self.cst['markers_oi']+self.cst['markers_ref']+self.cst["rotation_markers"]]
            req_keys = self.cst['trc_data_keys'] + spec_keys
    
            mocap_data = TRCData()
            try:
                mocap_data.load(os.path.join(self.cst['dataset_path'],file))
                for key in req_keys:
                    aux = key.split('_')
                    if key not in self.cst["rotation_markers"] and (len(aux)<2 or aux[-1][-1]!='d'):
                        aux_dict[key] = copy.deepcopy(mocap_data[key])
                    else: 
                        if list(aux[1])[0] == "L":
                            aux_dict[key] = self.load_rotation(name_oi,"LeftHand")
                        else:
                            aux_dict[key] = self.load_rotation(name_oi,"RightHand")

                if aux_dict['Units'] != 'mm':
                    print('Trajectory unit different than mm, unit is:', aux_dict['Units'])
                    break

                self.candidates_dict[index]['traj_'+str(num)] = copy.deepcopy(aux_dict)
            # print(aux_dict)
            # sys.exit()
            except:
                count+=1
                missing_files['names'].append(name_oi) 
                continue
        
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'missing_files.json')
        utilities.save_json(filename,missing_files)
        del mocap_data, aux_dict

    def get_reference_point(self):
        for val in self.cst['candidate']:
            spec_keys = [val+'_'+mark for mark in self.cst['markers_ref']]
            for traj in self.candidates_dict[val].keys():
                self.candidates_dict[val][traj]['ref_pos'] = (np.array(self.candidates_dict[val][traj][spec_keys[0]])/1000 \
                                                                + np.array(self.candidates_dict[val][traj][spec_keys[1]])/1000) / 2 

    def skew(self,x):
     if (isinstance(x,np.ndarray) and len(x.shape)>=2):
         return np.array([[0., -x[2][0], x[1][0]],
                          [x[2][0], 0., -x[0][0]],
                          [-x[1][0], x[0][0], 0.]])
     else:
         return np.array([[0., -x[2], x[1]],
                          [x[2], 0., -x[0]],
                          [-x[1], x[0], 0.]])

    def calculate_rot_matrix(self, a):
        b = np.repeat(np.reshape(np.array([1,0,0]),(1,3)), np.shape(a)[0],axis=0)
        I = np.tile(np.identity(3), (np.shape(a)[0],1)).reshape(np.shape(a)[0],3,3)
        R = np.empty(np.shape(I))
        dot = np.einsum('ij,ij->i',b,a)
        # dot = 1/(1+dot)
        for i in range(len(R)):
            aux = a[i] / np.linalg.norm(a[i])
            cross = np.cross(b[i],aux)
            skew = self.skew(cross)
            R[i] = I[i] + skew + np.linalg.matrix_power(skew,2) * ((1 - dot[i])/np.linalg.norm(cross)**2)

        return R

    def get_wrist_rot(self):
        for val in self.cst['candidate']:
            spec_keys = [val+'_'+mark for mark in self.cst['markers_oi']]
            for traj in self.candidates_dict[val].keys():
                self.candidates_dict[val][traj]['wrist_rot_LW'] = (np.array(self.candidates_dict[val][traj][spec_keys[0]]) \
                                                                - np.array(self.candidates_dict[val][traj][spec_keys[1]]))
                self.candidates_dict[val][traj]['wrist_rot_RW'] = (np.array(self.candidates_dict[val][traj][spec_keys[3]]) \
                                                                - np.array(self.candidates_dict[val][traj][spec_keys[2]]))
                self.candidates_dict[val][traj]['wrist_rot_RW'] = self.calculate_rot_matrix(self.candidates_dict[val][traj]['wrist_rot_RW'])
                self.candidates_dict[val][traj]['wrist_rot_LW'] = self.calculate_rot_matrix(self.candidates_dict[val][traj]['wrist_rot_LW'])

    def preprocess_mocap_data(self):

        self.get_reference_point()
        # self.get_wrist_rot()
        for val in self.cst['candidate']:
            spec_keys = [val+'_'+mark for mark in self.cst['markers_oi']]
            for traj in self.candidates_dict[val].keys():
                for i in spec_keys:
                    self.candidates_dict[val][traj][i] = np.array(self.candidates_dict[val][traj][i])/1000
                    self.candidates_dict[val][traj][i] -= self.candidates_dict[val][traj]['ref_pos']
                    self.candidates_dict[val][traj][i] -= self.candidates_dict[val][traj][i][0]
                    self.candidates_dict[val][traj][i] = utilities.filter_signal(self.candidates_dict[val][traj][i])

                self.candidates_dict[val][traj][val+'_'+self.cst['new_markers'][0]] = (self.candidates_dict[val][traj][val+'_'+self.cst['markers_oi'][0]] \
                                                                + self.candidates_dict[val][traj][val+'_'+self.cst['markers_oi'][1]]) / 2
                self.candidates_dict[val][traj][val+'_'+self.cst['new_markers'][1]] = (self.candidates_dict[val][traj][val+'_'+self.cst['markers_oi'][2]] \
                                                                + self.candidates_dict[val][traj][val+'_'+self.cst['markers_oi'][3]]) / 2

                self.candidates_dict[val][traj]['wrist_rot_RW'] = self.candidates_dict[val][traj][val+'_'+"LeftHand"]
                self.candidates_dict[val][traj]['wrist_rot_LW'] = self.candidates_dict[val][traj][val+'_'+"RightHand"]
        # self.get_wrist_rot()

    def plot_mocap_data(self, candidate_val='NABA', traj_val='traj_77', marker='NABA_RWRA'):
        array_oi = self.candidates_dict[candidate_val][traj_val][marker]
        print(np.shape(array_oi))
        array_oi_size = np.shape(array_oi)

        fig, axs = plt.subplots(1, array_oi_size[1])
        fig.suptitle('Mocap Data Position')

        map_dict = {0:'X', 1:'Y', 2:'Z'}
        time = np.linspace(0,array_oi_size[0],array_oi_size[0])
        for num, axes in enumerate(axs.flat):
            axes.plot(time, array_oi[:,num])
            axes.set_title(map_dict[num] + '- Axis')
            axes.set(xlabel='Sample number', ylabel='Position in meters(m)')

        plt.show()

    def start_main_process(self):
        self.load_dataset()
        self.preprocess_mocap_data()

    def return_data(self):

        return self.candidates_dict