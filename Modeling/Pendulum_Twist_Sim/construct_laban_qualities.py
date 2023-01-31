import os 
import pandas as pd 
import numpy as np 
from scipy import integrate, signal
from .utilities_data_proc import calc_expressive_qualities

class Laban_Dict():
    def __init__(self, data) -> None:
        self.input_data = data

    def derivate_tensor(self,input_tensor, dx=0.00001):

        x = np.linspace(0, np.shape(input_tensor)[0]/100000, np.shape(input_tensor)[0])
        # print(x[1]-x[0])
        # print(np.shape(input_tensor)[0]/100000)
        output_arr = []
        for cord in range(np.shape(input_tensor)[1]):
            diff_arr = np.diff(input_tensor[:,cord],n=1)/np.diff(x)
            diff_arr = np.insert(diff_arr,-1,diff_arr[-1])
            output_arr.append(diff_arr)
        return np.asarray(output_arr).transpose(1,0)

    def integrate_tensor(self,input_tensor, dx=0.000016):
        output_arr = []
        x = np.linspace(0, np.shape(input_tensor)[0]/100000, np.shape(input_tensor)[0])
        for cord in range(np.shape(input_tensor)[1]):
            integration_arr = integrate.cumulative_trapezoid(input_tensor[:,cord], x)
            integration_arr = signal.detrend(integration_arr)
            integration_arr = np.insert(integration_arr,-1,integration_arr[-1])
            output_arr.append(integration_arr)
        return np.asarray(output_arr).transpose(1,0)

    def construct_signals_dict_vel(self, input_tensor, output_dict, dx= 0.000001):
        
        for key in output_dict.keys():
            if key == "acc":
                output_dict[key] = self.derivate_tensor(input_tensor[:,:3])
            elif key == "pos":
                output_dict[key] = self.integrate_tensor(input_tensor[:,:3],dx=dx)
            elif key == "vel":
                output_dict[key] = input_tensor[:,:3]
            else:
                output_dict[key] = self.derivate_tensor(self.derivate_tensor(input_tensor[:,:3]))
            output_dict[key] = np.expand_dims(output_dict[key],axis=0)

        return output_dict

    def construct_signals_dict_acc(self, input_tensor, output_dict, dx=0.00001):
        for key in output_dict.keys():
            if key == "acc":
                output_dict[key] = input_tensor[:,:3]
            elif key == "pos":
                output_dict[key] = self.integrate_tensor(input_tensor[:,:3],dx=dx)
                output_dict[key] = self.integrate_tensor(output_dict[key], dx=dx)
            elif key == "vel":
                output_dict[key] = self.integrate_tensor(input_tensor[:,:3], dx=dx)
            else:
                output_dict[key] = self.derivate_tensor(input_tensor[:,:3], dx=dx)
            output_dict[key] = np.expand_dims(output_dict[key],axis=0)

        return output_dict

    def start_process(self, name="", human=False, mass=0.4):
        expressive_qualities = []
        if not human:
            dict_gen = self.construct_signals_dict_vel(self.input_data,
                                                    {key:None  for key in ["pos","vel","acc","jerk"]})
        else:
            dict_gen = self.construct_signals_dict_acc(self.input_data,
                                                    {key:None  for key in ["pos","vel","acc","jerk"]})
        expressive_qualities.append(calc_expressive_qualities(dict_gen, alpha=mass))

        # print("check generated")
        df = pd.DataFrame(expressive_qualities)
        print(df.head())
        # df.to_csv(os.path.join("C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling", 
        #                             name+".csv"))

        return df, dict_gen