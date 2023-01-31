import os 
import pandas as pd 
import numpy as np 
from scipy import integrate, signal
import utilities

class Laban_Dict():
    def __init__(self, data) -> None:
        self.input_data = data

    def derivate_tensor(self,input_tensor, dx=0.016):
        output_arr = []
        for cord in range(np.shape(input_tensor)[1]):
            diff_arr = np.diff(input_tensor[:,cord],n=1)/dx
            diff_arr = np.insert(diff_arr,-1,diff_arr[-1])
            output_arr.append(diff_arr)
        return np.asarray(output_arr).transpose(1,0)

    def integrate_tensor(self,input_tensor, dx=0.016):
        output_arr = []
        for cord in range(np.shape(input_tensor)[1]):
            integration_arr = integrate.cumulative_trapezoid(input_tensor[:,cord],dx=dx)
            integration_arr = signal.detrend(integration_arr)
            integration_arr = np.insert(integration_arr,-1,integration_arr[-1])
            output_arr.append(integration_arr)
        return np.asarray(output_arr).transpose(1,0)

    def construct_signals_dict_vel(self, input_tensor, output_dict):
        
        for key in output_dict.keys():
            if key == "acc":
                output_dict[key] = self.derivate_tensor(input_tensor[:,:3])
            elif key == "pos":
                output_dict[key] = self.integrate_tensor(input_tensor[:,:3])
            elif key == "vel":
                output_dict[key] = input_tensor[:,:3]
            else:
                output_dict[key] = self.derivate_tensor(self.derivate_tensor(input_tensor[:,:3]))
            output_dict[key] = np.expand_dims(output_dict[key],axis=0)
        return output_dict

    def construct_signals_dict_acc(self, input_tensor, output_dict):
        dx = 0.012
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

    def start_process(self, name="", human=False):
        expressive_qualities = []
        for i in range(np.shape(self.input_data)[0]):
            if not human:
                dict_gen = self.construct_signals_dict_vel(self.input_data[i],
                                                        {key:None  for key in ["pos","vel","acc","jerk"]})
            else:
                dict_gen = self.construct_signals_dict_acc(self.input_data[i],
                                                        {key:None  for key in ["pos","vel","acc","jerk"]})
            expressive_qualities.append(utilities.calc_expressive_qualities(dict_gen, alpha=1))

        print("check generated")
        df = pd.DataFrame(expressive_qualities)
        print(df.head())
        df.to_csv(os.path.join("C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling", 
                                    name+".csv"))

        return df