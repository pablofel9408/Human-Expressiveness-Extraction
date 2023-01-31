import io
import torch 
import torch.linalg as LA
import numpy as np
from scipy import integrate
from scipy import signal
import matplotlib.pyplot as plt
import scipy.linalg as linalg

import random

import utilities

class Evaluation_Methods():
    def __init__(self) -> None:
        
        # self.cst = constants
        self.dx = 0.012

        self.input_robot = None
        self.input_human = None

        self.generated_robot = None

        self.human_latent = None
        self.robot_latent = None
        self.generated_human_latent = None
        self.generated_robot_latent = None

        self.basic_signals_human = {i:None for i in ["pos","vel","acc","jerk"]}
        self.basic_signals_robot = {i:None for i in ["pos","vel","acc","jerk"]}
        self.basic_signals_robot_hat = {i:None for i in ["pos","vel","acc","jerk"]}

        self.expressive_dict = {i:None for i in ["robot","human","robot_generated"]}

        self.expressive_qualities = ["weight","time","flow","space","bound_vol"]
        self.expressivity_features_diff_stats = {i:None for i in self.expressive_qualities}

    def set_inputs(self, inputs:torch.tensor, outputs:torch.tensor, latent_features:torch.tensor):

        self.input_robot = inputs[0].detach().cpu().numpy()
        self.input_human = inputs[1].detach().cpu().numpy()

        self.generated_robot = outputs.detach().cpu().numpy()

        self.human_latent = latent_features[0].detach().cpu().numpy()
        self.robot_latent = latent_features[1].detach().cpu().numpy()
        self.generated_human_latent = latent_features[2].detach().cpu().numpy()
        self.generated_robot_latent = latent_features[3].detach().cpu().numpy()

    def main_evaluation(self, visualize=False):

        self.basic_signals_human = self.construct_signals_dict(self.input_human,
                                                                self.basic_signals_human)
        self.basic_signals_robot = self.construct_signals_dict(self.input_robot,
                                                                 self.basic_signals_robot)
        self.basic_signals_robot_hat = self.construct_signals_dict(self.generated_robot,
                                                                 self.basic_signals_robot_hat)

        self.expressive_dict["human"] = self.expressive_evaluation(self.basic_signals_human)
        self.expressive_dict["robot"] = self.expressive_evaluation(self.basic_signals_robot)
        self.expressive_dict["robot_generated"] = self.expressive_evaluation(self.basic_signals_robot_hat)

        human_features_dis = self.get_distance_per_feature(self.human_latent, self.generated_human_latent)
        robot_features_dis = self.get_distance_per_feature(self.robot_latent, self.generated_robot_latent)

        if visualize:
            for qualitie in self.expressive_qualities:
                self.visualize_qualities([value for value in self.expressive_dict.values()], qualitie)
            plt.show()

            self.visualize_stats([value["stats"] for value in self.expressive_dict.values()], "mean")
            plt.show()

            self.visualize_stats([value["stats"] for value in self.expressive_dict.values()], "variance")
            plt.show()

        return self.expressive_dict, (human_features_dis,robot_features_dis)

    def visualize_stats(self, input_data, qualitie):

        fig, ax = plt.subplots()
        x_axis_values =input_data[1]["mean"].keys()
        l1 = ax.bar(x_axis_values, input_data[0][qualitie].values(), alpha=0.5)
        l2 = ax.bar(x_axis_values, input_data[1][qualitie].values(), alpha=0.5)
        l3 = ax.bar(x_axis_values, input_data[2][qualitie].values(), alpha=0.5)
        ax.legend((l1,l2,l3), ("human", "robot", "generated"))
        ax.set_title("Bar plot Qualities Stats - " + qualitie)
        ax.set_xlabel("Expressive Feature")
        ax.set_ylabel(qualitie)
        plt.legend(loc="upper right")

    def get_distance_per_feature(self, latent1, latent2):

        output_res = []
        for i in range(np.shape(latent1)[1]):
            mu1, sigma1 = self.calculate_activation_statistics(latent1[:,i,:])
            mu2, sigma2 = self.calculate_activation_statistics(latent2[:,i,:])
            output_res.append(self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2))

        return output_res

    def calculate_activation_statistics(self, act):
        """Calculates the statistics used by FID
        Args:
            images: torch.tensor, shape: (N, 3, H, W), dtype: torch.float32 in range 0 - 1
            batch_size: batch size to use to calculate inception scores
        Returns:
            mu:     mean over all activations from the last pool layer of the inception model
            sigma:  covariance matrix over all activations from the last pool layer 
                    of the inception model.
        """
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    # Modified from: https://github.com/bioinf-jku/TTUR/blob/master/fid.py
    def calculate_frechet_distance(self,mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
                
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1 : Numpy array containing the activations of the pool_3 layer of the
                inception net ( like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                on an representive data set.
        -- sigma1: The covariance matrix over activations of the pool_3 layer for
                generated samples.
        -- sigma2: The covariance matrix over activations of the pool_3 layer,
                precalcualted on an representive data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2
        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-1):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    def visualize_qualities(self, input_data, qualitie):

        bars = []
        keys = ["human", "robot", "generated"]
        fig, ax = plt.subplots()
        # for n,data_oi in enumerate(input_data):
        timeAxis = np.linspace(0, len(input_data[2][qualitie]),len(input_data[2][qualitie]))
        l1 = ax.bar(timeAxis,input_data[0][qualitie], alpha=0.5)
        l2 = ax.bar(timeAxis,input_data[1][qualitie], alpha=0.5)
        l3 = ax.bar(timeAxis,input_data[2][qualitie], alpha=0.5)
        # labels = [l.get_label() for l in bars
        ax.legend((l1,l2,l3), ("human", "robot", "generated"))
        ax.set_title("Bar plot Qualities Comparission On Generation - " + qualitie)
        ax.set_xlabel("Dataset Index")
        ax.set_ylabel(qualitie)
        # plt.legend(loc="upper right")

    def visualize_qualities_stream(self, input_data, qualitie):

        bars = []
        keys = ["human", "robot", "generated"]
        fig, ax = plt.subplots()
        # for n,data_oi in enumerate(input_data):
        timeAxis = np.linspace(0, len(input_data[2][qualitie]),len(input_data[2][qualitie]))
        l1 = ax.bar(timeAxis,input_data[0][qualitie], alpha=0.5)
        l2 = ax.bar(timeAxis,input_data[1][qualitie], alpha=0.5)
        l3 = ax.bar(timeAxis,input_data[2][qualitie], alpha=0.5)
        # labels = [l.get_label() for l in bars
        ax.legend((l1,l2,l3), ("human", "robot", "generated"))
        ax.set_title("Bar plot Qualities Comparission On Generation - " + qualitie)
        ax.set_xlabel("Dataset Index")
        ax.set_ylabel(qualitie)
        # plt.legend(loc="upper right")

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        plt.close()

        return buf

    def construct_signals_dict(self, input_tensor, output_dict):
        
        for key in output_dict.keys():
            if key == "acc":
                output_dict[key] = self.derivate_tensor(input_tensor[:,:,:3])
            elif key == "pos":
                output_dict[key] = self.integrate_tensor(input_tensor[:,:,:3])
            elif key == "vel":
                output_dict[key] = input_tensor[:,:,:3]
            else:
                output_dict[key] = self.derivate_tensor(self.derivate_tensor(input_tensor[:,:,:3]))

        # self.plot_signals(output_dict)

        return output_dict

    def plot_signals(self, signals_dict, index_oi=32):
        index_oi= random.randint(0,len(signals_dict["acc"])-1)
        sample = np.linspace(0,len(signals_dict["acc"][index_oi,:,0]),len(signals_dict["acc"][index_oi,:,0]))
        fig, axs = plt.subplots(3, 4)
        axs[0, 0].plot(sample,signals_dict["jerk"][index_oi,:,0])
        axs[0, 1].plot(sample, signals_dict["acc"][index_oi,:,0])
        axs[0, 2].plot(sample, signals_dict["vel"][index_oi,:,0])
        axs[0, 3].plot(sample, signals_dict["pos"][index_oi,:,0])

        axs[1, 0].plot(sample,signals_dict["jerk"][index_oi,:,1])
        axs[1, 1].plot(sample, signals_dict["acc"][index_oi,:,1])
        axs[1, 2].plot(sample, signals_dict["vel"][index_oi,:,1])
        axs[1, 3].plot(sample, signals_dict["pos"][index_oi,:,1])

        axs[2, 0].plot(sample,signals_dict["jerk"][index_oi,:,2])
        axs[2, 1].plot(sample, signals_dict["acc"][index_oi,:,2])
        axs[2, 2].plot(sample, signals_dict["vel"][index_oi,:,2])
        axs[2, 3].plot(sample, signals_dict["pos"][index_oi,:,2])

    def derivate_tensor(self,input_tensor):
        output_arr = []
        for arr in input_tensor:
            cord_arr = []
            for cord in range(np.shape(arr)[1]):
                diff_arr = np.diff(arr[:,cord],n=1)/self.dx
                diff_arr = np.insert(diff_arr,-1,diff_arr[-1])
                cord_arr.append(diff_arr)
            output_arr.append(cord_arr)
        return np.asarray(output_arr).transpose(0,2,1)
    
    def integrate_tensor(self,input_tensor):
        output_arr = []
        for arr in input_tensor:
            coord_arr = []
            for cord in range(np.shape(arr)[1]):
                integration_arr = integrate.cumulative_trapezoid(arr[:,cord],dx=self.dx)
                integration_arr = signal.detrend(integration_arr)
                integration_arr = np.insert(integration_arr,-1,integration_arr[-1])
                coord_arr.append(integration_arr)
            output_arr.append(coord_arr)
        return np.asarray(output_arr).transpose(0,2,1)

    def expressive_evaluation(self, output_dict):
        
        qualities_dict = {i:None for i in self.expressive_qualities}
        qualities_dict["weight"] = self.calc_laban_weight(output_dict["vel"])
        qualities_dict["time"] = self.calc_laban_time(output_dict["acc"])
        qualities_dict["flow"] = self.calc_laban_flow(output_dict["jerk"])
        qualities_dict["space"] = self.calc_laban_space(output_dict["pos"])
        qualities_dict["bound_vol"] = self.calc_vol_bound_box(output_dict["pos"])

        qualities_dict["stats"] = {"mean": {label: np.mean(arr) for label, arr in qualities_dict.items()}}
        qualities_dict["stats"]["variance"] = {label: np.var(qualities_dict[label]) for label in self.expressive_qualities}
        # output_qualities = np.stack((weight,time,flow,space,bound_vol),axis=1)

        return qualities_dict

    def calc_laban_weight(self,input_sign, time_inter=10.0): 
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

        # weight = torch.cat((weight_tot,torch.unsqueeze(weight_sum,dim=1)),dim=1)
        weight = np.array(weight_sum)
        return weight

    def calc_laban_time(self,input_sign, time_inter=10.0): 
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
        
        laban_time = np.array(laban_time_sum)
        # laban_time = np.concatenate((laban_time_tot,np.expand_dims(laban_time_sum,axis=1)),axis=1)

        return laban_time

    def calc_laban_flow(self,input_sign, time_inter=10.0): 
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
        
        laban_flow = np.array(laban_flow_sum)
        # laban_flow = np.concatenate((laban_flow_tot,np.expand_dims(laban_flow_sum,axis=1)),axis=1)

        return laban_flow

    def calc_laban_space(self,input_sign, time_inter=10.0): 
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

        laban_space = np.array(laban_space_sum)
        # laban_space = np.concatenate((laban_space_tot,np.expand_dims(laban_space_sum,axis=1)),axis=1)

        return laban_space

    def calc_vol_bound_box(self,input_sign, time_inter=10.0): 
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

        bound_vol = np.array(bound_vol_sum)
        return bound_vol

