import os 
import pandas as pd 
import numpy as np 
import random

import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,Isomap
from sklearn import manifold

import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

from Robot_Feature_Extraction.Modeling.model_VAE_Laban_reg import NVAE_LabReg
import utilities

class Latent_Viz():
    def __init__(self) -> None:
        self.emotion_data = None

    def load_emotional_data(self, tag="train"):
        self.emotion_data = pd.read_csv("C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Translation_Dataset_Analysis\\Data\\emotion_dataset_"+ tag+ ".csv")
        self.emotion_data = self.emotion_data.drop(['Unnamed: 0'], axis=1)

    def preprocess_data(self, input_data, dataset_tags):
        shape = np.shape(input_data)
        # input_data = np.array([features_2d.flatten() for features_2d in input_data])
        input_data = input_data.reshape(shape[0],shape[1]*shape[2])
        print(np.shape(input_data))
        input_df = pd.DataFrame(input_data)
        # input_df = pd.concat([input_df, self.emotion_data], ignore_index=True, axis=1) 
        emotion_data = pd.DataFrame(dataset_tags)
        print(emotion_data.shape)
        print(emotion_data.head())
        input_df = pd.concat([input_df, emotion_data], ignore_index=True, axis=1)
        input_df.columns = [i for i in range(0,shape[1]*shape[2])] + ["emo", "act"]

        return input_df

    def generate_translation_latent(self,model,input_dataset_human,input_dataset_robot):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dataset_h = input_dataset_human.to(dev)
        dataset_r = input_dataset_robot.to(dev)
        dataset_r = dataset_r[random.randint(0,dataset_r.size()[0])].unsqueeze(0).repeat(dataset_h.size()[0],1,1)
        # dataset_r = torch.zeros_like(dataset_r).repeat(dataset_h.size()[0],1,1)
        dataset_h = dataset_h.permute(0,2,1)
        dataset_r = dataset_r.permute(0,2,1)
        out_hat_twist, z_human, z_robot, mu, log_var, \
            z_human_hat, z_robot_hat, mu_hat,log_var_hat, \
                cont_att, loc_att_human, loc_att_robot = model(dataset_r.float(), dataset_h.float())
        gen_trajectories = out_hat_twist.cpu().numpy()

        fig5, axs5 = plt.subplots(6, 3)

        dataset_r = dataset_r[0].clone().cpu().numpy()
        dataset_h = dataset_h[0].clone().cpu().numpy()
        out_hat_twist = out_hat_twist.detach().clone().cpu().numpy()
        for i in range(np.shape(out_hat_twist)[0]):
            out_hat_twist[i] = utilities.filter_signal_2(gen_trajectories[i])
        sample = np.linspace(0,len(dataset_h[0,:]),len(dataset_h[0,:]))
        axs5[0, 0].plot(sample, dataset_r[0,:],linewidth=2.0)
        axs5[0, 1].plot(sample, dataset_h[0,:],linewidth=2.0)
        axs5[0, 2].plot(sample, out_hat_twist[0,0,:],linewidth=2.0)

        axs5[1, 0].plot(sample, dataset_r[1,:],linewidth=2.0)
        axs5[1, 1].plot(sample, dataset_h[1,:],linewidth=2.0)
        axs5[1, 2].plot(sample, out_hat_twist[0,1,:],linewidth=2.0)

        axs5[2, 0].plot(sample, dataset_r[2,:],linewidth=2.0)
        axs5[2, 1].plot(sample, dataset_h[2,:],linewidth=2.0)
        axs5[2, 2].plot(sample, out_hat_twist[0,2,:],linewidth=2.0)

        axs5[3, 0].plot(sample, dataset_r[3,:],linewidth=2.0)
        axs5[3, 1].plot(sample, dataset_h[3,:],linewidth=2.0)
        axs5[3, 2].plot(sample, out_hat_twist[0,3,:],linewidth=2.0)

        axs5[4, 0].plot(sample, dataset_r[4,:],linewidth=2.0)
        axs5[4, 1].plot(sample, dataset_h[4,:],linewidth=2.0)
        axs5[4, 2].plot(sample, out_hat_twist[0,4,:],linewidth=2.0)

        axs5[5, 0].plot(sample, dataset_r[5,:],linewidth=2.0)
        axs5[5, 1].plot(sample, dataset_h[5,:],linewidth=2.0)
        axs5[5, 2].plot(sample, out_hat_twist[0,5,:],linewidth=2.0)

        plt.show()

        return np.transpose(out_hat_twist,(0,2,1))


    def generate_translation_latent_neutral_style(self,model,input_dataset_human,
                                                        input_dataset_robot, dataset_neutral,
                                                        dataset_tags):
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        data_neutral_style = np.asarray([dataset_neutral[act] for act in dataset_tags])
        data_neutral_style = torch.tensor(data_neutral_style).to(dev)
        dataset_h = torch.tensor(input_dataset_human).to(dev)
        dataset_r = input_dataset_robot.to(dev)
        dataset_r = dataset_r[random.randint(0,dataset_r.size()[0])].unsqueeze(0).repeat(dataset_h.size()[0],1,1)
        # dataset_r = torch.zeros_like(dataset_r).repeat(dataset_h.size()[0],1,1)
        dataset_h = dataset_h.permute(0,2,1)
        dataset_r = dataset_r.permute(0,2,1)
        out_hat_twist, z_human, z_robot, mu, log_var, \
            z_human_hat, z_robot_hat, mu_hat,log_var_hat, \
                cont_att, loc_att_human, loc_att_robot = model(dataset_r.float(), dataset_h.float(),
                                                                data_neutral_style)
        gen_trajectories = out_hat_twist.cpu().numpy()

        fig5, axs5 = plt.subplots(6, 3)

        dataset_r = dataset_r[0].clone().cpu().numpy()
        dataset_h = dataset_h[0].clone().cpu().numpy()
        out_hat_twist = out_hat_twist.detach().clone().cpu().numpy()
        for i in range(np.shape(out_hat_twist)[0]):
            out_hat_twist[i] = utilities.filter_signal_2(gen_trajectories[i])
        sample = np.linspace(0,len(dataset_h[0,:]),len(dataset_h[0,:]))
        axs5[0, 0].plot(sample, dataset_r[0,:],linewidth=2.0)
        axs5[0, 1].plot(sample, dataset_h[0,:],linewidth=2.0)
        axs5[0, 2].plot(sample, out_hat_twist[0,0,:],linewidth=2.0)

        axs5[1, 0].plot(sample, dataset_r[1,:],linewidth=2.0)
        axs5[1, 1].plot(sample, dataset_h[1,:],linewidth=2.0)
        axs5[1, 2].plot(sample, out_hat_twist[0,1,:],linewidth=2.0)

        axs5[2, 0].plot(sample, dataset_r[2,:],linewidth=2.0)
        axs5[2, 1].plot(sample, dataset_h[2,:],linewidth=2.0)
        axs5[2, 2].plot(sample, out_hat_twist[0,2,:],linewidth=2.0)

        axs5[3, 0].plot(sample, dataset_r[3,:],linewidth=2.0)
        axs5[3, 1].plot(sample, dataset_h[3,:],linewidth=2.0)
        axs5[3, 2].plot(sample, out_hat_twist[0,3,:],linewidth=2.0)

        axs5[4, 0].plot(sample, dataset_r[4,:],linewidth=2.0)
        axs5[4, 1].plot(sample, dataset_h[4,:],linewidth=2.0)
        axs5[4, 2].plot(sample, out_hat_twist[0,4,:],linewidth=2.0)

        axs5[5, 0].plot(sample, dataset_r[5,:],linewidth=2.0)
        axs5[5, 1].plot(sample, dataset_h[5,:],linewidth=2.0)
        axs5[5, 2].plot(sample, out_hat_twist[0,5,:],linewidth=2.0)

        plt.show()

        return np.transpose(out_hat_twist,(0,2,1))

    def generate_latent(self, dirpath, config, dataset, vae=False):
        
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dataset = torch.tensor(dataset).to(dev)
        dataset = dataset.permute(0,2,1)
        if not vae:
            new_acc = torch.diff(dataset[:,:3,:]) / 0.012
            dataset[:,:3,:] = torch.cat((new_acc,new_acc[:,:,-1].unsqueeze(2)),dim=2) 
        model = NVAE_LabReg(config, dev).to(dev)
        checkpoint = torch.load(dirpath, map_location=torch.device(dev))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        _,mu,log_var = model.encoder(dataset.float())
        z_human = model.calculateOutput(mu, log_var)

        z_human = z_human.detach().clone().cpu().numpy()
        return z_human
    
    def visualization(self, df_subset):
        
        df_subset = df_subset[df_subset["emo"]!="NEE"]
        df_subset["emo"] = np.where(df_subset["emo"]=="TRE","sad",df_subset["emo"])
        df_subset["emo"] = np.where(df_subset["emo"]=="JOE","happy",df_subset["emo"])
        df_subset["emo"] = np.where(df_subset["emo"]=="COE","angry",df_subset["emo"])
        pca_50 = PCA(n_components=2)
        tsne_results = pca_50.fit_transform(df_subset[[i for i in range(0,len(df_subset.columns)-2)]])
        # tsne = TSNE(n_components=2, verbose=1, perplexity=12, n_iter=1000,
        #             init="random")
        # tsne_results = tsne.fit_transform(tsne_results)

        df_subset['tsne-2d-one'] = tsne_results[:,0]
        df_subset['tsne-2d-two'] = tsne_results[:,1]
        # df_subset['tsne-2d-three'] = tsne_results[:,2]

        sns.scatterplot(
                x="tsne-2d-one", y="tsne-2d-two",
                hue=df_subset['emo'],
                data=df_subset,
                legend="full",
                marker='o',
                linewidths=2
            )

        plt.show()

        # for act in df_subset["act"].unique():
        #     print("Curent Actor", act)
        #     print("------------------")
        #     print("PCA")
        #     sns.scatterplot(
        #         x="tsne-2d-one", y="tsne-2d-two",
        #         hue=df_subset[df_subset["act"]==act]['emo'],
        #         data=df_subset[df_subset["act"]==act],
        #         legend="full",
        #         alpha=0.3,
        #         marker='o',
        #         linewidths=2
        #     )

        #     plt.show()

        #     print("------------------")
        #     print("TSNE")
        #     pca_50 = PCA(n_components=50)
        #     tsne_results = pca_50.fit_transform(df_subset[[i for i in range(0,len(df_subset.columns)-4)]])
        #     tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000,
        #                 init="random")
        #     tsne_results = tsne.fit_transform(tsne_results)

        #     df_subset['tsne-2d-one'] = tsne_results[:,0]
        #     df_subset['tsne-2d-two'] = tsne_results[:,1]
        #     # df_subset['tsne-2d-three'] = tsne_results[:,2]

        #     sns.scatterplot(
        #         x="tsne-2d-one", y="tsne-2d-two",
        #         hue=df_subset[df_subset["act"]==act]['emo'],
        #         data=df_subset[df_subset["act"]==act],
        #         legend="full",
        #         alpha=0.3,
        #         marker='o',
        #         linewidths=2
        #     )

        #     plt.show()

        pca_50 = PCA(n_components=50)
        tsne_results = pca_50.fit_transform(df_subset[[i for i in range(0,len(df_subset.columns)-4)]])
        tsne = TSNE(n_components=2, verbose=1, perplexity=15, n_iter=1000)
        tsne_results = tsne.fit_transform(tsne_results)

        df_subset['tsne-first-component'] = tsne_results[:,0]
        df_subset['tsne-second-component'] = tsne_results[:,1]

        ax = plt.axes()
        ax.set_facecolor('white')
        ax.grid(False)

        sns.scatterplot(
                x="tsne-first-component", y="tsne-second-component",
                hue=df_subset['emo'],
                data=df_subset,
                legend="full",
                marker='o',
                linewidths=2
            )
        plt.xlabel('', fontsize=18)
        plt.ylabel('', fontsize=18)
        legend = plt.legend(frameon = 1,fontsize=12)
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('black')
        plt.show()

        # pca_50 = PCA(n_components=100)
        # tsne_results = pca_50.fit_transform(df_subset[[i for i in range(0,len(df_subset.columns)-4)]])
        # tsne = TSNE(n_components=3, verbose=1, perplexity=50, n_iter=1000,
        #             init="random")
        # tsne_results = tsne.fit_transform(tsne_results)

        # df_subset['tsne-2d-one'] = tsne_results[:,0]
        # df_subset['tsne-2d-two'] = tsne_results[:,1]
        # df_subset['tsne-2d-three'] = tsne_results[:,2]

        # fig = plt.figure(figsize=(16,10))
        # ax = Axes3D(fig)
        # # fig.add_axes(ax)

        # # get colormap from seaborn
        # cmap = ListedColormap(sns.color_palette("husl", len(df_subset['emo'].unique())).as_hex())

        # # plot
        # for s in df_subset['emo'].unique():
        #     ax.scatter(df_subset['tsne-2d-one'][df_subset['emo']==s],
        #                 df_subset['tsne-2d-two'][df_subset['emo']==s],
        #                 df_subset['tsne-2d-three'][df_subset['emo']==s],label=s, s=20, marker='o',
        #                 linewidths=2)
            
        # ax.legend()