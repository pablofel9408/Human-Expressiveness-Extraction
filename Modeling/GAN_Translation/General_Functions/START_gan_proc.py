import os
import re 
import io

import copy
import json
import tqdm
import datetime
import random

from scipy.interpolate import make_interp_spline, BSpline

import pandas as pd
import torch
import torchvision
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchsummary import torchsummary
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from . import loss_functions_variations
from ..Model_Blocks.main_GAN_model import GAN_Translation_Disc, GAN_Translation_Gen
import utilities
from Robot_Feature_Extraction.Modeling.utilitiesClass import utility
from .evaluation_method import Evaluation_Methods
from .dataset_classes import ConcatDataset,CustomDataset,CustomDataset_Hum, CustomDataset_Hum_Style_Neutral

#TODO: add validation function

class ModelingGAN():
    def __init__(self, constants) -> None:
        self.cst = constants
        
        self.data = {}
        self.raw_data = {}
        
        self.data_batches_robot = {}
        self.data_batches_human = {}
        self.data_batches = {}
        self.data_batches_neutral_style = {}

        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.task_name = self.cst["task_name"] + " " + self.cst["modelKey"] 

        if not self.cst["pretrained"]:
            log_dir = "logs/" + self.task_name + " logs" + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.writer = SummaryWriter(log_dir=log_dir)
            
        self.utils = utility(self.dev)
        print("Using {} device".format(self.dev))

        self.evaluation_method = Evaluation_Methods()

    def set_input_data(self, input_data, expressive=False):
        if not expressive:
            self.data_rob = input_data[0]
            self.data_human = input_data[1]
        else:
            self.data_rob = input_data[0]
            self.dataset_tags = input_data[2]
            input_data = input_data[1]
            signals_input_data = {i:None for i in input_data.keys()}
            self.qualities_input_data = {i:None for i in input_data.keys()}
            for key,value in input_data.items():
                signals_input_data[key] = value[0]
                self.qualities_input_data[key] = value[1]
            self.data_human = signals_input_data
        
        if self.cst["neutral_style"]:
            self.set_input_data_neutral()

        self.set_batches()
        print('------loading data done')

    def set_input_data_neutral(self):
        
        files = os.listdir(self.cst["netural_data_path_mean"])
        set_tags = set(val for val in self.dataset_tags.keys() if val not in ["emotion", "actor"])
        set_participants = set(val for val in self.dataset_tags["train"]["act"])
        self.neutral_data = {i:{j:None} for j in set_participants for i in set_tags}
        for file in files:
            file_name_tags = re.split('_|\.', file)
            self.neutral_data[file_name_tags[0]][file_name_tags[3]] = np.load(os.path.join(self.cst["netural_data_path_mean"],file))
    
    def return_neutral_data(self):
        return self.neutral_data

    def seed_worker(self,worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def set_batches(self):
        # Create data loaders.
        g = torch.Generator()
        g.manual_seed(0)
        for key in self.data_rob.keys():
            if not isinstance(self.data_rob[key], torch.Tensor) or not isinstance(self.data_human[key], torch.Tensor):
                self.data_rob[key] = torch.tensor(self.data_rob[key])
                self.data_human[key] = torch.tensor(self.data_human[key])

            self.data_batches_robot[key] = DataLoader(CustomDataset(self.data_rob[key]), batch_size=self.cst["batch_size"],
                                        worker_init_fn=self.seed_worker,generator=g,shuffle=False)
            self.data_batches_human[key] = DataLoader(CustomDataset(self.data_human[key]), batch_size=self.cst["batch_size"],
                                        worker_init_fn=self.seed_worker,generator=g,shuffle=True)

            self.data_batches[key] = DataLoader(ConcatDataset(CustomDataset(self.data_rob[key]),CustomDataset_Hum(self.data_human[key], self.data_rob[key].size(0))), 
                                        batch_size=self.cst["batch_size"],
                                        worker_init_fn=self.seed_worker,generator=g,shuffle=False)


            if self.cst["neutral_style"]:
                self.data_batches_neutral_style[key] = DataLoader(ConcatDataset(CustomDataset(self.data_rob[key]),
                                                                    CustomDataset_Hum_Style_Neutral(self.data_human[key],self.neutral_data[key], 
                                                                                                    self.dataset_tags[key]["act"],self.data_rob[key].size(0))), 
                                                                    batch_size=self.cst["batch_size"],
                                                                    worker_init_fn=self.seed_worker,generator=g,shuffle=False)
                
        print('------batches done')

    def getModel(self,config):
        
        disc_netw = GAN_Translation_Disc(config=config["discriminator"]).to(self.dev)
        gen_netw = GAN_Translation_Gen(config=config["generator"],device=self.dev).to(self.dev)
        # if not self.cst["pretrained"]:
        self.load_from_path(self.cst["pretrained_model_path_Human_VAE"],gen_netw.VAE_human)
        self.load_from_path(self.cst["pretrained_model_path_Robot_VAE"],gen_netw.VAE_robot)

        gen_netw.VAE_human.requires_grad_(False)
        gen_netw.VAE_robot.requires_grad_(False)

        return disc_netw, gen_netw
    
    def load_from_path(self,dirpath,model):

        checkpoint = torch.load(dirpath, map_location=torch.device(self.dev))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    def add_tensorboard_weight(self, layer, epoch, name="", bias=False):

        if bias:
            self.writer.add_histogram(name+'.bias', layer.bias, epoch)

        self.writer.add_histogram(name, layer, epoch)

        try:
            self.writer.add_histogram(
                name + '.grad'
                ,layer.grad
                ,epoch
            )
        except:
            pass

    def visualize_data(self, original_tensor, original_tensor_hum, pred_tensor, epoch, index=0, tag="first"):
        """Create a pyplot plot and save to buffer."""

        if len(np.shape(original_tensor))<3:
            original_tensor = np.expand_dims(original_tensor,axis=0)
        
        if len(np.shape(pred_tensor))<3:
            pred_tensor = np.expand_dims(pred_tensor,axis=0)

        featureDimLenght = np.shape(original_tensor)[1]
        timeTrain = np.linspace(0,len(original_tensor[0,0,:]),len(original_tensor[0,0,:]))
        timeTest = np.linspace(0,len(pred_tensor[0,0,:]),len(pred_tensor[0,0,:]))
        fig, axes = plt.subplots(3,featureDimLenght, sharex=True)

        axesColors = ['tab:blue','tab:orange', 'tab:purple', 'tab:green', 'tab:red', 'tab:pink']
        axesTitles = ['V X', 'V Y', 'V Z', 'AV X', 'AV Y', 'AV Z']

        for n,row in enumerate(axes.flat):
            if n < featureDimLenght:
                row.set_title(axesTitles[n])
                row.plot(timeTrain, original_tensor[index,n,:], axesColors[n])
                row.set_ylabel('Robot Original Signals')
            elif n < 2*featureDimLenght:
                row.set_title(axesTitles[n - featureDimLenght])
                row.plot(timeTest,original_tensor_hum[index,n-featureDimLenght,:], 
                            axesColors[n-featureDimLenght])
                row.set_ylabel('Human Original Signals')
            else:
                row.set_title(axesTitles[n - (2*featureDimLenght)])
                row.plot(timeTest,pred_tensor[index,n-(2*featureDimLenght),:], 
                            axesColors[n-(2*featureDimLenght)])
                row.set_ylabel('Network Output')

        fig.suptitle(f"Val Input Signals Reconstruction Comparisson, Index: {index}, Epoch: {epoch}" + tag)

        #determine axes and their limits 
        ax_selec = [(ax, ax.get_ylim()) for ax in axes.flat]

        #find maximum y-limit spread
        max_delta = max([lmax-lmin for _, (lmin, lmax) in ax_selec])

        #expand limits of all subplots according to maximum spread
        for ax, (lmin, lmax) in ax_selec:
            ax.set_ylim(lmin-(max_delta-(lmax-lmin))/2, lmax+(max_delta-(lmax-lmin))/2)

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        plt.close()

        return buf

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def save_images(self, input_tensor, featureDimLenght=4):

        input_tensor = [i[0,:,:].cpu().detach().numpy() for i in input_tensor if torch.is_tensor(i)]
        axesTitles = ['Similarity Matrix', 'Co-attention Output', 'Self-attention Robot', 'Self-attention Human']
        fig, axes = plt.subplots(1,len(input_tensor), sharex=False)
        for n,row in enumerate(axes.flat):
            row.imshow(input_tensor[n])
            # row.set_title(axesTitles[n])

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        plt.close()

        return buf
    
    def training_loop_gan(self):
        """
        Trainning module for a GAN architecture. 
        Based on tutorial from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        """
        # self.set_deterministic_env()
        disc_model, gen_model = self.getModel(self.cst["model_config"])
        disc_model_R = copy.deepcopy(disc_model)

        if self.cst["optimizer"]=="adam":
            print("adam")
            optimizerG = torch.optim.Adam(gen_model.parameters(), lr=self.cst["h_param"]["generator"]['learning_rate'], 
                                            weight_decay=self.cst["h_param"]["generator"]["L2regularizer"], amsgrad=False, 
                                            betas=(self.cst["h_param"]["generator"]["beta1"], 0.98),eps=1e-8)
            optimizerD = torch.optim.Adam(disc_model.parameters(), lr=self.cst["h_param"]["discriminator"]['learning_rate'], 
                                            weight_decay=self.cst["h_param"]["discriminator"]["L2regularizer"], amsgrad=False,
                                            betas=(self.cst["h_param"]["discriminator"]["beta1"], 0.98),eps=1e-8)
            optimizerD_R = torch.optim.AdamW(disc_model_R.parameters(), lr=self.cst["h_param"]["discriminator"]['learning_rate'], 
                                            weight_decay=self.cst["h_param"]["discriminator"]["L2regularizer"], amsgrad=False,
                                            betas=(self.cst["h_param"]["discriminator"]["beta1"], 0.98),eps=1e-8)
            # optimizerG = torch.optim.RMSprop(gen_model.parameters(), lr=self.cst["h_param"]["generator"]['learning_rate'], 
            #                                 weight_decay=self.cst["h_param"]["generator"]["L2regularizer"], momentum=0.9)
            # optimizerD = torch.optim.RMSprop(disc_model.parameters(), lr=self.cst["h_param"]["discriminator"]['learning_rate'], 
            #                                 weight_decay=self.cst["h_param"]["discriminator"]["L2regularizer"], momentum=0.9)
        else:
            optimizerG = torch.optim.SGD(gen_model.parameters(), lr=self.cst["h_param"]["generator"]['learning_rate'], 
                                            weight_decay=self.cst["h_param"]["generator"]["L2regularizer"],momentum=0.9,)
            optimizerD = torch.optim.SGD(disc_model.parameters(), lr=self.cst["h_param"]["discriminator"]['learning_rate'], 
                                            weight_decay=self.cst["h_param"]["discriminator"]["L2regularizer"],momentum=0.9)

        if self.cst["anneiling"]:
            steps = 1103
            schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerG, T_0=steps, eta_min=1e-6)
            schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerD, T_0=steps, eta_min=1e-6)
        else:
            print("plateau")
            schedulerG = ReduceLROnPlateau(optimizerG, mode=self.cst["LRschedulerMode"], factor=self.cst["LRfactor"], 
                                        patience=self.cst["LRpatience"], verbose=True, cooldown=5)
            schedulerD = ReduceLROnPlateau(optimizerD, mode=self.cst["LRschedulerMode"], factor=self.cst["LRfactor"], 
                                        patience=self.cst["LRpatience"], verbose=True, cooldown=5)
                
        history = {'generator':{'train':[],'val':[]}, 
                    'discriminator':{'train':[],'val':[]}
                }

        # huber_loss = torch.nn.MSELoss()
        criterionD = torch.nn.BCELoss()
        huber_loss = torch.nn.MSELoss()

        best_model_wtsG = copy.deepcopy(gen_model)
        best_model_wtsD = copy.deepcopy(disc_model)

        earlyStopPatience = self.cst["earlyStopPatience"]
        n_epochs = self.cst['epochs_num']
        epochCount = 0
        best_loss = 1000000.0
        real_label = 1.
        fake_label = 0.

        finalTrainLossD = []
        finalTrainLossG = []
        finalValLossD = []
        finalValLossG = []
        fid_robot_history = {}
        fid_human_history = {}

        for epoch in tqdm.tqdm(range(1, n_epochs + 1)):

            disc_model.train()
            gen_model.train()

            train_losses_G = []
            train_losses_D = []
            train_losses_D_real = []
            train_losses_D_fake = []
            train_losses_D_R = []
            train_losses_D_real_R = []
            train_losses_D_fake_R = []
            train_style_losses = []
            train_nsgan_losses = []
            train_nsgan_losses_R = []
            train_huber_losses = []
            train_kdh_losses = []

            for _,(seq_robot, seq_human_t) in enumerate(tqdm.tqdm(self.data_batches["train"])):
                
                # pairs = utilities.get_pairs([i for i in range(seq_human.size()[0])])

                seq_robot = seq_robot.to(self.dev)
                seq_robot_mod = seq_robot.permute(0,2,1)

                seq_human = seq_human_t[0].to(self.dev)
                seq_human_mod = seq_human.permute(0,2,1)

                seq_human_f2 = seq_human_t[1].to(self.dev)
                seq_human_mod_f2 = seq_human_f2.permute(0,2,1)

                gen_model.VAE_human.encoder.init_hidden(len(seq_robot_mod))
                gen_model.VAE_human.decoder.init_hidden(len(seq_robot_mod))

                gen_model.VAE_robot.encoder.init_hidden(len(seq_robot_mod))
                gen_model.VAE_robot.decoder.init_hidden(len(seq_robot_mod))

                # #Write the network graph at epoch 0, batch 0
                # if epoch == 1:
                #     disc_model.eval()
                #     gen_model.eval()
                #     _, z_human, _, _, _, _, _ = gen_model.VAE_human(seq_human_mod.float())
                #     self.writer.add_graph(gen_model, input_to_model=(seq_robot_mod.float().detach(), 
                #                             seq_human_mod.float().detach()), verbose=False)
                #     self.writer.add_graph(disc_model, input_to_model=z_human.float().detach(), verbose=False)
                #     disc_model.train()
                #     gen_model.train()

                ###################################################################
                # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z))) #
                ###################################################################

                optimizerD.zero_grad()
                optimizerD_R.zero_grad()

                # Train with the estimation of latent human (z_human_hat) and latent human (z_human)
                out_hat_twist, z_human, z_robot, mu, log_var, \
                    z_human_hat, z_robot_hat, mu_hat,log_var_hat, \
                        cont_att, loc_att_human, loc_att_robot = gen_model(seq_robot_mod.float(), seq_human_mod.float())
                real_score = disc_model(z_human.float()).view(-1)
                est_score = disc_model(z_human_hat.detach().float()).view(-1)

                real_score_R = disc_model_R(z_robot.float()).view(-1)
                est_score_R = disc_model_R(z_robot_hat.detach().float()).view(-1)

                # # Compute error of D as sum over the fake and the real batches
                # D_loss = torch.sum(-torch.mean(torch.log(real_score + 1e-8)
                #             + torch.log(1 - est_score + 1e-8)))
                
                b_size = seq_robot_mod.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.dev)

                # Calculate loss on all-real batch
                errD_real = criterionD(real_score, label)
                errD_real_R = criterionD(real_score_R, label)

                # Update D
                if epoch > self.cst["wait_epoch_num"]:
                    errD_real.backward()
                    # errD_real_R.backward()

                label.fill_(fake_label)

                # Calculate loss on all-fake batch
                errD_fake = criterionD(est_score, label)
                errD_fake_R = criterionD(est_score_R, label)
                # print(errD_fake)

                D_loss = errD_fake + errD_real
                D_loss_R = errD_fake_R + errD_real_R

                # Update D
                if epoch > self.cst["wait_epoch_num"]:
                    errD_fake.backward()
                    # errD_fake_R.backward()
                # torch.nn.utils.clip_grad_norm_(disc_model.parameters(), 1.0)

                    optimizerD.step()
                    # optimizerD_R.step()

                if self.cst["anneiling"]:
                    schedulerD.step()

                #############################################################################
                # (2) Update Generator: maximize log(D(G(z))) + L_huber + L_style + L_nsgan #
                #############################################################################

                optimizerG.zero_grad()

                out_hat_twist_f2, z_human_f2, z_robot_f2, mu_f2, log_var_f2, \
                    z_human_hat_f2, z_robot_hat2, mu_hat_f2,log_var_hat_f2, \
                        cont_att_f2, loc_att_human_f2, loc_att_robot_f2 = gen_model(seq_robot_mod.float(), seq_human_mod_f2.float())
                    
                
                label.fill_(real_label)
                est_score1 = disc_model(z_human_hat.float()).view(-1)
                est_score1_R = disc_model_R(z_robot_hat.float()).view(-1)
                # nsgan = -torch.mean(torch.log(est_score1 + 1e-8))
                nsgan = criterionD(est_score1, label)
                nsgan_R = criterionD(est_score1_R, label)
                huber = huber_loss(out_hat_twist,seq_robot_mod.float())
                style = loss_functions_variations.funct_match(self.cst["generator_loss_add"],
                                                                z_hat=z_human_f2, z=z_human, 
                                                                x_hat=out_hat_twist, x=out_hat_twist_f2,
                                                                log_var=log_var, log_var_hat=log_var_hat,
                                                                mu_hat=mu_hat, mu=mu)
                kd_h = loss_functions_variations.funct_match("kld",
                                                                z_hat=z_human_hat, z=z_human, 
                                                                x_hat=out_hat_twist, x=seq_robot_mod.float(),
                                                                log_var=log_var, log_var_hat=log_var_hat,
                                                                mu_hat=mu_hat, mu=mu)
                kd_r = loss_functions_variations.funct_match("kld",
                                                                z_hat=z_human_hat, z=z_human, 
                                                                x_hat=out_hat_twist, x=seq_robot_mod.float(),
                                                                log_var=log_var, log_var_hat=log_var_hat,
                                                                mu_hat=mu_hat, mu=mu)

                if epoch > self.cst["wait_epoch_num"]:
                    G_loss =    self.cst["lambda_coeff"][1]*huber + self.cst["lambda_coeff"][2]*style  + \
                                 self.cst["lambda_coeff"][0]*nsgan #+ self.cst["lambda_coeff"][3]*kd_h[0]
                                    # self.cst["lambda_coeff"][0]*nsgan +
                                    # self.cst["lambda_coeff"][2]*style +
                else:
                    G_loss =   self.cst["lambda_coeff"][1]*huber + \
                               self.cst["lambda_coeff"][2]*style # + self.cst["lambda_coeff"][3]*kd_h[0]


                # G_loss = loss_functions_variations.calc_l1_norm(G_loss,gen_model)
                
                G_loss.backward()
                # torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 1.0)
                optimizerG.step()

                if self.cst["anneiling"]:
                    # print(self.get_lr(optimizerG))
                    # print(schedulerG.get_lr())
                    schedulerG.step()

                train_losses_D_fake.append(errD_real.item())
                train_losses_D_real.append(errD_fake.item())
                train_losses_D_fake_R.append(errD_real_R.item())
                train_losses_D_real_R.append(errD_fake_R.item())
                train_losses_G.append(G_loss.item())
                train_losses_D.append(D_loss.item())
                train_losses_D_R.append(D_loss_R.item())
                train_style_losses.append(style.item())
                train_nsgan_losses.append(nsgan.item())
                train_nsgan_losses_R.append(nsgan_R.item())
                train_huber_losses.append(huber.item())
                train_kdh_losses.append(kd_h[0].item())

                # if self.cst["anneiling"]:
                #     schedulerD.step()
                #     schedulerG.step()
                    
            disc_model.eval()
            gen_model.eval()
            val_losses_G = []
            val_losses_D = []
            val_losses_D_real = []
            val_losses_D_fake = []
            val_losses_D_R = []
            val_losses_D_real_R = []
            val_losses_D_fake_R = []
            val_style_losses = []
            val_nsgan_losses = []
            val_nsgan_losses_R = []
            val_huber_losses = []
            val_kdh_losses = []
            fid_distance_human = []
            fid_distance_robot = []
            with torch.no_grad():
                for _,(seq_robot, seq_human_t) in enumerate(tqdm.tqdm(self.data_batches["val"])):

                    # pairs = utilities.get_pairs([i for i in range(seq_human.size()[0])])

                    seq_robot = seq_robot.to(self.dev)
                    seq_robot_mod = seq_robot.permute(0,2,1)

                    seq_human = seq_human_t[0].to(self.dev)
                    seq_human_mod = seq_human.permute(0,2,1)

                    seq_human_f2 = seq_human_t[1].to(self.dev)
                    seq_human_mod_f2 = seq_human_f2.permute(0,2,1)

                    gen_model.VAE_human.encoder.init_hidden(len(seq_human_mod))
                    gen_model.VAE_human.decoder.init_hidden(len(seq_human_mod))

                    gen_model.VAE_robot.encoder.init_hidden(len(seq_robot_mod))
                    gen_model.VAE_robot.decoder.init_hidden(len(seq_robot_mod))

                    # _, z_human, _, _, _, _, _ = gen_model.VAE_human(seq_human_mod.float())

                    # Train with the estimation of latent human (z_human)
                    out_hat_twist, z_human, z_robot, mu, log_var, \
                        z_human_hat, z_robot_hat, mu_hat,log_var_hat, \
                        cont_att, loc_att_human, loc_att_robot = gen_model(seq_robot_mod.float(), seq_human_mod.float())
                    est_score = disc_model(z_human_hat.detach()).view(-1)
                    real_score = disc_model(z_human.float()).view(-1)

                    est_score_R = disc_model_R(z_robot_hat.detach()).view(-1)
                    real_score_R = disc_model_R(z_robot.float()).view(-1)

                    self.evaluation_method.set_inputs((seq_human,seq_robot),out_hat_twist, 
                                                        (z_human, z_robot, z_human_hat, z_robot_hat))
                    expressive_dict, fid_feat_distance = self.evaluation_method.main_evaluation(self.cst["visualize_evaluation"])

                    # Compute error of D as sum over the fake and the real batches
                    # D_loss = torch.sum(-torch.mean(torch.log(real_score + 1e-8)
                    #             + torch.log(1 - est_score + 1e-8)))
                    # Calculate loss on all-real batch
                    b_size = seq_robot_mod.size(0)
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=self.dev)
                    errD_real = criterionD(real_score, label)
                    errD_real_R = criterionD(real_score_R, label)

                    label.fill_(fake_label)

                    # Calculate loss on all-real batch
                    errD_fake = criterionD(est_score, label)
                    errD_fake_R = criterionD(est_score_R, label)
                    D_loss = errD_fake + errD_real
                    D_loss_R = errD_fake_R + errD_real_R

                    # nsgan_val = -torch.mean(torch.log(est_score + 1e-8))
                    out_hat_twist_f2, z_human_f2, z_robot_f2, mu_f2, log_var_f2, \
                    z_human_hat_f2,z_robot_hat2,mu_hat_f2,log_var_hat_f2, \
                        cont_att_f2, loc_att_human_f2, loc_att_robot_f2 = gen_model(seq_robot_mod.float(), seq_human_mod_f2.float())

                    label.fill_(real_label)
                    nsgan_val = criterionD(est_score, label)
                    nsgan_val_R = criterionD(est_score_R, label)
                    huber_val = huber_loss(out_hat_twist,seq_robot_mod.float())
                    style_val = loss_functions_variations.funct_match(self.cst["generator_loss_add"],
                                                                z_hat=z_human, z=z_human_f2, 
                                                                x_hat=out_hat_twist, x=out_hat_twist_f2,
                                                                log_var=log_var, log_var_hat=log_var_hat,
                                                                mu_hat=mu_hat, mu=mu)
                    kd_h_val = loss_functions_variations.funct_match("kld",
                                                                z_hat=z_human_hat, z=z_human, 
                                                                x_hat=out_hat_twist, x=seq_robot_mod.float(),
                                                                log_var=log_var, log_var_hat=log_var_hat,
                                                                mu_hat=mu_hat, mu=mu)
                    kd_r_val = loss_functions_variations.funct_match("kld",
                                                                z_hat=z_human_hat, z=z_human, 
                                                                x_hat=out_hat_twist, x=seq_robot_mod.float(),
                                                                log_var=log_var, log_var_hat=log_var_hat,
                                                                mu_hat=mu_hat, mu=mu)

                    if epoch > self.cst["wait_epoch_num"]:
                        G_loss =   self.cst["lambda_coeff"][0]*nsgan_val + self.cst["lambda_coeff"][1]*huber_val + \
                                    self.cst["lambda_coeff"][2]*style_val + self.cst["lambda_coeff"][0]*nsgan_val_R #+ \
                                        # self.cst["lambda_coeff"][3]*kd_h_val[0]
                    else:
                        G_loss =   self.cst["lambda_coeff"][1]*huber_val + \
                                    self.cst["lambda_coeff"][2]*style_val #+ self.cst["lambda_coeff"][3]*kd_h_val[0] 
                        
                    val_losses_D_fake.append(errD_fake.item())
                    val_losses_D_real.append(errD_real.item())
                    val_losses_D_fake_R.append(errD_fake_R.item())
                    val_losses_D_real_R.append(errD_real_R.item())
                    val_losses_G.append(G_loss.item())
                    val_losses_D.append(D_loss.item())
                    val_losses_D_R.append(D_loss_R.item())
                    val_style_losses.append(style_val.item())
                    val_nsgan_losses.append(nsgan_val.item())
                    val_nsgan_losses_R.append(nsgan_val_R.item())
                    val_huber_losses.append(huber_val.item())
                    val_kdh_losses.append(kd_h_val[0].item())

                    fid_distance_human.append(fid_feat_distance[0])
                    fid_distance_robot.append(fid_feat_distance[1])

            val_lossG = np.mean(val_losses_G)
            val_lossD = np.mean(val_losses_D)
            val_lossD_R = np.mean(val_losses_D_R)
            train_lossG = np.mean(train_losses_G)
            train_lossD = np.mean(train_losses_D)
            train_lossD_R = np.mean(train_losses_D_R)

            train_lossD_fake = np.mean(train_losses_D_fake)
            train_lossD_real = np.mean(train_losses_D_real)
            val_lossD_fake = np.mean(val_losses_D_fake)
            val_lossD_real = np.mean(val_losses_D_real)

            train_lossD_fake_R = np.mean(train_losses_D_fake_R)
            train_lossD_real_R = np.mean(train_losses_D_real_R)
            val_lossD_fake_R = np.mean(val_losses_D_fake_R)
            val_lossD_real_R = np.mean(val_losses_D_real_R)

            fid_human_history["epoch"+str(epoch)] = fid_distance_human
            fid_robot_history["epoch"+str(epoch)] = fid_distance_robot
            test_human_mean_fid = np.mean(fid_distance_human)
            test_robot_mean_fid = np.mean(fid_distance_robot)

            self.writer.add_scalar('Generator/Loss/train', train_lossG, epoch)
            self.writer.add_scalar('Generator/Loss/test', val_lossG, epoch)
            self.writer.add_scalar('Discriminator/Loss/train', train_lossD, epoch)
            self.writer.add_scalar('Discriminator/Loss/test', val_lossD, epoch)
            self.writer.add_scalar('Discriminator/Loss/train_R', train_lossD_R, epoch)
            self.writer.add_scalar('Discriminator/Loss/test_R', val_lossD_R, epoch)
            self.writer.add_scalar('Discriminator/Loss Fake/train', train_lossD_fake, epoch)
            self.writer.add_scalar('Discriminator/Loss Fake/train_R', train_lossD_fake_R, epoch)
            self.writer.add_scalar('Discriminator/Loss Fake/test', val_lossD_fake, epoch)
            self.writer.add_scalar('Discriminator/Loss Fake/test_R', val_lossD_fake_R, epoch)
            self.writer.add_scalar('Discriminator/Loss Real/train', train_lossD_real, epoch)
            self.writer.add_scalar('Discriminator/Loss Real/train_R', train_lossD_real_R, epoch)
            self.writer.add_scalar('Discriminator/Loss Real/test', val_lossD_real, epoch)
            self.writer.add_scalar('Discriminator/Loss Real/test_R', val_lossD_real_R, epoch)

            for tag, arr in expressive_dict["human"].items():
                if tag=="stats":
                    continue
                laban_value = ((arr-expressive_dict["robot_generated"][tag])**2).mean()
                self.writer.add_scalar('Evaluation Method/Laban Qualities/Test ' + str(tag), laban_value, epoch)
            
            self.writer.add_scalar('Evaluation Method/FID Feature Distance Mean/Test Human', test_human_mean_fid, epoch)
            self.writer.add_scalar('Evaluation Method/FID Feature Distance Mean/Test Robot', test_robot_mean_fid, epoch)
            self.writer.add_scalar('Evaluation Method/Cosine Similarity Mean/Test Human', 
                                        np.mean(torch.nn.functional.cosine_similarity(z_human.detach().cpu(), z_human_hat.detach().cpu()).numpy()), 
                                        epoch)
            self.writer.add_scalar('Evaluation Method/Cosine Similarity Mean/Test Robot', 
                                        np.mean(torch.nn.functional.cosine_similarity(z_robot.detach().cpu(),z_robot_hat.detach().cpu()).numpy()), 
                                        epoch)

            history['generator']['train'].append(train_lossG)
            history['discriminator']['train'].append(train_lossD)
            history['generator']['val'].append(val_lossG)
            history['discriminator']['val'].append(val_lossD)
            
            if epoch > self.cst["wait_epoch_num"]:
                if val_lossG < best_loss:
                    epochCount = 0
                    best_loss = val_lossG
                    best_model_wtsD = copy.deepcopy(disc_model.state_dict())
                    best_model_wtsG = copy.deepcopy(gen_model.state_dict())
                else:
                    if epoch > self.cst["wait_epoch_num"]:
                        epochCount+=1
                    else: 
                        epochCount=0

            # self.compare_models(model_a,model)
            
            # if self.cst["anneiling"]:
            #     schedulerD.step(0)
            #     schedulerG.step(0)
            # else:
            #     # Learning rate scheduler update
            #     if epoch > 50:
            #         schedulerG.step(val_lossG)
            #         schedulerD.step(val_lossD)

            # Early stop
            # if epochCount > earlyStopPatience:
            #     print('Early Stop')
            #     break

            finalTrainLossG.append(train_losses_G)
            finalTrainLossD.append(train_losses_D)
            finalValLossG.append(val_losses_G)
            finalValLossD.append(val_losses_D)

            # if epoch%25==0:
            #     plot_buf = self.save_images([loc_att_human[0], loc_att_robot[0],z_human,z_robot])
            #     image = PIL.Image.open(plot_buf)
            #     image = ToTensor()(image).squeeze(0)
            #     self.writer.add_image(f'Attention/ Attention Outputs, Epoch {epoch}', image, epoch)

            # if epoch%25==0:
            #     for qualitie in self.evaluation_method.expressive_qualities:
            #         plot_buf = self.evaluation_method.visualize_qualities_stream([value for value in expressive_dict.values()], qualitie)
            #         image = PIL.Image.open(plot_buf)
            #         image = ToTensor()(image).squeeze(0)
            #         self.writer.add_image(f'Evaluation/ Expressive Qualities/, Epoch {epoch} Qualitie{qualitie}', image, epoch)


            if epoch%10==0:
                plot_buf = self.visualize_data(seq_robot_mod.cpu().detach().numpy(),
                                    seq_human_mod.cpu().detach().numpy(), 
                                    out_hat_twist.cpu().detach().numpy(),
                                    epoch)
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).squeeze(0)
                self.writer.add_image(f'Reconstruction/ Validation, Epoch {epoch} First', image, epoch)

                plot_buf = self.visualize_data(seq_robot_mod.cpu().detach().numpy(),
                                    seq_human_mod_f2.cpu().detach().numpy(), 
                                    out_hat_twist_f2.cpu().detach().numpy(),
                                    epoch, tag="second")
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).squeeze(0)
                self.writer.add_image(f'Reconstruction/ Validation, Epoch {epoch} Second', image, epoch)

            del seq_human
            del seq_robot
            del out_hat_twist

            models_list = [gen_model, disc_model]
            for model in models_list:
                for name, param in model.named_parameters():
                    self.add_tensorboard_weight(param, epoch, name=name)
            
            self.writer.add_scalar('Generator/LR/Learning rate',self.get_lr(optimizerG), epoch)
            self.writer.add_scalar('Discriminator/LR/Learning rate',self.get_lr(optimizerD), epoch)
            
            self.writer.add_scalar('Generator/Loss/Loss variation/Train ' + self.cst["generator_loss_add"],
                                    self.cst["lambda_coeff"][2]*np.mean(train_style_losses), epoch)
            self.writer.add_scalar('Generator/Loss/Loss variation/Val ' + self.cst["generator_loss_add"],
                                    self.cst["lambda_coeff"][2]*np.mean(val_style_losses), epoch)

            self.writer.add_scalar('Generator/Loss/KLD/Train ',
                                    self.cst["lambda_coeff"][3]*np.mean(train_kdh_losses), epoch)
            self.writer.add_scalar('Generator/Loss/KLD/Val ',
                                    self.cst["lambda_coeff"][3]*np.mean(val_kdh_losses), epoch)

            self.writer.add_scalar('Generator/Loss/Huber Loss/Train',
                                    self.cst["lambda_coeff"][1]*np.mean(train_huber_losses), epoch)
            self.writer.add_scalar('Generator/Loss/Huber Loss/Val',
                                    self.cst["lambda_coeff"][1]*np.mean(val_huber_losses), epoch)
            
            self.writer.add_scalar('Generator/Loss/NSGAN/Train',
                                    self.cst["lambda_coeff"][0]*np.mean(train_nsgan_losses), epoch)
            self.writer.add_scalar('Generator/Loss/NSGAN/Val',
                                    self.cst["lambda_coeff"][0]*np.mean(val_nsgan_losses), epoch)
            self.writer.add_scalar('Generator/Loss/NSGAN/Train_R',
                                    self.cst["lambda_coeff"][0]*np.mean(train_nsgan_losses_R), epoch)
            self.writer.add_scalar('Generator/Loss/NSGAN/Val_R',
                                    self.cst["lambda_coeff"][0]*np.mean(val_nsgan_losses_R), epoch)

            # for i in range(len(fid_distance_human)):
            #     index_h = np.argmin()
            #     index_h = np.argmin()
            #     self.writer.add_scalar('Evaluation Method/FID Feature Distance/Test Human Feature Mean Feature- ' + str(i), 
            #                             np.mean(fid_distance_human[i]), epoch)
            #     self.writer.add_scalar('Evaluation Method/FID Feature Distance/Test Robot Feature Mean Feature- ' + str(i), 
            #                             np.mean(fid_distance_robot[i]), epoch)

            print(f'Epoch {epoch}: Generator loss Train: {train_lossG}, Generator loss Val: {val_lossG},\n \
                    Discriminator Real loss Train: {train_lossD_real}, Discriminator Real loss Val: {val_lossD_real},\n \
                    Discriminator Fake loss Train: {train_lossD_fake}, Discriminator Real Fake loss Val: {val_lossD_fake}, \n \
                    Discriminator Real loss Train Robot: {train_lossD_real_R}, Discriminator Real loss Val Robot: {val_lossD_real_R},\n \
                    Discriminator Fake loss Train Robot: {train_lossD_fake_R}, Discriminator Real Fake loss Val Robot: {val_lossD_fake_R}, \n \
                    Discriminator loss Train: {train_lossD}, Discriminator loss Val: {val_lossD}, \n \
                    Discriminator loss Train Robot: {train_lossD_R}, Discriminator loss Val Robot: {val_lossD_R} \n')

        # model = best_model_wts
        x = np.linspace(0,len(history['generator']['train']),len(history['generator']['train']))
        x1 = np.linspace(0,len(history['generator']['val']),len(history['generator']['val']))
        plt.plot(x,history['generator']['train'])
        plt.plot(x1,history['generator']['val'])

        plt.figure()
        x = np.linspace(0,len(history['discriminator']['train']),len(history['discriminator']['train']))
        x1 = np.linspace(0,len(history['discriminator']['val']),len(history['discriminator']['val']))
        plt.plot(x,history['discriminator']['train'])
        plt.plot(x1,history['discriminator']['val'])
        plt.show()

        with open(os.path.join(self.cst['historyPath'], "history_fretchet_human" +'.json'), 'w') as json_file:
            json.dump(fid_human_history, json_file)

        with open(os.path.join(self.cst['historyPath'], "history_fretchet_robot" +'.json'), 'w') as json_file:
            json.dump(fid_robot_history, json_file)

        best_model_wtsD = copy.deepcopy(disc_model.state_dict())
        best_model_wtsG = copy.deepcopy(gen_model.state_dict())
        optimizers = [optimizerD, optimizerG]
        best_model_wts = (best_model_wtsD, best_model_wtsG)
        finalTrainLoss = (finalTrainLossD,finalTrainLossG)
        finalValLoss = (finalValLossD, finalValLossG)
        self.saveOutput(best_model_wts,epoch,history,optimizers, best_loss, finalTrainLoss, finalValLoss, self.cst["modelKey"], multipleModels=True)
        self.writer.flush()

        return history, best_model_wts, gen_model
    
    def training_loop_gan_neutral_style(self):
        """
        Trainning module for a GAN architecture. 
        Based on tutorial from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        """
        # self.set_deterministic_env()
        disc_model, gen_model = self.getModel(self.cst["model_config"])
        disc_model_R = copy.deepcopy(disc_model)

        if self.cst["optimizer"]=="adam":
            print("adam")
            optimizerG = torch.optim.Adam(gen_model.parameters(), lr=self.cst["h_param"]["generator"]['learning_rate'], 
                                            weight_decay=self.cst["h_param"]["generator"]["L2regularizer"], amsgrad=False, 
                                            betas=(self.cst["h_param"]["generator"]["beta1"], 0.98),eps=1e-8)
            optimizerD = torch.optim.Adam(disc_model.parameters(), lr=self.cst["h_param"]["discriminator"]['learning_rate'], 
                                            weight_decay=self.cst["h_param"]["discriminator"]["L2regularizer"], amsgrad=False,
                                            betas=(self.cst["h_param"]["discriminator"]["beta1"], 0.98),eps=1e-8)
            optimizerD_R = torch.optim.AdamW(disc_model_R.parameters(), lr=self.cst["h_param"]["discriminator"]['learning_rate'], 
                                            weight_decay=self.cst["h_param"]["discriminator"]["L2regularizer"], amsgrad=False,
                                            betas=(self.cst["h_param"]["discriminator"]["beta1"], 0.98),eps=1e-8)
            # optimizerG = torch.optim.RMSprop(gen_model.parameters(), lr=self.cst["h_param"]["generator"]['learning_rate'], 
            #                                 weight_decay=self.cst["h_param"]["generator"]["L2regularizer"], momentum=0.9)
            # optimizerD = torch.optim.RMSprop(disc_model.parameters(), lr=self.cst["h_param"]["discriminator"]['learning_rate'], 
            #                                 weight_decay=self.cst["h_param"]["discriminator"]["L2regularizer"], momentum=0.9)
        else:
            optimizerG = torch.optim.SGD(gen_model.parameters(), lr=self.cst["h_param"]["generator"]['learning_rate'], 
                                            weight_decay=self.cst["h_param"]["generator"]["L2regularizer"],momentum=0.9,)
            optimizerD = torch.optim.SGD(disc_model.parameters(), lr=self.cst["h_param"]["discriminator"]['learning_rate'], 
                                            weight_decay=self.cst["h_param"]["discriminator"]["L2regularizer"],momentum=0.9)

        if self.cst["anneiling"]:
            steps = 1103
            schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerG, T_0=steps, eta_min=1e-6)
            schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerD, T_0=steps, eta_min=1e-6)
        else:
            print("plateau")
            schedulerG = ReduceLROnPlateau(optimizerG, mode=self.cst["LRschedulerMode"], factor=self.cst["LRfactor"], 
                                        patience=self.cst["LRpatience"], verbose=True, cooldown=5)
            schedulerD = ReduceLROnPlateau(optimizerD, mode=self.cst["LRschedulerMode"], factor=self.cst["LRfactor"], 
                                        patience=self.cst["LRpatience"], verbose=True, cooldown=5)
                
        history = {'generator':{'train':[],'val':[]}, 
                    'discriminator':{'train':[],'val':[]}
                }

        # huber_loss = torch.nn.MSELoss()
        criterionD = torch.nn.BCELoss()
        huber_loss = torch.nn.MSELoss()

        best_model_wtsG = copy.deepcopy(gen_model)
        best_model_wtsD = copy.deepcopy(disc_model)

        earlyStopPatience = self.cst["earlyStopPatience"]
        n_epochs = self.cst['epochs_num']
        epochCount = 0
        best_loss = 1000000.0
        real_label = 1.
        fake_label = 0.

        finalTrainLossD = []
        finalTrainLossG = []
        finalValLossD = []
        finalValLossG = []
        fid_robot_history = {}
        fid_human_history = {}

        for epoch in tqdm.tqdm(range(1, n_epochs + 1)):

            disc_model.train()
            gen_model.train()

            train_losses_G = []
            train_losses_D = []
            train_losses_D_real = []
            train_losses_D_fake = []
            train_losses_D_R = []
            train_losses_D_real_R = []
            train_losses_D_fake_R = []
            train_style_losses = []
            train_nsgan_losses = []
            train_nsgan_losses_R = []
            train_huber_losses = []
            train_kdh_losses = []

            for _,(seq_robot, seq_human_t) in enumerate(tqdm.tqdm(self.data_batches_neutral_style["train"])):
                
                # pairs = utilities.get_pairs([i for i in range(seq_human.size()[0])])

                seq_robot = seq_robot.to(self.dev)
                seq_robot_mod = seq_robot.permute(0,2,1)

                seq_human = seq_human_t[0].to(self.dev)
                seq_human_mod = seq_human.permute(0,2,1)

                seq_human_f2 = seq_human_t[2].to(self.dev)
                seq_human_mod_f2 = seq_human_f2.permute(0,2,1)

                seq_human_neutral_style_1 = seq_human_t[1].to(self.dev)
                seq_human_neutral_style_2 = seq_human_t[3].to(self.dev)

                gen_model.VAE_human.encoder.init_hidden(len(seq_robot_mod))
                gen_model.VAE_human.decoder.init_hidden(len(seq_robot_mod))

                gen_model.VAE_robot.encoder.init_hidden(len(seq_robot_mod))
                gen_model.VAE_robot.decoder.init_hidden(len(seq_robot_mod))

                # #Write the network graph at epoch 0, batch 0
                # if epoch == 1:
                #     disc_model.eval()
                #     gen_model.eval()
                #     _, z_human, _, _, _, _, _ = gen_model.VAE_human(seq_human_mod.float())
                #     self.writer.add_graph(gen_model, input_to_model=(seq_robot_mod.float().detach(), 
                #                             seq_human_mod.float().detach()), verbose=False)
                #     self.writer.add_graph(disc_model, input_to_model=z_human.float().detach(), verbose=False)
                #     disc_model.train()
                #     gen_model.train()

                ###################################################################
                # (1) Update Discriminator: maximize log(D(x)) + log(1 - D(G(z))) #
                ###################################################################

                optimizerD.zero_grad()
                optimizerD_R.zero_grad()

                # Train with the estimation of latent human (z_human_hat) and latent human (z_human)
                out_hat_twist, z_human, z_robot, mu, log_var, \
                    z_human_hat, z_robot_hat, mu_hat,log_var_hat, \
                        cont_att, loc_att_human, loc_att_robot = gen_model(seq_robot_mod.float(), seq_human_mod.float(), 
                                                                            seq_human_neutral_style_1)
                real_score = disc_model(z_human.float()).view(-1)
                est_score = disc_model(z_human_hat.detach().float()).view(-1)

                real_score_R = disc_model_R(z_robot.float()).view(-1)
                est_score_R = disc_model_R(z_robot_hat.detach().float()).view(-1)

                # # Compute error of D as sum over the fake and the real batches
                # D_loss = torch.sum(-torch.mean(torch.log(real_score + 1e-8)
                #             + torch.log(1 - est_score + 1e-8)))
                
                b_size = seq_robot_mod.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.dev)

                # Calculate loss on all-real batch
                errD_real = criterionD(real_score, label)
                errD_real_R = criterionD(real_score_R, label)

                # Update D
                if epoch > self.cst["wait_epoch_num"]:
                    errD_real.backward()
                    # errD_real_R.backward()

                label.fill_(fake_label)

                # Calculate loss on all-fake batch
                errD_fake = criterionD(est_score, label)
                errD_fake_R = criterionD(est_score_R, label)
                # print(errD_fake)

                D_loss = errD_fake + errD_real
                D_loss_R = errD_fake_R + errD_real_R

                # Update D
                if epoch > self.cst["wait_epoch_num"]:
                    errD_fake.backward()
                    # errD_fake_R.backward()
                # torch.nn.utils.clip_grad_norm_(disc_model.parameters(), 1.0)

                    optimizerD.step()
                    # optimizerD_R.step()

                if self.cst["anneiling"]:
                    schedulerD.step()

                #############################################################################
                # (2) Update Generator: maximize log(D(G(z))) + L_huber + L_style + L_nsgan #
                #############################################################################

                optimizerG.zero_grad()

                out_hat_twist_f2, z_human_f2, z_robot_f2, mu_f2, log_var_f2, \
                    z_human_hat_f2, z_robot_hat2, mu_hat_f2,log_var_hat_f2, \
                        cont_att_f2, loc_att_human_f2, loc_att_robot_f2 = gen_model(seq_robot_mod.float(), seq_human_mod_f2.float(),
                                                                                    seq_human_neutral_style_2)
                    
                
                label.fill_(real_label)
                est_score1 = disc_model(z_human_hat.float()).view(-1)
                est_score1_R = disc_model_R(z_robot_hat.float()).view(-1)
                # nsgan = -torch.mean(torch.log(est_score1 + 1e-8))
                nsgan = criterionD(est_score1, label)
                nsgan_R = criterionD(est_score1_R, label)
                huber = huber_loss(out_hat_twist,seq_robot_mod.float())
                # huber = torch.sum(huber,dim=1)
                # huber = torch.mean(huber,dim=(1,0))
                style = loss_functions_variations.funct_match(self.cst["generator_loss_add"],
                                                                z_hat=z_human_f2, z=z_human, 
                                                                x_hat=out_hat_twist, x=out_hat_twist_f2,
                                                                log_var=log_var, log_var_hat=log_var_hat,
                                                                mu_hat=mu_hat, mu=mu)
                kd_h = loss_functions_variations.funct_match("kld",
                                                                z_hat=z_human_hat, z=z_human, 
                                                                x_hat=out_hat_twist, x=seq_robot_mod.float(),
                                                                log_var=log_var, log_var_hat=log_var_hat,
                                                                mu_hat=mu_hat, mu=mu)
                kd_r = loss_functions_variations.funct_match("kld",
                                                                z_hat=z_human_hat, z=z_human, 
                                                                x_hat=out_hat_twist, x=seq_robot_mod.float(),
                                                                log_var=log_var, log_var_hat=log_var_hat,
                                                                mu_hat=mu_hat, mu=mu)

                if epoch > self.cst["wait_epoch_num"]:
                    G_loss =    self.cst["lambda_coeff"][1]*huber + self.cst["lambda_coeff"][2]*style  + \
                                 self.cst["lambda_coeff"][0]*nsgan + self.cst["lambda_coeff"][3]*kd_h[0]
                                    # self.cst["lambda_coeff"][0]*nsgan +
                                    # self.cst["lambda_coeff"][2]*style +
                else:
                    G_loss =   self.cst["lambda_coeff"][1]*huber + \
                               self.cst["lambda_coeff"][2]*style  + self.cst["lambda_coeff"][3]*kd_h[0]


                # G_loss = loss_functions_variations.calc_l1_norm(G_loss,gen_model)
                
                G_loss.backward()
                # torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 1.0)
                optimizerG.step()

                if self.cst["anneiling"]:
                    # print(self.get_lr(optimizerG))
                    # print(schedulerG.get_lr())
                    schedulerG.step()

                train_losses_D_fake.append(errD_real.item())
                train_losses_D_real.append(errD_fake.item())
                train_losses_D_fake_R.append(errD_real_R.item())
                train_losses_D_real_R.append(errD_fake_R.item())
                train_losses_G.append(G_loss.item())
                train_losses_D.append(D_loss.item())
                train_losses_D_R.append(D_loss_R.item())
                train_style_losses.append(style.item())
                train_nsgan_losses.append(nsgan.item())
                train_nsgan_losses_R.append(nsgan_R.item())
                train_huber_losses.append(huber.item())
                train_kdh_losses.append(kd_h[0].item())

                # if self.cst["anneiling"]:
                #     schedulerD.step()
                #     schedulerG.step()
                    
            disc_model.eval()
            gen_model.eval()
            val_losses_G = []
            val_losses_D = []
            val_losses_D_real = []
            val_losses_D_fake = []
            val_losses_D_R = []
            val_losses_D_real_R = []
            val_losses_D_fake_R = []
            val_style_losses = []
            val_nsgan_losses = []
            val_nsgan_losses_R = []
            val_huber_losses = []
            val_kdh_losses = []
            fid_distance_human = []
            fid_distance_robot = []
            with torch.no_grad():
                for _,(seq_robot, seq_human_t) in enumerate(tqdm.tqdm(self.data_batches_neutral_style["val"])):

                    # pairs = utilities.get_pairs([i for i in range(seq_human.size()[0])])

                    seq_robot = seq_robot.to(self.dev)
                    seq_robot_mod = seq_robot.permute(0,2,1)

                    seq_human = seq_human_t[0].to(self.dev)
                    seq_human_mod = seq_human.permute(0,2,1)

                    seq_human_f2 = seq_human_t[2].to(self.dev)
                    seq_human_mod_f2 = seq_human_f2.permute(0,2,1)

                    seq_human_neutral_style_1 = seq_human_t[1].to(self.dev)
                    seq_human_neutral_style_2 = seq_human_t[3].to(self.dev)

                    gen_model.VAE_human.encoder.init_hidden(len(seq_human_mod))
                    gen_model.VAE_human.decoder.init_hidden(len(seq_human_mod))

                    gen_model.VAE_robot.encoder.init_hidden(len(seq_robot_mod))
                    gen_model.VAE_robot.decoder.init_hidden(len(seq_robot_mod))

                    # _, z_human, _, _, _, _, _ = gen_model.VAE_human(seq_human_mod.float())

                    # Train with the estimation of latent human (z_human)
                    out_hat_twist, z_human, z_robot, mu, log_var, \
                        z_human_hat, z_robot_hat, mu_hat,log_var_hat, \
                        cont_att, loc_att_human, loc_att_robot = gen_model(seq_robot_mod.float(), seq_human_mod.float(),
                                                                            seq_human_neutral_style_1)
                    est_score = disc_model(z_human_hat.detach()).view(-1)
                    real_score = disc_model(z_human.float()).view(-1)

                    est_score_R = disc_model_R(z_robot_hat.detach()).view(-1)
                    real_score_R = disc_model_R(z_robot.float()).view(-1)

                    self.evaluation_method.set_inputs((seq_human,seq_robot),out_hat_twist, 
                                                        (z_human, z_robot, z_human_hat, z_robot_hat))
                    expressive_dict, fid_feat_distance = self.evaluation_method.main_evaluation(self.cst["visualize_evaluation"])

                    # Compute error of D as sum over the fake and the real batches
                    # D_loss = torch.sum(-torch.mean(torch.log(real_score + 1e-8)
                    #             + torch.log(1 - est_score + 1e-8)))
                    # Calculate loss on all-real batch
                    b_size = seq_robot_mod.size(0)
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=self.dev)
                    errD_real = criterionD(real_score, label)
                    errD_real_R = criterionD(real_score_R, label)

                    label.fill_(fake_label)

                    # Calculate loss on all-real batch
                    errD_fake = criterionD(est_score, label)
                    errD_fake_R = criterionD(est_score_R, label)
                    D_loss = errD_fake + errD_real
                    D_loss_R = errD_fake_R + errD_real_R

                    # nsgan_val = -torch.mean(torch.log(est_score + 1e-8))
                    out_hat_twist_f2, z_human_f2, z_robot_f2, mu_f2, log_var_f2, \
                    z_human_hat_f2,z_robot_hat2,mu_hat_f2,log_var_hat_f2, \
                        cont_att_f2, loc_att_human_f2, loc_att_robot_f2 = gen_model(seq_robot_mod.float(), seq_human_mod_f2.float(),
                                                                                        seq_human_neutral_style_2)

                    label.fill_(real_label)
                    nsgan_val = criterionD(est_score, label)
                    nsgan_val_R = criterionD(est_score_R, label)
                    huber_val = huber_loss(out_hat_twist,seq_robot_mod.float())
                    # huber_val = torch.sum(huber_val,dim=1)
                    # huber_val = torch.mean(huber_val,dim=(1,0))
                    style_val = loss_functions_variations.funct_match(self.cst["generator_loss_add"],
                                                                z_hat=z_human, z=z_human_f2, 
                                                                x_hat=out_hat_twist, x=out_hat_twist_f2,
                                                                log_var=log_var, log_var_hat=log_var_hat,
                                                                mu_hat=mu_hat, mu=mu)
                    kd_h_val = loss_functions_variations.funct_match("kld",
                                                                z_hat=z_human_hat, z=z_human, 
                                                                x_hat=out_hat_twist, x=seq_robot_mod.float(),
                                                                log_var=log_var, log_var_hat=log_var_hat,
                                                                mu_hat=mu_hat, mu=mu)
                    kd_r_val = loss_functions_variations.funct_match("kld",
                                                                z_hat=z_human_hat, z=z_human, 
                                                                x_hat=out_hat_twist, x=seq_robot_mod.float(),
                                                                log_var=log_var, log_var_hat=log_var_hat,
                                                                mu_hat=mu_hat, mu=mu)

                    if epoch > self.cst["wait_epoch_num"]:
                        G_loss =   self.cst["lambda_coeff"][0]*nsgan_val + self.cst["lambda_coeff"][1]*huber_val + \
                                    self.cst["lambda_coeff"][2]*style_val + self.cst["lambda_coeff"][0]*nsgan_val_R + \
                                        self.cst["lambda_coeff"][3]*kd_h_val[0]
                    else:
                        G_loss =   self.cst["lambda_coeff"][1]*huber_val + \
                                    self.cst["lambda_coeff"][2]*style_val + self.cst["lambda_coeff"][3]*kd_h_val[0] 
                        
                    val_losses_D_fake.append(errD_fake.item())
                    val_losses_D_real.append(errD_real.item())
                    val_losses_D_fake_R.append(errD_fake_R.item())
                    val_losses_D_real_R.append(errD_real_R.item())
                    val_losses_G.append(G_loss.item())
                    val_losses_D.append(D_loss.item())
                    val_losses_D_R.append(D_loss_R.item())
                    val_style_losses.append(style_val.item())
                    val_nsgan_losses.append(nsgan_val.item())
                    val_nsgan_losses_R.append(nsgan_val_R.item())
                    val_huber_losses.append(huber_val.item())
                    val_kdh_losses.append(kd_h_val[0].item())

                    fid_distance_human.append(fid_feat_distance[0])
                    fid_distance_robot.append(fid_feat_distance[1])

            val_lossG = np.mean(val_losses_G)
            val_lossD = np.mean(val_losses_D)
            val_lossD_R = np.mean(val_losses_D_R)
            train_lossG = np.mean(train_losses_G)
            train_lossD = np.mean(train_losses_D)
            train_lossD_R = np.mean(train_losses_D_R)

            train_lossD_fake = np.mean(train_losses_D_fake)
            train_lossD_real = np.mean(train_losses_D_real)
            val_lossD_fake = np.mean(val_losses_D_fake)
            val_lossD_real = np.mean(val_losses_D_real)

            train_lossD_fake_R = np.mean(train_losses_D_fake_R)
            train_lossD_real_R = np.mean(train_losses_D_real_R)
            val_lossD_fake_R = np.mean(val_losses_D_fake_R)
            val_lossD_real_R = np.mean(val_losses_D_real_R)

            fid_human_history["epoch"+str(epoch)] = fid_distance_human
            fid_robot_history["epoch"+str(epoch)] = fid_distance_robot
            test_human_mean_fid = np.mean(fid_distance_human)
            test_robot_mean_fid = np.mean(fid_distance_robot)

            self.writer.add_scalar('Generator/Loss/train', train_lossG, epoch)
            self.writer.add_scalar('Generator/Loss/test', val_lossG, epoch)
            self.writer.add_scalar('Discriminator/Loss/train', train_lossD, epoch)
            self.writer.add_scalar('Discriminator/Loss/test', val_lossD, epoch)
            self.writer.add_scalar('Discriminator/Loss/train_R', train_lossD_R, epoch)
            self.writer.add_scalar('Discriminator/Loss/test_R', val_lossD_R, epoch)
            self.writer.add_scalar('Discriminator/Loss Fake/train', train_lossD_fake, epoch)
            self.writer.add_scalar('Discriminator/Loss Fake/train_R', train_lossD_fake_R, epoch)
            self.writer.add_scalar('Discriminator/Loss Fake/test', val_lossD_fake, epoch)
            self.writer.add_scalar('Discriminator/Loss Fake/test_R', val_lossD_fake_R, epoch)
            self.writer.add_scalar('Discriminator/Loss Real/train', train_lossD_real, epoch)
            self.writer.add_scalar('Discriminator/Loss Real/train_R', train_lossD_real_R, epoch)
            self.writer.add_scalar('Discriminator/Loss Real/test', val_lossD_real, epoch)
            self.writer.add_scalar('Discriminator/Loss Real/test_R', val_lossD_real_R, epoch)

            for tag, arr in expressive_dict["human"].items():
                if tag=="stats":
                    continue
                laban_value = ((arr-expressive_dict["robot_generated"][tag])**2).mean()
                self.writer.add_scalar('Evaluation Method/Laban Qualities/Test ' + str(tag), laban_value, epoch)
            
            self.writer.add_scalar('Evaluation Method/FID Feature Distance Mean/Test Human', test_human_mean_fid, epoch)
            self.writer.add_scalar('Evaluation Method/FID Feature Distance Mean/Test Robot', test_robot_mean_fid, epoch)
            self.writer.add_scalar('Evaluation Method/Cosine Similarity Mean/Test Human', 
                                        np.mean(torch.nn.functional.cosine_similarity(z_human.detach().cpu(), z_human_hat.detach().cpu()).numpy()), 
                                        epoch)
            self.writer.add_scalar('Evaluation Method/Cosine Similarity Mean/Test Robot', 
                                        np.mean(torch.nn.functional.cosine_similarity(z_robot.detach().cpu(),z_robot_hat.detach().cpu()).numpy()), 
                                        epoch)

            history['generator']['train'].append(train_lossG)
            history['discriminator']['train'].append(train_lossD)
            history['generator']['val'].append(val_lossG)
            history['discriminator']['val'].append(val_lossD)
            
            if epoch > self.cst["wait_epoch_num"]:
                if val_lossG < best_loss:
                    epochCount = 0
                    best_loss = val_lossG
                    best_model_wtsD = copy.deepcopy(disc_model.state_dict())
                    best_model_wtsG = copy.deepcopy(gen_model.state_dict())
                else:
                    if epoch > self.cst["wait_epoch_num"]:
                        epochCount+=1
                    else: 
                        epochCount=0

            # self.compare_models(model_a,model)
            
            # if self.cst["anneiling"]:
            #     schedulerD.step(0)
            #     schedulerG.step(0)
            # else:
            #     # Learning rate scheduler update
            #     if epoch > 50:
            #         schedulerG.step(val_lossG)
            #         schedulerD.step(val_lossD)

            # Early stop
            # if epochCount > earlyStopPatience:
            #     print('Early Stop')
            #     break

            finalTrainLossG.append(train_losses_G)
            finalTrainLossD.append(train_losses_D)
            finalValLossG.append(val_losses_G)
            finalValLossD.append(val_losses_D)

            # if epoch%25==0:
            #     plot_buf = self.save_images([loc_att_human[0], loc_att_robot[0],z_human,z_robot])
            #     image = PIL.Image.open(plot_buf)
            #     image = ToTensor()(image).squeeze(0)
            #     self.writer.add_image(f'Attention/ Attention Outputs, Epoch {epoch}', image, epoch)

            # if epoch%25==0:
            #     for qualitie in self.evaluation_method.expressive_qualities:
            #         plot_buf = self.evaluation_method.visualize_qualities_stream([value for value in expressive_dict.values()], qualitie)
            #         image = PIL.Image.open(plot_buf)
            #         image = ToTensor()(image).squeeze(0)
            #         self.writer.add_image(f'Evaluation/ Expressive Qualities/, Epoch {epoch} Qualitie{qualitie}', image, epoch)


            if epoch%10==0:
                plot_buf = self.visualize_data(seq_robot_mod.cpu().detach().numpy(),
                                    seq_human_mod.cpu().detach().numpy(), 
                                    out_hat_twist.cpu().detach().numpy(),
                                    epoch)
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).squeeze(0)
                self.writer.add_image(f'Reconstruction/ Validation, Epoch {epoch} First', image, epoch)

                plot_buf = self.visualize_data(seq_robot_mod.cpu().detach().numpy(),
                                    seq_human_mod_f2.cpu().detach().numpy(), 
                                    out_hat_twist_f2.cpu().detach().numpy(),
                                    epoch, tag="second")
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).squeeze(0)
                self.writer.add_image(f'Reconstruction/ Validation, Epoch {epoch} Second', image, epoch)

            del seq_human
            del seq_robot
            del out_hat_twist

            models_list = [gen_model, disc_model]
            for model in models_list:
                for name, param in model.named_parameters():
                    self.add_tensorboard_weight(param, epoch, name=name)
            
            self.writer.add_scalar('Generator/LR/Learning rate',self.get_lr(optimizerG), epoch)
            self.writer.add_scalar('Discriminator/LR/Learning rate',self.get_lr(optimizerD), epoch)
            
            self.writer.add_scalar('Generator/Loss/Loss variation/Train ' + self.cst["generator_loss_add"],
                                    self.cst["lambda_coeff"][2]*np.mean(train_style_losses), epoch)
            self.writer.add_scalar('Generator/Loss/Loss variation/Val ' + self.cst["generator_loss_add"],
                                    self.cst["lambda_coeff"][2]*np.mean(val_style_losses), epoch)

            self.writer.add_scalar('Generator/Loss/KLD/Train ',
                                    self.cst["lambda_coeff"][3]*np.mean(train_kdh_losses), epoch)
            self.writer.add_scalar('Generator/Loss/KLD/Val ',
                                    self.cst["lambda_coeff"][3]*np.mean(val_kdh_losses), epoch)

            self.writer.add_scalar('Generator/Loss/Huber Loss/Train',
                                    self.cst["lambda_coeff"][1]*np.mean(train_huber_losses), epoch)
            self.writer.add_scalar('Generator/Loss/Huber Loss/Val',
                                    self.cst["lambda_coeff"][1]*np.mean(val_huber_losses), epoch)
            
            self.writer.add_scalar('Generator/Loss/NSGAN/Train',
                                    self.cst["lambda_coeff"][0]*np.mean(train_nsgan_losses), epoch)
            self.writer.add_scalar('Generator/Loss/NSGAN/Val',
                                    self.cst["lambda_coeff"][0]*np.mean(val_nsgan_losses), epoch)
            self.writer.add_scalar('Generator/Loss/NSGAN/Train_R',
                                    self.cst["lambda_coeff"][0]*np.mean(train_nsgan_losses_R), epoch)
            self.writer.add_scalar('Generator/Loss/NSGAN/Val_R',
                                    self.cst["lambda_coeff"][0]*np.mean(val_nsgan_losses_R), epoch)

            # for i in range(len(fid_distance_human)):
            #     index_h = np.argmin()
            #     index_h = np.argmin()
            #     self.writer.add_scalar('Evaluation Method/FID Feature Distance/Test Human Feature Mean Feature- ' + str(i), 
            #                             np.mean(fid_distance_human[i]), epoch)
            #     self.writer.add_scalar('Evaluation Method/FID Feature Distance/Test Robot Feature Mean Feature- ' + str(i), 
            #                             np.mean(fid_distance_robot[i]), epoch)

            print(f'Epoch {epoch}: Generator loss Train: {train_lossG}, Generator loss Val: {val_lossG},\n \
                    Discriminator Real loss Train: {train_lossD_real}, Discriminator Real loss Val: {val_lossD_real},\n \
                    Discriminator Fake loss Train: {train_lossD_fake}, Discriminator Real Fake loss Val: {val_lossD_fake}, \n \
                    Discriminator Real loss Train Robot: {train_lossD_real_R}, Discriminator Real loss Val Robot: {val_lossD_real_R},\n \
                    Discriminator Fake loss Train Robot: {train_lossD_fake_R}, Discriminator Real Fake loss Val Robot: {val_lossD_fake_R}, \n \
                    Discriminator loss Train: {train_lossD}, Discriminator loss Val: {val_lossD}, \n \
                    Discriminator loss Train Robot: {train_lossD_R}, Discriminator loss Val Robot: {val_lossD_R} \n')

        # model = best_model_wts
        x = np.linspace(0,len(history['generator']['train']),len(history['generator']['train']))
        x1 = np.linspace(0,len(history['generator']['val']),len(history['generator']['val']))
        plt.plot(x,history['generator']['train'])
        plt.plot(x1,history['generator']['val'])

        plt.figure()
        x = np.linspace(0,len(history['discriminator']['train']),len(history['discriminator']['train']))
        x1 = np.linspace(0,len(history['discriminator']['val']),len(history['discriminator']['val']))
        plt.plot(x,history['discriminator']['train'])
        plt.plot(x1,history['discriminator']['val'])
        plt.show()

        with open(os.path.join(self.cst['historyPath'], "history_fretchet_human" +'.json'), 'w') as json_file:
            json.dump(fid_human_history, json_file)

        with open(os.path.join(self.cst['historyPath'], "history_fretchet_robot" +'.json'), 'w') as json_file:
            json.dump(fid_robot_history, json_file)

        best_model_wtsD = copy.deepcopy(disc_model.state_dict())
        best_model_wtsG = copy.deepcopy(gen_model.state_dict())
        optimizers = [optimizerD, optimizerG]
        best_model_wts = (best_model_wtsD, best_model_wtsG)
        finalTrainLoss = (finalTrainLossD,finalTrainLossG)
        finalValLoss = (finalValLossD, finalValLossG)
        self.saveOutput(best_model_wts,epoch,history,optimizers, best_loss, finalTrainLoss, finalValLoss, self.cst["modelKey"], multipleModels=True)
        self.writer.flush()

        return history, best_model_wts, gen_model

    def saveOutput(self, model, epoch, hisotry, optimizer,best_loss, trainLoss, valLoss, modelKey, multipleModels=False): 
        
        if multipleModels:

            torch.save({
                'epoch': epoch,
                'modelD_state_dict': model[0],
                'modelG_state_dict': model[1],
                'optimizerD_state_dict': optimizer[0].state_dict(),
                'optimizerG_state_dict': optimizer[1].state_dict(),
                'trainloss': trainLoss, 'valLoss': valLoss}, 
                os.path.join(self.cst['modelPath'], 
                    self.task_name,'model_' + modelKey + '_' + str(best_loss) + '_' + str(self.cst['modelName']) + '.pth'))
        
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model,
                'optimizer_state_dict': optimizer.state_dict(),
                'trainloss': trainLoss, 'valLoss': valLoss}, os.path.join(self.cst['modelPath'], 
                                                                self.task_name,'model_' + modelKey + '_' + str(best_loss) + '_' + str(self.cst['modelName']) + '.pth'))

        hisotry["model_config"] = self.cst["model_config"]
        hisotry["h_param"] = {"batch_size": self.cst["batch_size"],
                            "generator": self.cst["h_param"]["generator"],
                            "discriminator": self.cst["h_param"]["discriminator"]
                            }
        with open(os.path.join(self.cst['historyPath'], 
                    'model_' + modelKey + '_' + str(best_loss) + 
                        '_' + str(self.cst['modelName']) +'_history' + "." + 'json'), 'w') as json_file:
            json.dump(hisotry, json_file)

    def load_model(self, evalFlag=False):
        
        checkpoint = torch.load(self.cst["pretrained_model_path"], map_location=torch.device(self.dev))
        disc_model, gen_model = self.getModel(self.cst["model_config"])
        disc_model.load_state_dict(checkpoint['modelD_state_dict'])
        gen_model.load_state_dict(checkpoint['modelG_state_dict'])

        self.load_from_path(self.cst["pretrained_model_path_Human_VAE"],gen_model.VAE_human)
        self.load_from_path(self.cst["pretrained_model_path_Robot_VAE"],gen_model.VAE_robot)

        gen_model.VAE_human.requires_grad_(False)
        gen_model.VAE_robot.requires_grad_(False)

        optimizerD = torch.optim.Adam(disc_model.parameters(), 
                                        lr=self.cst["h_param"]["discriminator"]['learning_rate'], 
                                        weight_decay=self.cst["h_param"]["discriminator"]['L2regularizer'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        optimizerD = torch.optim.Adam(disc_model.parameters(), 
                                        lr=self.cst["h_param"]["generator"]['learning_rate'], 
                                        weight_decay=self.cst["h_param"]["generator"]['L2regularizer'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        epoch = checkpoint['epoch']
        trainLoss = checkpoint['trainloss']
        valLoss = checkpoint['valLoss']

        if evalFlag: 
            disc_model.eval()
            gen_model.eval()
            models = (disc_model,gen_model)
            for model in models:
                for p in model.parameters():
                    p.requires_grad = False
        else: 
            disc_model.train()
            gen_model.train()

        return epoch, trainLoss, valLoss, gen_model