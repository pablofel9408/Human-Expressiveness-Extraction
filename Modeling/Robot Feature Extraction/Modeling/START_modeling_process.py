from operator import mod
import os 
import io

import copy
import json
import tqdm
import datetime
import random

from scipy.interpolate import make_interp_spline, BSpline

import torch
import torchvision
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchsummary import torchsummary
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import utilities
from .utilitiesClass import utility
from .model_VAE import VAE
from .model_VAE_NLoss import NVAE
from .model_VAE_Laban_reg import NVAE_LabReg

class CustomDataset(Dataset):
    def __init__(self, input_data):
        self.data = input_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx]
        return label

class ModelingProc():
    def __init__(self, constants) -> None:
        self.cst = constants
        self.data = {}
        self.raw_data = {}

        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.task_name = self.cst["task_name"] + " " + self.cst["modelKey"] 

        if not self.cst["pretrained_model"]:
            log_dir = "logs/" + self.task_name + " logs" + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.writer = SummaryWriter(log_dir=log_dir)
            
        self.utils = utility(self.dev)
        print("Using {} device".format(self.dev))

    def set_input_data(self, input_data):
        self.raw_data = copy.deepcopy(input_data)
        self.data = input_data
        print(self.raw_data.keys())
        self.set_batches()
        print('------loading data done')
    
    def test_corr_dataloader_raw(self, key):
        correlationCoeffArr = []
        aux=0
        for i in tqdm.tqdm(self.data[key]):
            for index in range(len(i)):
                correlationCoeffArr.append(self.utils.getCorrelationCoeff(i[index].to(self.dev),torch.tensor(self.raw_data[key][aux]).to(self.dev)))
                aux+=1
        correlationCoeffArr = np.array(correlationCoeffArr)

        print(f"Dataset {key} Correlation Coeff Mean {np.mean(correlationCoeffArr)}")

    def seed_worker(self,worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def set_batches(self):
        # Create data loaders.
        g = torch.Generator()
        g.manual_seed(0)
        for key,value in self.data.items():

            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)

            self.data[key] = DataLoader(CustomDataset(value), batch_size=self.cst["batch_size"],
                                        worker_init_fn=self.seed_worker,generator=g,shuffle=False)
            self.test_corr_dataloader_raw(key)

        print('------batches done')

    def getModel(self,config, modelKey=''):

        switcher = {
                'VAE': VAE(config, device=self.dev).to(self.dev),
                'NVAE': NVAE(config, device=self.dev).to(self.dev),
                "LVAE": NVAE_LabReg(config, device=self.dev).to(self.dev)
            }

        return switcher.get(modelKey, "nothing")

    def kl_divergence(self,z,mu,log_var):

        std = torch.exp(log_var/2)
        # print("Standard deviation shape:", std.shape)
        # print("Mean shape:", mu.shape)
        # print("Sample shape:", z.shape)
        # p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        # q = torch.distributions.Normal(mu, std)
        
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        # print("Log probability in q of z given x shape:", log_qzx.shape)
        # print("Log probability of z in p shape:", log_pz.shape)

        # kl
        kl = (log_qzx - log_pz)

        # print("Log probability difference shape:", kl.shape)
        # kl = kl.squeeze(2).sum(-1)
        # kl = kl.sum(dim=(1,2))

        kl = kl.sum(dim=-1)
        # print("Kullback divergance shape:", kl.shape)
        return kl, log_qzx, log_pz

    def kl_divergence_normal_dist(self,mu,log_var):
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=-1) # to go from multi-dimensional z to single dimensional z : (batch_size x latent_size) ---> (batch_size) 
                                                                        # i.e Z = [ [z1_1, z1_2 , ...., z1_lt] ] ------> z = [ z1] 
                                                                        #         [ [z2_1, z2_2, ....., z2_lt] ]             [ z2]
                                                                        #                   .                                [ . ]
                                                                        #                   .                                [ . ]
                                                                        #         [[zn_1, zn_2, ....., zn_lt] ]              [ zn]
                                                                        
                                                                        #        lt=latent_size 
        return kl, 0, 0

    def newELBOLoss(self,z,log_var,mu,recon_loss):

        # kl, log_probs0, log_probs1 = self.kl_divergence(z,mu,log_var)
        kl, log_probs0, log_probs1 = self.kl_divergence_normal_dist(mu,log_var)
        # print("Reconstruction loss shape:", recon_loss.shape)
        # elbo = (kl - recon_loss)
        elbo = (kl - recon_loss)    
        elbo = elbo.mean()

        # if elbo < 0:
        #     aux = (kl.item(),0) if np.abs(kl.item()) > np.abs(recon_loss.item()) else (recon_loss.item(),1)
        #     dic = {0:"Recon Loss", 1:"KL"}
        #     print(f"Loss value more negative {aux[0]} is {dic[aux[1]]}")

        return elbo, kl, recon_loss, log_probs0, log_probs1 

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

    def visualize_data(self, original_tensor,pred_tensor, epoch, index=0):
        """Create a pyplot plot and save to buffer."""

        if len(np.shape(original_tensor))<3:
            original_tensor = np.expand_dims(original_tensor,axis=0)
        
        if len(np.shape(pred_tensor))<3:
            pred_tensor = np.expand_dims(pred_tensor,axis=0)

        featureDimLenght = np.shape(original_tensor)[1]
        timeTrain = np.linspace(0,len(original_tensor[0,0,:]),len(original_tensor[0,0,:]))
        timeTest = np.linspace(0,len(pred_tensor[0,0,:]),len(pred_tensor[0,0,:]))
        fig, axes = plt.subplots(2,featureDimLenght, sharex=True)

        axesColors = ['tab:blue','tab:orange', 'tab:purple', 'tab:green', 'tab:red', 'tab:pink']
        axesTitles = ['AV X', 'AV Y', 'AV Z', 'A X', 'A Y', 'A Z']

        for n,row in enumerate(axes.flat):
            if n < featureDimLenght:
                row.set_title(axesTitles[n])
                row.plot(timeTrain, original_tensor[index,n,:], axesColors[n])
                row.set_ylabel('Original Signals')
            else:
                row.set_title(axesTitles[n - featureDimLenght])
                row.plot(timeTest,pred_tensor[index,n-featureDimLenght,:], 
                            axesColors[n-featureDimLenght])
                row.set_ylabel('Network Output')
        fig.suptitle(f"Val Input Signals Reconstruction Comparisson, Index: {index}, Epoch: {epoch}")

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

    def compare_models(self,model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            print('\nModels match perfectly! :)')

    def set_deterministic_env(self):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(0)
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
    
    def training_loop_laban_qualities(self):
        """
        Trainning module for a regular AE network, losses are computed as differences between 
        inputs and the reconstruction
        """
        # self.set_deterministic_env()
        model = self.getModel(self.cst["model_config"],self.cst["modelKey"])
        model = model.to(self.dev)

        if self.cst["optimizer"]=="adam":
            print("adam")
            optimizer = torch.optim.Adam(model.parameters(), lr=self.cst['learning_rate'], 
                                            weight_decay=self.cst["L2regularizer"], amsgrad=False)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.cst['learning_rate'], 
                                            weight_decay=self.cst["L2regularizer"],momentum=0.9)

        if self.cst["anneiling"]:
            steps = 552
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        else:
            print("plateau")
            scheduler = ReduceLROnPlateau(optimizer, mode=self.cst["LRschedulerMode"], factor=self.cst["LRfactor"], 
                                        patience=self.cst["LRpatience"], verbose=True, cooldown=5)
                
        history = {'train':[], 'val':[]}

        mse_loss = torch.nn.MSELoss()

        best_model_wts = copy.deepcopy(model)

        earlyStopPatience = self.cst["earlyStopPatience"]
        n_epochs = self.cst['epochs_num']
        epochCount = 0
        best_loss = 1000000.0
        finalTrainLoss = []
        finalValLoss = []
        for epoch in tqdm.tqdm(range(1, n_epochs + 1)):

            model.train()
            train_losses = []
            correlationCoeffArrTrain = []
            for _,seq_true in enumerate(tqdm.tqdm(self.data['train'])):
                seq_true = seq_true.to(self.dev)
                seq_true_mod = seq_true.permute(0,2,1)
                optimizer.zero_grad()

                # Write the network graph at epoch 0, batch 0
                # if epoch == 1:
                    # self.writer.add_graph(model, input_to_model=seq_true_mod.float().detach(), verbose=False)
                    # self.writer.add_hparams({'bsize': self.cst["batch_size"], 'l2_reg': self.cst["L2regularizer"]},
                    #                 {'hparam/accuracy': val_loss, 'hparam/loss': train_loss})

                model.encoder.init_hidden(len(seq_true_mod))
                model.decoder.init_hidden(len(seq_true_mod))

                seq_pred, z, _, mu, log_var, recon_loss, laban_output = model(seq_true_mod.float())
                loss, kl, recon, log_probs0, log_probs1 = self.newELBOLoss(z,log_var,mu,recon_loss)
                loss_mse_mu_dec = torch.nn.functional.mse_loss(seq_pred,seq_true_mod).item()
                loss_mse_qualities = mse_loss(laban_output)
                # loss_mse_out_samp = torch.nn.functional.mse_loss(out_samp,seq_true_mod).float()
                                
                # loss = loss_mse + 0.001*loss

                # print(seq_pred.size())
                # print(seq_true.size())
                # sys.exit()

                # l1_lambda = 0.001
                # l1_norm = sum(p.abs().sum() for p in model.parameters())

                # loss = loss + l1_lambda * l1_norm
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # if model_a.__str__() == model_b.__str__():
                #     print("Network not updating.")
                #     break

                # if loss_mse < 0.1:
                #     plt.plot(seq_pred.permute(0,2,1)[0,:,:].detach().cpu().numpy(),label='data')
                #     plt.plot(seq_true_mod.permute(0,2,1)[0,:,:].cpu().numpy(), label="ref")
                #     plt.legend(loc="upper left")
                #     plt.show()
                for i in range(seq_pred.shape[0]):
                    correlationCoeffArrTrain.append(self.utils.getCorrelationCoeff(seq_pred[i].cpu().detach(),
                                                                                    seq_true_mod[i].cpu().detach()))

                train_losses.append(loss.item())
                # print(f'\nBatch {epoch}: train loss {loss.item()}\n')

                if self.cst["anneiling"]:
                    scheduler.step()
                    
            val_losses = []
            correlationCoeffArrTest = []
            min_corr_coeff=1000
            max_corr_coeff=-1000
            model.eval()
            with torch.no_grad():
                for _,seq_true in enumerate(tqdm.tqdm(self.data['val'])):
                    seq_true = seq_true.to(self.dev)
                    seq_true_mod = seq_true.permute(0,2,1)
                    model.encoder.init_hidden(len(seq_true_mod))
                    seq_pred, z, _, mu, log_var, recon_loss = model(seq_true_mod.float())
                    loss, _, _, _, _ = self.newELBOLoss(z,log_var,mu,recon_loss)
                    val_losses.append(loss.item())

                    for i in range(seq_pred.shape[0]):
                        corr_value = self.utils.getCorrelationCoeff(seq_pred[i],seq_true_mod[i])
                        correlationCoeffArrTest.append(corr_value)

                        if min_corr_coeff>corr_value:
                            min_seq_corr=seq_true_mod[i]
                            min_seq_pred=seq_pred[i]
                            min_corr_coeff=corr_value

                        if max_corr_coeff<corr_value:
                            max_seq_corr=seq_true_mod[i]
                            max_seq_pred=seq_pred[i]
                            max_corr_coeff=corr_value
                        
                    # print(f'\nBatch {epoch}: val loss {loss.item()}\n')
            model_a = copy.deepcopy(model)
            correlationCoeffArrTrain = np.array(correlationCoeffArrTrain)
            correlationCoeffArrTest = np.array(correlationCoeffArrTest)
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/test', val_loss, epoch)

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            if val_loss < best_loss:
                epochCount = 0
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                epochCount+=1

            # self.compare_models(model_a,model)
            
            if self.cst["anneiling"]:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
            else:
                # Learning rate scheduler update
                scheduler.step(val_loss)

            # Early stop
            if epochCount > earlyStopPatience:
                print('Early Stop')
                break

            finalTrainLoss.append(train_losses)
            finalValLoss.append(val_loss)

            if epoch%25==0:
                plot_buf = self.visualize_data(seq_true_mod.cpu().detach().numpy(), 
                                    seq_pred.cpu().detach().numpy(),
                                    epoch)
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).squeeze(0)
                self.writer.add_image(f'Reconstruction Validation, Epoch {epoch}', image, epoch)

                plot_buf = self.visualize_data(max_seq_corr.cpu().detach().numpy(), 
                                    max_seq_pred.cpu().detach().numpy(),
                                    epoch)
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).squeeze(0)
                self.writer.add_image(f'Reconstruction Validation Max Corr, Epoch {epoch}', image, epoch)

                plot_buf = self.visualize_data(min_seq_corr.cpu().detach().numpy(), 
                                    min_seq_pred.cpu().detach().numpy(),
                                    epoch)
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).squeeze(0)
                self.writer.add_image(f'Reconstruction Validation Min Corr, Epoch {epoch}', image, epoch)

                # plot_buf = self.visualize_data(seq_true_mod.cpu().detach().numpy(), 
                #                     out_samp.cpu().detach().numpy(),
                #                     epoch)
                # image = PIL.Image.open(plot_buf)
                # image = ToTensor()(image).squeeze(0)
                # self.writer.add_image(f'Reconstruction Out Sampled Validation, Epoch {epoch}', image, epoch)

            del seq_true
            del seq_pred

            corrtest, corrtrain = self.visualize_output(model,model_a,train=True)
            # plt.bar(np.arange(len(corrtest)),corrtest)
            # plt.bar(np.arange(len(correlationCoeffArrTest)),correlationCoeffArrTest,color='red')
            # plt.show()
            # plt.bar(np.arange(len(correlationCoeffArrTest)),correlationCoeffArrTest - corrtest,color='red')
            # plt.show()
            # plt.bar(np.arange(len(corrtrain)),corrtrain)
            # plt.bar(np.arange(len(correlationCoeffArrTrain)),correlationCoeffArrTrain,color='red')
            # plt.show()
            # plt.bar(np.arange(len(correlationCoeffArrTrain)),correlationCoeffArrTrain - corrtrain,color='red')
            # plt.show()
            # torch.cuda.empty_cache()

            for name, param in model.named_parameters():
                self.add_tensorboard_weight(param, epoch, name=name)
            
            self.writer.add_scalar('LR/Learning rate',self.get_lr(optimizer), epoch)
            # self.writer.add_scalar('Prob/Log Prob P(Z)', log_probs0.sum(dim=-1).mean(), epoch)
            # self.writer.add_scalar('Prob/Log Prob Q(Z|X)', log_probs1.sum(dim=-1).mean(), epoch)
            self.writer.add_scalar('Prob/KL Loss Value', kl.mean(), epoch)
            self.writer.add_scalar('Prob/Reconstruction Prob Value', recon.mean(), epoch)

            self.writer.add_scalar('MSE/MSE Loss Mu Dec', loss_mse_mu_dec, epoch)
            # self.writer.add_scalar('MSE/MSE Loss Out Sampled', loss_mse_out_samp, epoch)

            self.writer.add_scalar("Correlation Train/Corr Train Between -0.1 and 0.1", 
                                        len(correlationCoeffArrTrain[(correlationCoeffArrTrain<0.1) \
                                                                & (correlationCoeffArrTrain>-0.1)]),epoch)
            self.writer.add_scalar("Correlation Train/Corr Train > 0.1", 
                                        len(correlationCoeffArrTrain[(correlationCoeffArrTrain>0.1)]),epoch)
            self.writer.add_scalar("Correlation Train/Corr Train < -0.1", 
                                        len(correlationCoeffArrTrain[(correlationCoeffArrTrain<-0.1)]),epoch)
            
            self.writer.add_scalar("Correlation Train/Corr Train Evaluation 2 Between -0.1 and 0.1", 
                                        len(corrtrain[(corrtrain<0.1) \
                                                                & (corrtrain>-0.1)]),epoch)
            self.writer.add_scalar("Correlation Train/Corr Train Evaluation 2 > 0.1", 
                                        len(corrtrain[(corrtrain>0.1)]),epoch)
            self.writer.add_scalar("Correlation Train/Corr Train Evaluation 2 < -0.1", 
                                        len(corrtrain[(corrtrain<-0.1)]),epoch)

            self.writer.add_scalar("Correlation Test/Corr Test Evaluation 2 Between -0.1 and 0.1", 
                                        len(corrtest[(corrtest<0.1) \
                                                                & (corrtest>-0.1)]),epoch)
            self.writer.add_scalar("Correlation Test/Corr Test Evaluation 2 > 0.1", 
                                        len(corrtest[(corrtest>0.1)]),epoch)
            self.writer.add_scalar("Correlation Test/Corr Test Evaluation 2 < -0.1", 
                                        len(corrtest[(corrtest<-0.1)]),epoch)

            # if len(correlationCoeffArrTest[(correlationCoeffArrTest<0.1) \
            #         & (correlationCoeffArrTest>-0.1)])<100 and len(correlationCoeffArrTrain[(correlationCoeffArrTrain<0.1) \
            #             & (correlationCoeffArrTrain>-0.1)])<100 and np.mean(correlationCoeffArrTrain)>0.7:
            #     break

            print(f"MSE Loss Mu Dec: {loss_mse_mu_dec}")
            # print(f"MSE Loss Out Sampled: {loss_mse_out_samp}")
            # print(f"Log Prob P(Z): {log_probs0.sum(dim=-1)}")
            # print(f"Log Prob Q(Z|X): {log_probs1.sum(dim=-1)}")
            print(f"KL Loss Value {kl}")
            print(f"Reconstruction Prob Value {recon}")
            print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

        self.writer.add_hparams({'bsize': self.cst["batch_size"], 'l2_reg': self.cst["L2regularizer"]},
                                    {'hparam/accuracy': val_loss, 'hparam/loss': train_loss})

        # model = best_model_wts
        x = np.linspace(0,len(history['train']),len(history['train']))
        x1 = np.linspace(0,len(history['val']),len(history['val']))
        plt.plot(x,history['train'])
        plt.plot(x1,history['val'])
        plt.show()

        history['correlation_coeff_train'] = np.mean(correlationCoeffArrTrain)
        history['correlation_coeff_test'] = np.mean(corrtest)

        self.saveOutput(best_model_wts,epoch,history,optimizer, finalTrainLoss, finalValLoss, self.cst["modelKey"])
        self.writer.flush()

        return history, best_model_wts, model
    
    def training_loop(self):
        """
        Trainning module for a regular AE network, losses are computed as differences between 
        inputs and the reconstruction
        """
        # self.set_deterministic_env()
        model = self.getModel(self.cst["model_config"],self.cst["modelKey"])
        model = model.to(self.dev)

        if self.cst["optimizer"]=="adam":
            print("adam")
            optimizer = torch.optim.Adam(model.parameters(), lr=self.cst['learning_rate'], 
                                            weight_decay=self.cst["L2regularizer"], amsgrad=False)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.cst['learning_rate'], 
                                            weight_decay=self.cst["L2regularizer"],momentum=0.9)

        if self.cst["anneiling"]:
            steps = 552
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        else:
            print("plateau")
            scheduler = ReduceLROnPlateau(optimizer, mode=self.cst["LRschedulerMode"], factor=self.cst["LRfactor"], 
                                        patience=self.cst["LRpatience"], verbose=True, cooldown=5)
                
        history = {'train':[], 'val':[]}

        mse_loss = torch.nn.MSELoss()

        best_model_wts = copy.deepcopy(model)

        earlyStopPatience = self.cst["earlyStopPatience"]
        n_epochs = self.cst['epochs_num']
        epochCount = 0
        best_loss = 1000000.0
        finalTrainLoss = []
        finalValLoss = []
        for epoch in tqdm.tqdm(range(1, n_epochs + 1)):

            model.train()
            train_losses = []
            correlationCoeffArrTrain = []
            for _,seq_true in enumerate(tqdm.tqdm(self.data['train'])):
                seq_true = seq_true.to(self.dev)
                seq_true_mod = seq_true.permute(0,2,1)
                optimizer.zero_grad()

                # Write the network graph at epoch 0, batch 0
                # if epoch == 1:
                    # self.writer.add_graph(model, input_to_model=seq_true_mod.float().detach(), verbose=False)
                    # self.writer.add_hparams({'bsize': self.cst["batch_size"], 'l2_reg': self.cst["L2regularizer"]},
                    #                 {'hparam/accuracy': val_loss, 'hparam/loss': train_loss})

                model.encoder.init_hidden(len(seq_true_mod))
                model.decoder.init_hidden(len(seq_true_mod))

                seq_pred, z, _, mu, log_var, recon_loss = model(seq_true_mod.float())
                loss, kl, recon, log_probs0, log_probs1 = self.newELBOLoss(z,log_var,mu,recon_loss)
                loss_mse_mu_dec = torch.nn.functional.mse_loss(seq_pred,seq_true_mod).item()
                # loss_mse_out_samp = torch.nn.functional.mse_loss(out_samp,seq_true_mod).float()
                                
                # loss = loss_mse + 0.001*loss

                # print(seq_pred.size())
                # print(seq_true.size())
                # sys.exit()

                # l1_lambda = 0.001
                # l1_norm = sum(p.abs().sum() for p in model.parameters())

                # loss = loss + l1_lambda * l1_norm
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # if model_a.__str__() == model_b.__str__():
                #     print("Network not updating.")
                #     break

                # if loss_mse < 0.1:
                #     plt.plot(seq_pred.permute(0,2,1)[0,:,:].detach().cpu().numpy(),label='data')
                #     plt.plot(seq_true_mod.permute(0,2,1)[0,:,:].cpu().numpy(), label="ref")
                #     plt.legend(loc="upper left")
                #     plt.show()
                for i in range(seq_pred.shape[0]):
                    correlationCoeffArrTrain.append(self.utils.getCorrelationCoeff(seq_pred[i].cpu().detach(),
                                                                                    seq_true_mod[i].cpu().detach()))

                train_losses.append(loss.item())
                # print(f'\nBatch {epoch}: train loss {loss.item()}\n')

                if self.cst["anneiling"]:
                    scheduler.step()
                    
            val_losses = []
            correlationCoeffArrTest = []
            min_corr_coeff=1000
            max_corr_coeff=-1000
            model.eval()
            with torch.no_grad():
                for _,seq_true in enumerate(tqdm.tqdm(self.data['val'])):
                    seq_true = seq_true.to(self.dev)
                    seq_true_mod = seq_true.permute(0,2,1)
                    model.encoder.init_hidden(len(seq_true_mod))
                    seq_pred, z, _, mu, log_var, recon_loss = model(seq_true_mod.float())
                    loss, _, _, _, _ = self.newELBOLoss(z,log_var,mu,recon_loss)
                    val_losses.append(loss.item())

                    for i in range(seq_pred.shape[0]):
                        corr_value = self.utils.getCorrelationCoeff(seq_pred[i],seq_true_mod[i])
                        correlationCoeffArrTest.append(corr_value)

                        if min_corr_coeff>corr_value:
                            min_seq_corr=seq_true_mod[i]
                            min_seq_pred=seq_pred[i]
                            min_corr_coeff=corr_value

                        if max_corr_coeff<corr_value:
                            max_seq_corr=seq_true_mod[i]
                            max_seq_pred=seq_pred[i]
                            max_corr_coeff=corr_value
                        
                    # print(f'\nBatch {epoch}: val loss {loss.item()}\n')
            model_a = copy.deepcopy(model)
            correlationCoeffArrTrain = np.array(correlationCoeffArrTrain)
            correlationCoeffArrTest = np.array(correlationCoeffArrTest)
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/test', val_loss, epoch)

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            if val_loss < best_loss:
                epochCount = 0
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            else:
                epochCount+=1

            # self.compare_models(model_a,model)
            
            if self.cst["anneiling"]:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
            else:
                # Learning rate scheduler update
                scheduler.step(val_loss)

            # Early stop
            if epochCount > earlyStopPatience:
                print('Early Stop')
                break

            finalTrainLoss.append(train_losses)
            finalValLoss.append(val_loss)

            if epoch%25==0:
                plot_buf = self.visualize_data(seq_true_mod.cpu().detach().numpy(), 
                                    seq_pred.cpu().detach().numpy(),
                                    epoch)
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).squeeze(0)
                self.writer.add_image(f'Reconstruction Validation, Epoch {epoch}', image, epoch)

                plot_buf = self.visualize_data(max_seq_corr.cpu().detach().numpy(), 
                                    max_seq_pred.cpu().detach().numpy(),
                                    epoch)
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).squeeze(0)
                self.writer.add_image(f'Reconstruction Validation Max Corr, Epoch {epoch}', image, epoch)

                plot_buf = self.visualize_data(min_seq_corr.cpu().detach().numpy(), 
                                    min_seq_pred.cpu().detach().numpy(),
                                    epoch)
                image = PIL.Image.open(plot_buf)
                image = ToTensor()(image).squeeze(0)
                self.writer.add_image(f'Reconstruction Validation Min Corr, Epoch {epoch}', image, epoch)

                # plot_buf = self.visualize_data(seq_true_mod.cpu().detach().numpy(), 
                #                     out_samp.cpu().detach().numpy(),
                #                     epoch)
                # image = PIL.Image.open(plot_buf)
                # image = ToTensor()(image).squeeze(0)
                # self.writer.add_image(f'Reconstruction Out Sampled Validation, Epoch {epoch}', image, epoch)

            del seq_true
            del seq_pred

            corrtest, corrtrain = self.visualize_output(model,model_a,train=True)
            # plt.bar(np.arange(len(corrtest)),corrtest)
            # plt.bar(np.arange(len(correlationCoeffArrTest)),correlationCoeffArrTest,color='red')
            # plt.show()
            # plt.bar(np.arange(len(correlationCoeffArrTest)),correlationCoeffArrTest - corrtest,color='red')
            # plt.show()
            # plt.bar(np.arange(len(corrtrain)),corrtrain)
            # plt.bar(np.arange(len(correlationCoeffArrTrain)),correlationCoeffArrTrain,color='red')
            # plt.show()
            # plt.bar(np.arange(len(correlationCoeffArrTrain)),correlationCoeffArrTrain - corrtrain,color='red')
            # plt.show()
            # torch.cuda.empty_cache()

            for name, param in model.named_parameters():
                self.add_tensorboard_weight(param, epoch, name=name)
            
            self.writer.add_scalar('LR/Learning rate',self.get_lr(optimizer), epoch)
            # self.writer.add_scalar('Prob/Log Prob P(Z)', log_probs0.sum(dim=-1).mean(), epoch)
            # self.writer.add_scalar('Prob/Log Prob Q(Z|X)', log_probs1.sum(dim=-1).mean(), epoch)
            self.writer.add_scalar('Prob/KL Loss Value', kl.mean(), epoch)
            self.writer.add_scalar('Prob/Reconstruction Prob Value', recon.mean(), epoch)

            self.writer.add_scalar('MSE/MSE Loss Mu Dec', loss_mse_mu_dec, epoch)
            # self.writer.add_scalar('MSE/MSE Loss Out Sampled', loss_mse_out_samp, epoch)

            self.writer.add_scalar("Correlation Train/Corr Train Between -0.1 and 0.1", 
                                        len(correlationCoeffArrTrain[(correlationCoeffArrTrain<0.1) \
                                                                & (correlationCoeffArrTrain>-0.1)]),epoch)
            self.writer.add_scalar("Correlation Train/Corr Train > 0.1", 
                                        len(correlationCoeffArrTrain[(correlationCoeffArrTrain>0.1)]),epoch)
            self.writer.add_scalar("Correlation Train/Corr Train < -0.1", 
                                        len(correlationCoeffArrTrain[(correlationCoeffArrTrain<-0.1)]),epoch)
            
            self.writer.add_scalar("Correlation Train/Corr Train Evaluation 2 Between -0.1 and 0.1", 
                                        len(corrtrain[(corrtrain<0.1) \
                                                                & (corrtrain>-0.1)]),epoch)
            self.writer.add_scalar("Correlation Train/Corr Train Evaluation 2 > 0.1", 
                                        len(corrtrain[(corrtrain>0.1)]),epoch)
            self.writer.add_scalar("Correlation Train/Corr Train Evaluation 2 < -0.1", 
                                        len(corrtrain[(corrtrain<-0.1)]),epoch)

            self.writer.add_scalar("Correlation Test/Corr Test Evaluation 2 Between -0.1 and 0.1", 
                                        len(corrtest[(corrtest<0.1) \
                                                                & (corrtest>-0.1)]),epoch)
            self.writer.add_scalar("Correlation Test/Corr Test Evaluation 2 > 0.1", 
                                        len(corrtest[(corrtest>0.1)]),epoch)
            self.writer.add_scalar("Correlation Test/Corr Test Evaluation 2 < -0.1", 
                                        len(corrtest[(corrtest<-0.1)]),epoch)

            # if len(correlationCoeffArrTest[(correlationCoeffArrTest<0.1) \
            #         & (correlationCoeffArrTest>-0.1)])<100 and len(correlationCoeffArrTrain[(correlationCoeffArrTrain<0.1) \
            #             & (correlationCoeffArrTrain>-0.1)])<100 and np.mean(correlationCoeffArrTrain)>0.7:
            #     break

            print(f"MSE Loss Mu Dec: {loss_mse_mu_dec}")
            # print(f"MSE Loss Out Sampled: {loss_mse_out_samp}")
            # print(f"Log Prob P(Z): {log_probs0.sum(dim=-1)}")
            # print(f"Log Prob Q(Z|X): {log_probs1.sum(dim=-1)}")
            print(f"KL Loss Value {kl}")
            print(f"Reconstruction Prob Value {recon}")
            print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

        self.writer.add_hparams({'bsize': self.cst["batch_size"], 'l2_reg': self.cst["L2regularizer"]},
                                    {'hparam/accuracy': val_loss, 'hparam/loss': train_loss})

        # model = best_model_wts
        x = np.linspace(0,len(history['train']),len(history['train']))
        x1 = np.linspace(0,len(history['val']),len(history['val']))
        plt.plot(x,history['train'])
        plt.plot(x1,history['val'])
        plt.show()

        history['correlation_coeff_train'] = np.mean(correlationCoeffArrTrain)
        history['correlation_coeff_test'] = np.mean(corrtest)

        self.saveOutput(best_model_wts,epoch,history,optimizer, finalTrainLoss, finalValLoss, self.cst["modelKey"])
        self.writer.flush()

        return history, best_model_wts, model

    def visualize_output(self, model,aa=None, train=False):
        
        if aa is None:
            aa = copy.deepcopy(model)

        trainDataTensor = self.raw_data['train']
        testDataTensor = self.raw_data['val']
        if isinstance(trainDataTensor, np.ndarray) or isinstance(testDataTensor, np.ndarray):
            trainDataTensor = torch.tensor(trainDataTensor)
            testDataTensor = torch.tensor(testDataTensor)

        trainDataTensor = trainDataTensor.permute(0,2,1)
        testDataTensor = testDataTensor.permute(0,2,1)

        model.eval()
        aa.eval()
        # with torch.no_grad():
        self.compare_models(aa,model)
        # trainDataTensor, testDataTensor = (trainData,testData)
        model.encoder.init_hidden(len(testDataTensor))
        # utilities.saliency(testDataTensor,model)
        correlationCoeffArr = []
        print("-------------Analyzing Test Data-------------------")
        # for _,testDataTensor in enumerate(tqdm.tqdm(self.data['val'])):
        # testDataTensor = testDataTensor.permute(0,2,1)
        predictions,latentSignal,_,_,_,_= model(testDataTensor.float().to(self.dev))
        # latentSignal = latentSignal.squeeze(dim=2)
        # Calculate test dataset correlation coefficient
        testDataTensor = testDataTensor.to(self.dev)
        for i in range(testDataTensor.shape[0]):
            correlationCoeffArr.append(self.utils.getCorrelationCoeff(predictions[i],testDataTensor[i]))
        correlationCoeffArr1 = np.array(correlationCoeffArr) 

        # Plot best and worgs signal latent space as both signal and bars
        print("Latent signal shape:", latentSignal.shape)

        # plt.plot(latentSignal[0,:].detach().cpu().numpy())
        # plt.show()
        # utilities.simpleViszation(latentSignal.cpu().detach().numpy(), 'Latent Signal ', index=np.argmin(correlationCoeffArr))
        # utilities.simpleViszation(latentSignal.cpu().detach().numpy(), 'Latent Signal ', index=np.argmax(correlationCoeffArr))
        # utilities.visualizeBarWeights(latentSignal.cpu().detach().numpy(), 'Latent Embedding Signal ', index=np.argmin(correlationCoeffArr))
        # utilities.visualizeBarWeights(latentSignal.cpu().detach().numpy(), 'Latent Embedding Signal ', index=np.argmax(correlationCoeffArr))

        print("Predictions signal shape:", predictions.shape)
        print("Test signal shape:", testDataTensor.shape)
        print("Mean Correlation coefficient test data:", np.mean(correlationCoeffArr1))
        print("Min Correlation coefficient test data:", np.min(correlationCoeffArr1))
        print("Max Correlation coefficient test data:", np.max(correlationCoeffArr1))
        print("Argmin Correlation coefficient test data:", np.argmin(correlationCoeffArr1))
        print("Argmax Correlation coefficient test data:", np.argmax(correlationCoeffArr1))
        print("Percentage above 0.1  test data:", len(correlationCoeffArr1[correlationCoeffArr1>0.1]))
        print("Percentage below -0.1 test data:", len(correlationCoeffArr1[correlationCoeffArr1<-0.1]))
        print("Percentage between -0.1 and 0.1 test data:", len(correlationCoeffArr1[(correlationCoeffArr1<0.1) & (correlationCoeffArr1>-0.1)]))
        print("Uncorrelated test data:", len(correlationCoeffArr1[correlationCoeffArr1==0]))

        # Plot best and worst samples from the test data 
        # from scipy.interpolate import interp1d
        # import matplotlib.pyplot as plt

        # T = testDataTensor[np.argmax(correlationCoeffArr1),:,:].permute(1,0).cpu().detach().numpy()
        # print(np.shape(T))
        # x = np.linspace(0,np.shape(T)[0],np.shape(T)[0])
        # cubic_interploation_model = []
        # for i  in range(np.shape(T)[1]):   
        #     cubic_interploation_model.append(interp1d(x, T[:,i], kind = "cubic"))
        
        # Plotting the Graph
        # plt.figure(figsize=(100,100))
        # X_ = np.linspace(x.min(), x.max(), 500)
        # for j in range(np.shape(T)[1]):
        #     Y_ = cubic_interploation_model[j](X_)
        #     plt.plot(Y_, color="white",linewidth=11.0)

        # plt.tight_layout()
        # plt.savefig('demo.png', transparent=True)
        # plt.show()

        testDataTensor = testDataTensor.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()

        if not train:
            self.utils.visualizeData(np.argmax(correlationCoeffArr1),testDataTensor,predictions, title="Input Signals, Index: ")
            self.utils.visualizeData(np.argmin(correlationCoeffArr1),testDataTensor,predictions, title="Input Signals, Index: ")
            plt.show()
            torch.cuda.empty_cache()

            # Plot correlation coefficents for each test data sample
            plt.bar(np.arange(len(correlationCoeffArr1)),correlationCoeffArr1)
            plt.show()

        # Calculate correlation coefficient for training data
        print("-------------Analyzing Train Data-------------------")
        model.encoder.init_hidden(len(trainDataTensor))
        correlationCoeffArr = []
        # for _,trainDataTensor in enumerate(tqdm.tqdm(self.data['train'])):
        # trainDataTensor = trainDataTensor.permute(0,2,1)
        predictions,latentSignal,_,_,_,_= model(trainDataTensor.float().to(self.dev))
        trainDataTensor = trainDataTensor.to(self.dev)
        for i in range(trainDataTensor.shape[0]):
            correlationCoeffArr.append(self.utils.getCorrelationCoeff(predictions[i],trainDataTensor[i]))
        correlationCoeffArr2 = np.array(correlationCoeffArr) 

        print("Predictions signal shape:", predictions.shape)
        print("Test signal shape:", trainDataTensor.shape)
        print("Mean Correlation coefficient training data:", np.mean(correlationCoeffArr2))
        print("Min Correlation coefficient training data:", np.min(correlationCoeffArr2))
        print("Max Correlation coefficient training data:", np.max(correlationCoeffArr2))
        print("Argmin Correlation coefficient training data:", np.argmin(correlationCoeffArr2))
        print("Argmax Correlation coefficient training data:", np.argmax(correlationCoeffArr2))
        print("Percentage above 0.1  training data:", len(correlationCoeffArr2[correlationCoeffArr2>0.1]))
        print("Percentage below -0.1 training data:", len(correlationCoeffArr2[correlationCoeffArr2<-0.1]))
        print("Percentage between -0.1 and 0.1 training data:", len(correlationCoeffArr2[(correlationCoeffArr2<0.1) & (correlationCoeffArr2>-0.1)]))
        print("Uncorrelated training data:", len(correlationCoeffArr2[correlationCoeffArr2==0]))

        trainDataTensor = trainDataTensor.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()

        if not train:
            self.utils.visualizeData(np.argmax(correlationCoeffArr2),trainDataTensor,predictions, title="Input Signals, Index: ")
            self.utils.visualizeData(np.argmin(correlationCoeffArr2),trainDataTensor,predictions, title="Input Signals, Index: ")

            #Plot corrleation coefficient as bar plot for each training dataset sample
            plt.figure(10)
            timeAxis = np.linspace(-1, 1,20)
            plt.bar(np.arange(len(correlationCoeffArr2)),correlationCoeffArr2)
            plt.show()
            torch.cuda.empty_cache()

        return correlationCoeffArr1, correlationCoeffArr2

    def saveOutput(self, model, epoch, hisotry, optimizer, trainLoss, valLoss, modelKey, multipleModels=False): 
        
        if multipleModels:

            torch.save({
                'epoch': epoch,
                'modelAE_state_dict': model[0].state_dict(),
                'modelLaban_state_dict': model[1].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainloss': trainLoss, 'valLoss': valLoss}, os.path.join(self.cst['modelPath'], 
                                                                'model_' + modelKey + '_' + str(self.cst['modelName']) + '.pth'))
        
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model,
                'optimizer_state_dict': optimizer.state_dict(),
                'trainloss': trainLoss, 'valLoss': valLoss}, os.path.join(self.cst['modelPath'], 
                                                                self.task_name,'model_' + modelKey + '_' + str(valLoss[-1]) + '_' + str(self.cst['modelName']) + '.pth'))

        with open(os.path.join(self.cst['historyPath'], 
                    'model_' + modelKey + '_' + str(valLoss[-1]) + 
                        '_' + str(self.cst['modelName']) +'_history' + "." + 'json'), 'w') as json_file:
            json.dump(hisotry, json_file)

    def load_model(self, evalFlag=False):
        
        checkpoint = torch.load(self.cst["pretrained_model_path"], map_location=torch.device(self.dev))
        model = self.getModel(self.cst["model_config"],self.cst["modelKey"])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-7, weight_decay=self.cst['L2regularizer'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        trainLoss = checkpoint['trainloss']
        valLoss = checkpoint['valLoss']

        if evalFlag: 
            model.eval()
        else: 
            model.train()

        for p in model.parameters():
            p.requires_grad = False

        return epoch, trainLoss, valLoss, model
