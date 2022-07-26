import os 
import sys 

import copy
import json
import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utilities
from .utilitiesClass import utility
from .model_VAE import VAE
from .model_VAE_NLoss import NVAE

class ModelingProc():
    def __init__(self, constants) -> None:
        self.cst = constants
        self.data = {}
        self.raw_data = {}

        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.utils = utility(self.dev)
        print("Using {} device".format(self.dev))

    def set_input_data(self, input_data):
        self.data = input_data
        self.raw_data = copy.deepcopy(input_data)
        self.set_batches()
        print('------loading data done')

    def set_batches(self):
        # Create data loaders.
        for key,value in self.data.items():
            self.data[key] = DataLoader(torch.tensor(value), batch_size=self.cst["batch_size"])
        print('------batches done')

    def getModel(self,config, modelKey=''):

        switcher = {
                'VAE':VAE(config).to(self.dev),
                'NVAE':NVAE(config).to(self.dev),
                "AE": ''
            }

        return switcher.get(modelKey, "nothing")

    def kl_divergence(self,z,mu,log_var):

        std = torch.exp(log_var/2)
        # print("Standard deviation shape:", std.shape)
        # print("Mean shape:", mu.shape)
        # print("Sample shape:", z.shape)
        # p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        # q = torch.distributions.Normal(mu, std)
        
        p = torch.distributions.Independent(torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std)),1)
        q = torch.distributions.Independent(torch.distributions.Normal(mu, std),1)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        # print("Log probability in q of z given x shape:", log_qzx.shape)
        # print("Log probability of z in p shape:", log_pz.shape)

        # kl
        print(f"Log Prob P(Z): {log_pz.sum(dim=-1)}")
        print(f"Log Prob Q(Z|X): {log_qzx.sum(dim=-1)}")
        kl = (log_qzx - log_pz)

        # print("Log probability difference shape:", kl.shape)
        # kl = kl.squeeze(2).sum(-1)
        # kl = kl.sum(dim=(1,2))

        kl = kl.sum(dim=-1)
        # print("Kullback divergance shape:", kl.shape)
        return kl

    def newELBOLoss(self,z,log_var,mu,recon_loss):
        kl = self.kl_divergence(z,mu,log_var)
        # print("Reconstruction loss shape:", recon_loss.shape)
        # elbo = (kl - recon_loss)
        elbo = (kl - recon_loss)    
        print(f"KL Loss Value {kl}")
        print(f"Reconstruction Prob Value {recon_loss}")
        elbo = elbo.mean()

        # if elbo < 0:
        #     aux = (kl.item(),0) if np.abs(kl.item()) > np.abs(recon_loss.item()) else (recon_loss.item(),1)
        #     dic = {0:"Recon Loss", 1:"KL"}
        #     print(f"Loss value more negative {aux[0]} is {dic[aux[1]]}")

        return elbo  

    def training_loop(self):
        """
        Trainning module for a regular AE network, losses are computed as differences between 
        inputs and the reconstruction
        """
        model = self.getModel(self.cst["model_config"],self.cst["modelKey"])
        model = model.to(self.dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cst['learning_rate'], 
                                        weight_decay=self.cst["L2regularizer"])
        history = {'train':[], 'val':[]}

        mse_loss = torch.nn.MSELoss()

        best_model_wts = copy.deepcopy(model)
        scheduler = ReduceLROnPlateau(optimizer, mode=self.cst["LRschedulerMode"], factor=self.cst["LRfactor"], 
                                        patience=self.cst["LRpatience"], verbose=True)

        earlyStopPatience = self.cst["earlyStopPatience"]
        n_epochs = self.cst['epochs_num']
        epochCount = 0
        best_loss = 1000000.0
        finalTrainLoss = []
        finalValLoss = []

        for epoch in tqdm.tqdm(range(1, n_epochs + 1)):
            model = model.train()

            train_losses = []
            for seq_true in tqdm.tqdm(self.data['train']):

                seq_true = seq_true.to(self.dev)
                seq_true_mod = seq_true.permute(0,2,1)

                optimizer.zero_grad()

                seq_pred, z, _, mu, log_var, recon_loss = model(seq_true_mod.float())
                loss = self.newELBOLoss(z,log_var,mu,recon_loss)
                loss_mse = torch.nn.functional.mse_loss(seq_pred,seq_true_mod).float()
                print(f"MSE Loss: {loss_mse}")
                loss = loss_mse + 0.001*loss

                # print(seq_pred.size())
                # print(seq_true.size())
                # sys.exit()
                if loss_mse < 0.1:
                    plt.plot(seq_pred.permute(0,2,1)[0,:,:].cpu().numpy(),label='data')
                    plt.plot(seq_true_mod.permute(0,2,1)[0,:,:].cpu().numpy(), label="ref")
                    plt.legend(loc="upper left")
                    plt.show()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

                train_losses.append(loss.item())
                print(f'\nBatch {epoch}: train loss {loss.item()}\n')

            val_losses = []
            model = model.eval()
            with torch.no_grad():
                for seq_true in tqdm.tqdm(self.data['val']):

                    seq_true = seq_true.to(self.dev)
                    seq_true_mod = seq_true.permute(0,2,1)
                    seq_pred, z, _, mu, log_var, recon_loss = model(seq_true_mod.float())
                    loss = self.newELBOLoss(z,log_var,mu,recon_loss)
                    val_losses.append(loss.item())
                    print(f'\nBatch {epoch}: val loss {loss.item()}\n')
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model)
                epochCount = 0
            else:
                epochCount+=1

            # Learning rate scheduler update
            scheduler.step(val_loss)

            # Early stop
            if epochCount > earlyStopPatience:
                break

            finalTrainLoss.append(train_losses)
            finalValLoss.append(val_loss)

            del seq_true
            del seq_pred
            torch.cuda.empty_cache()

            print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

        model = best_model_wts
        x = np.linspace(0,len(history['train']),len(history['train']))
        x1 = np.linspace(0,len(history['val']),len(history['val']))
        plt.plot(x,history['train'])
        plt.plot(x1,history['val'])
        plt.show()

        self.saveOutput(model,epoch,history,optimizer, finalTrainLoss, finalValLoss, self.cst["modelKey"])

        return history, best_model_wts, model

    def visualize_output(self, model):

        trainData = self.raw_data['train']
        trainData = trainData.permute(0,2,1)
        testData = self.raw_data['val']
        testData = testData.permute(0,2,1)

        with torch.no_grad():
            trainDataTensor, testDataTensor = (trainData,testData)
            predictions,latentSignal,_,_,_,_= model(testDataTensor.float().to(self.dev))
            print("-------------Analyzing Test Data-------------------")
            # latentSignal = latentSignal.squeeze(dim=2)
            # Calculate test dataset correlation coefficient
            correlationCoeffArr = []
            testDataTensor = testDataTensor.to(self.dev)
            for i in range(testDataTensor.shape[0]):
                correlationCoeffArr.append(self.utils.getCorrelationCoeff(predictions[i],testDataTensor[i]))
            correlationCoeffArr = np.array(correlationCoeffArr) 

            # Plot best and worgs signal latent space as both signal and bars
            print("Latent signal shape:", latentSignal.shape)
            # utilities.simpleViszation(latentSignal.cpu().detach().numpy(), 'Latent Signal ', index=np.argmin(correlationCoeffArr))
            # utilities.simpleViszation(latentSignal.cpu().detach().numpy(), 'Latent Signal ', index=np.argmax(correlationCoeffArr))
            # utilities.visualizeBarWeights(latentSignal.cpu().detach().numpy(), 'Latent Embedding Signal ', index=np.argmin(correlationCoeffArr))
            # utilities.visualizeBarWeights(latentSignal.cpu().detach().numpy(), 'Latent Embedding Signal ', index=np.argmax(correlationCoeffArr))

            print("Predictions signal shape:", predictions.shape)
            print("Test signal shape:", testDataTensor.shape)
            print("Mean Correlation coefficient test data:", np.mean(correlationCoeffArr))
            print("Min Correlation coefficient test data:", np.min(correlationCoeffArr))
            print("Max Correlation coefficient test data:", np.max(correlationCoeffArr))
            print("Argmin Correlation coefficient test data:", np.argmin(correlationCoeffArr))
            print("Argmax Correlation coefficient test data:", np.argmax(correlationCoeffArr))
            print("Percentage above 0.1  test data:", len(correlationCoeffArr[correlationCoeffArr>0.1]))
            print("Percentage below 0.1 test data:", len(correlationCoeffArr[correlationCoeffArr<0.1]))
            print("Percentage between -0.1 and 0.1 test data:", len(correlationCoeffArr[(correlationCoeffArr<0.1) & (correlationCoeffArr>-0.1)]))
            print("Uncorrelated test data:", len(correlationCoeffArr[correlationCoeffArr==0]))

            # Plot best and worst samples from the test data 
            testDataTensor = testDataTensor.cpu().detach().numpy()
            predictions = predictions.cpu().detach().numpy()
            self.utils.visualizeData(np.argmax(correlationCoeffArr),testDataTensor,predictions, title="Input Signals, Index: ")
            self.utils.visualizeData(np.argmin(correlationCoeffArr),testDataTensor,predictions, title="Input Signals, Index: ")
            plt.show()
            torch.cuda.empty_cache()

            # Plot correlation coefficents for each test data sample
            plt.bar(np.arange(len(correlationCoeffArr)),correlationCoeffArr)
            plt.show()

            # Calculate correlation coefficient for training data
            print("-------------Analyzing Train Data-------------------")
            predictions,latentSignal,_,_,_,_= model(trainDataTensor.float().to(self.dev))
            correlationCoeffArr = []
            trainDataTensor = trainDataTensor.to(self.dev)
            for i in range(trainDataTensor.shape[0]):
                correlationCoeffArr.append(self.utils.getCorrelationCoeff(predictions[i],trainDataTensor[i]))
            correlationCoeffArr = np.array(correlationCoeffArr) 

            print("Predictions signal shape:", predictions.shape)
            print("Test signal shape:", trainDataTensor.shape)
            print("Mean Correlation coefficient training data:", np.mean(correlationCoeffArr))
            print("Min Correlation coefficient training data:", np.min(correlationCoeffArr))
            print("Max Correlation coefficient training data:", np.max(correlationCoeffArr))
            print("Argmin Correlation coefficient training data:", np.argmin(correlationCoeffArr))
            print("Argmax Correlation coefficient training data:", np.argmax(correlationCoeffArr))
            print("Percentage above 0.1  training data:", len(correlationCoeffArr[correlationCoeffArr>0.1]))
            print("Percentage below 0.1 training data:", len(correlationCoeffArr[correlationCoeffArr<0.1]))
            print("Percentage between -0.1 and 0.1 training data:", len(correlationCoeffArr[(correlationCoeffArr<0.1) & (correlationCoeffArr>-0.1)]))
            print("Uncorrelated training data:", len(correlationCoeffArr[correlationCoeffArr==0]))

            trainDataTensor = trainDataTensor.cpu().detach().numpy()
            predictions = predictions.cpu().detach().numpy()
            self.utils.visualizeData(np.argmax(correlationCoeffArr),trainDataTensor,predictions, title="Input Signals, Index: ")
            self.utils.visualizeData(np.argmin(correlationCoeffArr),trainDataTensor,predictions, title="Input Signals, Index: ")

            # Plot corrleation coefficient as bar plot for each training dataset sample
            plt.figure(10)
            timeAxis = np.linspace(-1, 1,20)
            plt.bar(np.arange(len(correlationCoeffArr)),correlationCoeffArr)
            plt.show()
            torch.cuda.empty_cache()

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
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainloss': trainLoss, 'valLoss': valLoss}, os.path.join(self.cst['modelPath'], 
                                                                'model_' + modelKey + '_' + str(valLoss[-1]) + '_' + str(self.cst['modelName']) + '.pth'))

        with open(os.path.join(self.cst['historyPath'], 'history' + "." + 'json'), 'w') as json_file:
            json.dump(hisotry, json_file)
