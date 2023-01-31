import torch 
import json 
import constants
import numpy as np
import os 
from torch.utils.data import DataLoader
import numpy as np
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
import copy
import model as modelLib
from model_VAE import VAE
import smallerVAE
import smallerLatentVAE
from adamp import AdamP
import scipy.signal

class networkFunctions(): 
    def __init__(self, config):
        """
        Args: path -> string holding the dataset path 
              modelClass -> class object that hold the Pytorch model of the framework to train
              optimizerClass -> Pytorch optimizer class, used in tranning and resume trainning 
              preloadModel -> boolean flag to signal if the training is to be continued

        Function: it will define the initial variables and constructs of the class in order to access
                the data as needed in the other function calls

        Return: None   
        """
        super(networkFunctions, self).__init__()

        self.trainDataBatches = {'normalized': None, 'regular': None}
        self.testDataBatches = {'normalized': None, 'regular': None}

        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print("Using {} device".format(self.dev))
        
        if config['train']:
            self.trainData, self.testData = self.loadDataset(config['path'])
            self.trainDataTensor, self.testDataTensor = (torch.tensor(self.trainData),torch.tensor(self.testData))
            self.trainDataBatches['normalized'],\
                self.testDataBatches['normalized'],_,_ = self.preProcessData(config['batch_size'],normalized=constants.normalize)
            self.trainDataBatches['regular'],\
                 self.testDataBatches['regular'],_,_  = self.preProcessData(config['batch_size'],normalized=not constants.normalize)

        print(np.shape(self.trainData))

        if not config['preLoadModel']:

            self.modelClass = config['modelClass']
            self.optimizerClass = config['optimizerClass']

        self.modelPath = config['modelPath']
        self.historyPath = config['historyPath']
        self.modelName = config['modelName']

        self.modelsDict = {}

        print('new class')
    
    def loadDataset(self,path):
        """
        Args: path -> string that hold the path to the main fodler that holds the dataset 

        return: trainData, testData -> both are numpy array holding the dataset

        Function: the call is used to load al the dataset information into the class for easier handling 
        """
        
        # Auxiliary arrays to import the dataset and matrices definition to hold the import data
        filesname = ['x','y','z']
        components = ['gyro', 'acc']
        trainData = np.zeros([7352, 6, 128])
        testData = np.zeros([2947, 6, 128])
        copyTrain = np.copy(trainData)
        copyTest = np.copy(testData)
        countRange = iter(range(6))

        # For loop to load all the dataset
        for j in components:
            for i in filesname:
                n = next(countRange)
                trainData[:,n,:] = np.loadtxt(os.path.join(path, 'train', 'Inertial Signals', 'body_' + j + '_' + i + '_train' + '.txt'))
                testData[:,n,:] = np.loadtxt(os.path.join(path, 'test', 'Inertial Signals', 'body_' + j + '_' + i + '_test' + '.txt'))

        b, a = scipy.signal.butter(3, 0.1)
        for i in range(np.shape(trainData)[1]):
            for j in range(len(trainData)):
                trainData[j,i,:] = scipy.signal.filtfilt(b, a, trainData[j,i,:])
                if j <= len(testData)-1:
                    testData[j,i,:] = scipy.signal.filtfilt(b, a, testData[j,i,:])

        # Check that array is properly loaded, no values repeated or shared among the dimensions 
        results = [False for i in range(7)]
        results[0] = np.all(trainData == trainData[0])
        results[1] = np.all(testData == testData[0])
        results[2] = np.array_equal(trainData, copyTrain)
        results[3] = np.array_equal(testData, copyTest)
        results[4] = np.array_equal(trainData, testData)
        results[5] = False if np.count_nonzero(trainData) > 0 else True
        results[6] = False if np.count_nonzero(testData) > 0 else True

        if any(results):
            print(results) 
            print("All values are equal")

        return trainData, testData

    def preProcessData(self, batch_size,normalized=False):

        trainData = np.copy(self.trainData)
        testData = np.copy(self.testData)
        if normalized: 
            for i in range(np.shape(trainData)[1]):
                trainData[:,i,:] = np.transpose(MinMaxScaler().fit_transform(np.transpose(self.trainData[:,i,:])))
                testData[:,i,:] = np.transpose(MinMaxScaler().fit_transform(np.transpose(self.testData[:,i,:])))
                #trainData[:,i,:] = np.transpose(StandardScaler().fit_transform(np.transpose(self.trainData[:,i,:])))
                #testData[:,i,:] = np.transpose(StandardScaler().fit_transform(np.transpose(self.testData[:,i,:])))

        # Create data loaders.
        train_dataloader = DataLoader(torch.tensor(trainData), batch_size=batch_size)
        test_dataloader = DataLoader(torch.tensor(testData), batch_size=batch_size)

        print('preprocess')

        return train_dataloader, test_dataloader, trainData, testData

    def getModel(self,config, modelKey=''):

        switcher = {
                'VAE':VAE(config).to(self.dev),
                "AE": '',
                "SmallerVAE": smallerVAE.VAE(config).to(self.dev),
                "SmallerLatentVAE": smallerLatentVAE.VAE(config).to(self.dev)
            }

        return switcher.get(modelKey, "nothing")  


    def customLossCoompu(self):
        # TODO: Custom loss for expressivness 
        print("Custom loss")


    def computeVAELoss(self,log_var,mu,recon_loss, batch_size=1.0):

        KLD =  torch.sum(-0.5 *(1 + log_var - mu.pow(2) - log_var.exp()),dim=1).mean(dim =0) 
        return recon_loss[0] + (KLD / batch_size)

        # kl_loss =  (-0.5*(1+log_var - mu**2- torch.exp(log_var)).sum(dim = 1)).mean(dim =0)        
        # return recon_loss*128 + kl_loss

    def kl_divergence(self,z,mu,log_var):

        std = torch.exp(log_var/2)
        # print("Standard deviation shape:", std.shape)
        # print("Mean shape:", mu.shape)
        # print("Sample shape:", z.shape)
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
        kl = kl.sum(-1)
        # print("Kullback divergance shape:", kl.shape)
        return kl

    def newELBOLoss(self,z,log_var,mu,recon_loss):
        kl = self.kl_divergence(z,mu,log_var)
        # print("Reconstruction loss shape:", recon_loss.shape)
        elbo = (kl - recon_loss)    
        elbo = elbo.mean()
        return elbo

    def trainVAE(self, config, modelKey=''):
        """
        Trainning module for a regular AE network, losses are computed as differences between 
        inputs and the reconstruction
        """
        model = self.getModel(config,modelKey)
        criterion = torch.nn.MSELoss()
        model = model.to(self.dev)
        # optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=constants.L2regularizer)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=constants.L2regularizer)
        history = {'train':[], 'val':[]}

        best_model_wts = copy.deepcopy(model)
        best_loss = 1000000.0
        scheduler = ReduceLROnPlateau(optimizer, mode=constants.LRschedulerMode, factor=constants.LRfactor, patience=constants.LRpatience, verbose=True)
        epochCount = constants.epochCount 
        earlyStopPatience = constants.earlyStopPatience
        prevLoss = constants.prevLoss
        finalTrainLoss = []
        finalValLoss = []
        n_epochs = config['epochs_num']
        for epoch in range(1, n_epochs + 1):
            model = model.train()

            train_losses = []
            for seq_true in self.trainDataBatches['normalized']:

                seq_true = seq_true.to(self.dev)
                optimizer.zero_grad()
                seq_pred, z, _, mu, log_var, recon_loss = model(seq_true.float())
                loss = self.newELBOLoss(z,log_var,mu,recon_loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

                train_losses.append(loss.item())

            val_losses = []
            model = model.eval()
            with torch.no_grad():
                for seq_true in self.testDataBatches['normalized']:

                    seq_true = seq_true.to(self.dev)
                    seq_pred, z, _, mu, log_var, recon_loss = model(seq_true.float())
                    loss = self.newELBOLoss(z,log_var,mu,recon_loss)
                    val_losses.append(loss.item())

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

        self.saveOutput(model,epoch,history,optimizer, finalTrainLoss, finalValLoss, modelKey)

        return history, best_model_wts, model

    def trainModuleAE(self, config):
        """
        Trainning module for a regular AE network, losses are computed as differences between 
        inputs and the reconstruction
        """
        model = modelLib.LSTM_AE(config)
        criterion = torch.nn.MSELoss()
        model = model.to(self.dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=constants.L2regularizer)
        history = {'train':[], 'val':[]}

        best_model_wts = copy.deepcopy(model)
        best_loss = 10000.0
        scheduler = ReduceLROnPlateau(optimizer, mode=constants.LRschedulerMode, factor=constants.LRfactor, patience=constants.LRpatience, verbose=True)
        epochCount = constants.epochCount 
        earlyStopPatience = constants.earlyStopPatience
        prevLoss = constants.prevLoss
        finalTrainLoss = []
        finalValLoss = []
        n_epochs = config['epochs_num']
        for epoch in range(1, n_epochs + 1):
            print(epoch)
            model = model.train()

            train_losses = []
            for seq_true in self.trainDataBatches['normalized']:

                seq_true = seq_true.to(self.dev)
                optimizer.zero_grad()
                seq_pred,_, mu = model(seq_true.float())
                loss = criterion(seq_pred.float(), seq_true.float())
                #loss = self.final_loss(loss, mu, log_var)

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            val_losses = []
            model = model.eval()
            with torch.no_grad():
                for seq_true in self.testDataBatches['normalized']:

                    seq_true = seq_true.to(self.dev)
                    seq_pred,_, mu = model(seq_true.float())
                    loss = criterion(seq_pred.float(), seq_true.float())
                    #loss = self.final_loss(loss, mu, log_var)
                    val_losses.append(loss.item())

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
            finalValLoss.append(val_losses)

            print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

        model = best_model_wts
        # outputSave = {}
        # outputSave['epoch'] = epoch
        # outputSave['history'] = history
        # outputSave['optimizer'] = optimizer
        # outputSave['finalTrainLoss'] = finalTrainLoss
        # outputSave['finalValLoss'] = finalValLoss.item
        # self.modelsDict['experiment' + str(np.mean(finalValLoss) + str(np.random.randint(0,50000)))] = {"valLoss": np.mean(finalValLoss), 
        #                                                                                                             "model": model, "outputSave": outputSave}

        # tempBest = 10000000
        # bestModel = {}
        # for key,value in self.modelsDict.items():
        #     if value['valLoss'] < tempBest:
        #         bestModel = copy.deepcopy(value)
        
        # dictValues = self.setValuesFromDict(bestModel['outputSave'])
        # self.saveOutput(model,dictValues[0],dictValues[1],dictValues[2],dictValues[3],dictValues[4])

        self.saveOutput(model,epoch,history,optimizer, finalTrainLoss, finalValLoss)

        return history, best_model_wts, model

    def setValuesFromDict(self,inputDict):

        epochs = inputDict['epoch']
        history = inputDict['history']
        optimizer = inputDict['optimizer']
        finalTrainLoss = inputDict['finalTrainLoss']
        finalValLoss = inputDict['finalValLoss']

        return epochs, history, optimizer, finalTrainLoss, finalValLoss

    def trainModuleCustomNN(self, featureModel, regularizerModel, train_dataset, val_dataset, n_epochs, criterion):
        """
        Custom neural netowrk designed for shared learning, aiming for a regularization in the 
        feature selection process for the AE. 
        Networks used:
                        - Time series autoencoders
                        - Images autoencoders
                        - Laban analysis classification 
        """
        # criterion = nn.L1Loss(reduction='sum')
        model_AE = featureModel.to(self.dev)
        model_LabanClassifier = regularizerModel.to(self.dev)
        optimizer = torch.optim.Adam([{'params': model_AE.parameters(), 'lr':1e-3},
                        {'params': model_LabanClassifier.parameters(), 'lr': 1e-3}])

        history = dict(train=[], val=[])

        best_model_wts= {'AE': copy.deepcopy(model_AE.state_dict()), 'labanClassifier': copy.deepcopy(model_LabanClassifier.state_dict())}
        best_loss = 10000.0
        
        for epoch in range(1, n_epochs + 1):
            model_AE = model_AE.train()
            model_LabanClassifier = model_LabanClassifier.train()

            train_losses = []
            for seq_true in train_dataset:

                seq_true = seq_true.to(self.dev)
                seq_true = seq_true.to(self.dev)
                
                optimizer.zero_grad()
                seq_pred = model_AE(seq_true)
                seq_pred = model_LabanClassifier(seq_true)

                lossAE = criterion['AE'](seq_pred, seq_true)
                lossLaban = criterion['labanClassifier'](seq_pred, seq_true)
                loss = lossAE + lossLaban

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            val_losses = []
            model = model.eval()
            with torch.no_grad():
                for seq_true in val_dataset:

                    seq_true = seq_true.to(self.dev)
                    seq_pred = model_AE(seq_true)
                    seq_pred = model_LabanClassifier(seq_true)

                    vallossAE = criterion['AE'](seq_pred, seq_true)
                    vallossLaban = criterion['labanClassifier'](seq_pred, seq_true)
                    valloss = vallossAE + vallossLaban
                    val_losses.append(valloss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            history['train'].append(train_losses)
            history['val'].append(val_losses)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = {'AE': copy.deepcopy(model_AE.state_dict()), 'labanClassifier': copy.deepcopy(model_LabanClassifier.state_dict())}

            print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

        model_AE = model_AE.load_state_dict(best_model_wts['AE'])
        model_LabanClassifier = model_LabanClassifier.load_state_dict(best_model_wts['labanClassifier'])
        self.saveOutput([model_AE,model_LabanClassifier],epoch,history,optimizer, train_loss, val_loss)

        return history, best_model_wts

    def setupRay(self):

        # Scheduler based Ray search
        scheduler = ASHAScheduler(
        metric=constants.rayMetric,
        mode=constants.rayMode,
        max_t=constants.epochMaxNum,
        grace_period=10,
        reduction_factor=3)

        reporter = CLIReporter(
                # parameter_columns=["l1", "l2", "lr", "batch_size"],
                metric_columns=constants.rayMetricsColumns)

        result = tune.run(
            self.trainModuleAE,
            resources_per_trial={"cpu": constants.cpuNumber, "gpu": constants.gpuNumber},
            config=constants.configModelRay,
            num_samples=constants.samplesNumber,
            scheduler=scheduler,
            progress_reporter=reporter,
            checkpoint_at_end=False)

        tempBest = 10000000
        bestModel = {}
        for key,value in self.modelsDict.items():
            if value['valLoss'] < tempBest:
                bestModel = copy.deepcopy(value)
        
        dictValues = self.setValuesFromDict(bestModel['outputSave'])
        self.saveOutput(bestModel['model'],dictValues[0],dictValues[1],dictValues[2],dictValues[3],dictValues[4])

        # Output best models
        print(type(result))
        print(result)

        best_trial = result.get_best_trial(constants.rayMetric, constants.rayMode)
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final test loss: {}".format(best_trial.last_result[constants.rayMetricsColumns[0]]))
        print("Best trial final test accuracy: {}".format(best_trial.last_result[constants.rayMetricsColumns[1]]))
        temp = best_trial.config
        temp['activation_function'] = str(temp['activation_function'])
        temp['attn_activation_function'] = str(temp['attn_activation_function'])
        temp['accuracy'] = best_trial.last_result[constants.rayMetricsColumns[1]]
        self.saveJSON(temp)
    
    def saveOutput(self, model, epoch, hisotry, optimizer, trainLoss, valLoss, modelKey, multipleModels=False): 
        
        if multipleModels:

            torch.save({
                'epoch': epoch,
                'modelAE_state_dict': model[0].state_dict(),
                'modelLaban_state_dict': model[1].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainloss': trainLoss, 'valLoss': valLoss}, os.path.join(self.modelPath, 'model_' + modelKey + '_' + self.modelName + "." + 'pth'))
        
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'trainloss': trainLoss, 'valLoss': valLoss}, os.path.join(self.modelPath, 'model_' + modelKey + '_' + str(valLoss[-1]) + '_' + self.modelName + "." + 'pth'))

        with open(os.path.join(self.historyPath, 'history' + "." + 'json'), 'w') as json_file:
            json.dump(hisotry, json_file)
    
    def saveJSON(self,hisotry):
        with open(os.path.join(self.historyPath, 'history' + str(hisotry['accuracy']) + "." + 'json'), 'w') as json_file:
            json.dump(hisotry, json_file)

    def loadModel(self, config,modelPath, evalFlag, modelKey):
        
        checkpoint = torch.load(modelPath, map_location=torch.device(self.dev))
        model =  self.getModel(config, modelKey)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=constants.L2regularizer)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        trainLoss = checkpoint['trainloss']
        valLoss = checkpoint['valLoss']

        if evalFlag: 
            model.eval()
        else: 
            model.train()

        return epoch, trainLoss, valLoss, model

    def printDataShape(self):

        print('Train Input Shape:', self.trainDataTensor.shape)
        print('Test Input Shape:', self.testDataTensor.shape)
