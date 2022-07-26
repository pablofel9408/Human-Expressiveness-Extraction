import platform
import sys
import torch 
import constants
import neuralNetworkClass
import numpy as np
from  utilitiesClass import utility
import matplotlib.pyplot as plt

def main(): 
    """
    Initialize main object and access functions thorugh the class as needed
    Args: None
    Output: None
    """
    print('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    if platform.system() == 'Darwin':
        print('Using Mac Constants')
        modelDefinition = constants.modelDefinitionMac
    elif platform.system() == 'Windows':
        print('Using Windows Constants')
        modelDefinition = constants.modelDefinitionWindows
    else: 
        print('Linux version not yet implemented')
        sys.exit()
    
    # Instantiate the nerual network calss, get dataset, train model or obtain the model
    # from intial file
    nt = neuralNetworkClass.networkFunctions(modelDefinition)
    utilities = utility(device)
    trainData, testData = nt.loadDataset(modelDefinition['path'])
    utilities.visualizeData(0, trainData, testData, title='')
    nt.printDataShape()
    _,_,trainData,testData = nt.preProcessData(64,normalized=True)
    # plt.show()
    
    if not constants.trainFlag:
        # best model so far -> r'C:\Users\posorio\Documents\ModelTesting\Models\model_-216.40023968507936_VAE_.pth'
        #_,_,_,model = nt.loadModel(constants.configModel,r'C:\Users\posorio\Documents\ModelTesting\Models\model_-284.5510693626821_VAE_.pth',True)
        _,_,_,model = nt.loadModel(constants.configModel,r'C:\Users\posorio\Documents\ModelTesting\Models\model_15.014928499857584_VAE_.pth',True)
    else:
        history,_,model = nt.trainVAE(constants.configModel,modelKey=constants.modelType)
    model = model.to(device)

    with torch.no_grad():
        trainDataTensor, testDataTensor = (torch.tensor(trainData),torch.tensor(testData))
        predictions,latentSignal,_,_,_,_= model(testDataTensor.float().to(device))
        # latentSignal = latentSignal.squeeze(dim=2)
        # Calculate test dataset correlation coefficient
        correlationCoeffArr = []
        testDataTensor = testDataTensor.to(device)
        for i in range(testDataTensor.shape[0]):
            correlationCoeffArr.append(utilities.getCorrelationCoeff(predictions[i],testDataTensor[i]))
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
        utilities.visualizeData(np.argmax(correlationCoeffArr),testDataTensor,predictions, title="Input Signals, Index: ")
        utilities.visualizeData(np.argmin(correlationCoeffArr),testDataTensor,predictions, title="Input Signals, Index: ")
        plt.show()
        torch.cuda.empty_cache()

        # Plot correlation coefficents for each test data sample
        plt.bar(np.arange(len(correlationCoeffArr)),correlationCoeffArr)
        plt.show()

        # Calculate correlation coefficient for training data
        predictions,latentSignal,_,_,_,_= model(trainDataTensor.float().to(device))
        correlationCoeffArr = []
        trainDataTensor = trainDataTensor.to(device)
        for i in range(trainDataTensor.shape[0]):
            correlationCoeffArr.append(utilities.getCorrelationCoeff(predictions[i],trainDataTensor[i]))
        correlationCoeffArr = np.array(correlationCoeffArr) 

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
        utilities.visualizeData(np.argmax(correlationCoeffArr),trainDataTensor,predictions, title="Input Signals, Index: ")
        utilities.visualizeData(np.argmin(correlationCoeffArr),trainDataTensor,predictions, title="Input Signals, Index: ")

        # Plot corrleation coefficient as bar plot for each training dataset sample
        plt.figure(10)
        timeAxis = np.linspace(-1, 1,20)
        plt.bar(np.arange(len(correlationCoeffArr)),correlationCoeffArr)
        plt.show()
        torch.cuda.empty_cache()
    

if __name__ == "__main__": 

    main()
