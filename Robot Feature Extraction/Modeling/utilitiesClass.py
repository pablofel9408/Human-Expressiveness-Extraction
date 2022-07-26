import numpy as np
import matplotlib.pyplot as plt
import torch 

class utility():
    def __init__(self, device, common=None):
        super(utility, self).__init__()

        self.dev = device

    def simpleViszation(self, inputData, title, index=0):

        timeAxis = np.linspace(0, np.shape(inputData)[1],np.shape(inputData)[1])
        fig = plt.figure()
        plt.plot(timeAxis,inputData[index])
        plt.title(title + str(index))

    def getCorrelationCoeff(self, inputTensor, preidctions):

        x = preidctions
        y = inputTensor.float().to(self.dev)

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost.item()

    def visualizeData(self, index, array1, array2, title=''): 
        """
        Args: self -> class call and access to the variables that lay within it. 
            index -> int that will be used as index of both train and test arrays

        Return: None

        Function: Plot both the train and test in subplots, only the arrays associated to the
        index are plotted.
        """

        featureDimLenght = np.shape(array1)[1]
        timeTrain = np.linspace(0,len(array1[0,0,:]),len(array1[0,0,:]))
        timeTest = np.linspace(0,len(array2[0,0,:]),len(array2[0,0,:]))
        fig, axes = plt.subplots(2,featureDimLenght)

        axesColors = ['tab:blue','tab:orange', 'tab:purple', 'tab:green', 'tab:red', 'tab:pink']
        axesTitles = ['AV X', 'AV Y', 'AV Z', 'A X', 'A Y', 'A Z']

        for n,row in enumerate(axes.flat):
            if n < featureDimLenght:
                row.set_title(axesTitles[n])
                row.plot(timeTrain, array1[index,n,:], axesColors[n])
                row.set_ylabel('Original Signals')
            else:
                row.set_title(axesTitles[n - featureDimLenght])
                row.plot(timeTest,array2[index,n-featureDimLenght,:], axesColors[n-featureDimLenght])
                row.set_ylabel('Network Output')
        fig.suptitle(title + str(index))
        #plt.show()
        print('visualize')
    
    def visualizeBarWeights(self, inputData, title, index=0):

        timeAxis = np.linspace(0, np.shape(inputData)[1],np.shape(inputData)[1])
        fig = plt.figure()
        plt.bar(timeAxis,inputData[index])
        plt.title(title + str(index))
        plt.xlabel("Embedding Weight Index")
        plt.ylabel("Weight Value")