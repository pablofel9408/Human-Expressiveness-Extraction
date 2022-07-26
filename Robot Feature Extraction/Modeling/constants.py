from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ray import tune
import torch.nn as nn

### Constants File for Attention LSTM Autoencoder model 

encoderDefinition = {
    'hidden_size_encoder': 0,
    'seq_len': 0,
    'denoising': 0,
    'directions': 0
}

decoderDefinition = {
    'hidden_size_encoder': 0,
    'hidden_size_decoder'
    'seq_len': 0,
    'output_size': 0
}

epochMaxNum = 500
samplesNumber = 1
cpuNumber = 8
gpuNumber = 0
rayMetric = 'loss'
rayMode = 'min'
rayMetricsColumns = ["loss", "accuracy"]

configModelRay = {'attention_layer':tune.grid_search([12,24,48]),
          'convolutional_layer':tune.grid_search([6,12,24]),
          'lstm_latent':tune.grid_search([64,128,256]),
          'lstm_num':tune.grid_search([1]),
          'activation_function':tune.grid_search([nn.ReLU()]),
          'attn_activation_function':tune.grid_search([nn.ReLU()]),             
          'learning_rate':tune.loguniform(1e-3, 1e-1),
          'epochs_num':tune.grid_search([500]),
          'input_channels': 3,
          'scheduler': False,
          'earlyStop': True,
          'fixed': False,
          'p':0}

configModel= {'attention_layer':tune.grid_search([12,24,48]),
          'convolutional_layer':tune.grid_search([6,12,24]),
          'lstm_latent':tune.grid_search([64,128,256]),
          'lstm_num':tune.grid_search([1]),
          'activation_function':nn.ReLU(),
          'attn_activation_function':nn.ReLU(),             
          'learning_rate':1e-3,
          'epochs_num':5000,
          'input_channels': 3,
          'scheduler': False,
          'earlyStop': True,
          'fixed': False,
          'p':0}

# Model Constants
printInNetwork = False
trainFlag = True

# Preprocessing and loading 
# Two different defintions because of paths differnece in Windows and Mac
modelType = 'SmallerVAE'

epochNum = 500
LRschedulerMode = 'min'
LRpatience = 8
LRfactor = 0.1
epochCount = 0 
earlyStopPatience = 20
prevLoss = 1000000.0
L2regularizer = 1e-7
visualize = False
normalize = True

modelDefinitionMac = {
    'path':  '/Users/Pablo/Documents/GVLAB/Phd/Shared Scripts/Datasets/UCI_HAR_Dataset',
    'batch_size': 64,
    'optimizerClass': None,
    'modelClass': None,
    'preLoadModel': False,
    'Scaler': MinMaxScaler,
    'train': True,
    'modelPath': '/Users/Pablo/Documents/GVLAB/Phd/Shared Scripts/Models/',
    'historyPath': '/Users/Pablo/Documents/GVLAB/Phd/Shared Scripts/History/',
    'modelName': 'MSELoss_WithSoftmax_State_Dict'
}

modelDefinitionWindows = {
    'path':  r'C:\Users\posorio\Documents\ModelTesting\Data\UCI HAR Dataset\UCI HAR Dataset',
    'batch_size': 512,
    'optimizerClass': None,
    'modelClass': None,
    'modelName': "VAE_",
    'preLoadModel': False,
    'Scaler': MinMaxScaler,
    'train': True,
    'modelPath': r'C:\Users\posorio\Documents\ModelTesting\Models',
    'historyPath':  r'C:\Users\posorio\Documents\ModelTesting\Models'
}