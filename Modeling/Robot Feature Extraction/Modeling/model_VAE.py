from numpy import indices
import torch
from torch import nn
import sys
#TODO:
# 1 - Set the layer parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAE_Encoder(nn.Module):
    def __init__(self, config):
        super(VAE_Encoder, self).__init__()
        torch.manual_seed(0)

        # Convolutional layers for feature extraction
        self.convAcc = nn.Conv1d(
            in_channels=6,
            out_channels=6,
            kernel_size=5
        )
        nn.init.xavier_normal_(self.convAcc.weight, gain=nn.init.calculate_gain('relu'))

        self.convAng = nn.Conv1d(
            in_channels=6,
            out_channels=6,
            kernel_size=5
        )
        nn.init.xavier_normal_(self.convAng.weight, gain=nn.init.calculate_gain('relu'))

        # LSTM layers
        self.LSTM1 = nn.LSTM(
            input_size = 12,
            hidden_size = 12,
            num_layers = 120,
            batch_first=True
        )

        self.attn = nn.Linear(
            in_features=6,
            out_features=12
        )
        nn.init.xavier_normal_(self.attn.weight, gain=nn.init.calculate_gain('relu'))

        self.lin1 = nn.Linear(
            in_features=12,
            out_features=12
        )
        nn.init.xavier_normal_(self.attn.weight, gain=nn.init.calculate_gain('relu'))

        self.lin2 = nn.Linear(
            in_features=12,
            out_features=12
        )
        nn.init.xavier_normal_(self.attn.weight, gain=nn.init.calculate_gain('relu'))

        # Latent vector representation
        self.hidden2mu = nn.Linear(12,12)
        nn.init.xavier_normal_(self.hidden2mu.weight, gain=nn.init.calculate_gain('linear'))
        self.hidden2log_var = nn.Linear(12,12)
        nn.init.xavier_normal_(self.hidden2log_var.weight, gain=nn.init.calculate_gain('linear'))

        # Activation functions 
        self.activationFunction = nn.ReLU()
        self.activationFunction1 = nn.ReLU()
        self.activationFunction2 = nn.ReLU()

        self.lin_act1 = nn.ReLU()
        self.lin_act2 = nn.ReLU()

        self.layerNomr1 = nn.LayerNorm([24,126])
        self.layerNomr2 = nn.LayerNorm([24,126])
        self.layerNomr3 = nn.LayerNorm([126,128])

        # self.maxpool1 = nn.AvgPool1d(3)
        # self.maxpool2 = nn.AvgPool1d(3)

        self.batchnorm1 = nn.BatchNorm1d(6)
        self.batchnorm2 = nn.BatchNorm1d(6)

    def reparametrize(self,mu,log_var):
        
        std = torch.exp(log_var/2)
        q = torch.distributions.Normal(mu,std)
        z = q.rsample()

        return z

    def encoder(self,inputSignals):
        # Split inputs
        # inputAcc = inputSignals[:,:3,:]
        # inputAngVel = inputSignals[:,3:,:]

        # Run feature selector, concatenate outputs and run attention
        outAcc = self.activationFunction(self.batchnorm1(self.convAcc(inputSignals)))
        outAng = self.activationFunction1(self.batchnorm2(self.convAng(outAcc)))

        # outAcc = outAcc.permute(0,2,1)
        # outAng = outAng.permute(0,2,1)
        # out = torch.cat((outAng,outAcc),2)

        out = self.activationFunction2(self.attn(outAng.permute(0,2,1)))

        self.LSTM1.flatten_parameters()
        out, (hidden_state, cell_state) = self.LSTM1(out)
        #out, (hidden_state, cell_state) = self.LSTM2(out) 

        last_lstm_layer_hidden_state = hidden_state.permute(1,0,2)
        last_lstm_layer_hidden_state = self.lin_act1(self.lin1(last_lstm_layer_hidden_state))
        out = self.lin_act2(self.lin2(last_lstm_layer_hidden_state))

        # last_lstm_layer_hidden_state = hidden_state[-1,:,:]

        mu = self.mu_act(self.hidden2mu(out))
        log_var = self.sigm_act(self.hidden2log_var(out))

        return out, mu, log_var , 0, 0

    def forward(self, x):    
        
        latentVector,mu,log_var, inidices1, inidices2 = self.encoder(x)
        sampleVector = self.reparametrize(mu,log_var)

        return sampleVector,latentVector, mu, log_var, inidices1, inidices2

class VAE_Decoder(nn.Module):
    def __init__(self, config):
        super(VAE_Decoder, self).__init__()
        torch.manual_seed(0)

        # Convolutional layers for feature extraction
        self.deConvAng = nn.ConvTranspose1d(
            in_channels=6,
            out_channels=6,
            kernel_size=5
        )
        nn.init.xavier_normal_(self.deConvAng.weight, gain=nn.init.calculate_gain('relu'))

        self.deConvAcc = nn.ConvTranspose1d(
            in_channels=6,
            out_channels=6,
            kernel_size=5
        )    
        nn.init.xavier_normal_(self.deConvAcc.weight, gain=nn.init.calculate_gain('relu'))

        # LSTM layers
        self.LSTM1 = nn.LSTM(
            input_size = 12,
            hidden_size = 12,
            num_layers = 120,
            batch_first=True
        )
        
        self.LSTM2 = nn.LSTM(
            input_size = 64,
            hidden_size = 64,
            num_layers = 1,
            batch_first=True
        )

        # Attention layer
        self.attn = nn.Linear(
            in_features=12,
            out_features=12
        )
        nn.init.xavier_normal_(self.attn.weight, gain=nn.init.calculate_gain('relu'))

        self.outputAng = nn.Linear(
            in_features=12,
            out_features=6
        )
        nn.init.xavier_normal_(self.outputAng.weight, gain=nn.init.calculate_gain('relu'))

        self.outputAcc = nn.Linear(
            in_features=6,
            out_features=6
        )
        nn.init.xavier_normal_(self.outputAcc.weight, gain=nn.init.calculate_gain('linear'))

        self.lin1 = nn.Linear(
            in_features=12,
            out_features=12
        )
        nn.init.xavier_normal_(self.attn.weight, gain=nn.init.calculate_gain('relu'))

        self.lin2 = nn.Linear(
            in_features=12,
            out_features=12
        )
        nn.init.xavier_normal_(self.attn.weight, gain=nn.init.calculate_gain('relu'))
        

        # Activation functions 
        self.activationFunction = nn.ReLU()
        self.activationFunction1 = nn.ReLU()
        self.activationFunction2 = nn.ReLU()
        self.actFunc = nn.ReLU()

        self.lin_act1 = nn.ReLU()
        self.lin_act2 = nn.ReLU()

        self.hiddenState = (torch.zeros(1,64,24),
                            torch.zeros(1,64,24))

        self.sigmoid = nn.Sigmoid()
        self.layerNorm1 = nn.LayerNorm(48)
        self.layerNorm2 = nn.LayerNorm([6,128])
        self.layerNorm3 = nn.LayerNorm([6,128])

        # self.maxunpool1 = (5,stride=3)
        # self.maxunpool2 = nn.MaxUnpool1d(4,stride=3)

        self.batchnorm1 = nn.BatchNorm1d(6)
        self.batchnorm2 = nn.BatchNorm1d(6)

    def decoder(self,latentVector, inidices1, inidices2):

        # latentVector = latentVector.unsqueeze(1).repeat(1, 124, 1)

        latentVector = self.lin_act1(self.lin1(latentVector))
        latentVector = self.lin_act2(self.lin2(latentVector))

        self.LSTM1.flatten_parameters()
        out,(hidden,cell) = self.LSTM1(latentVector)

        out = self.activationFunction(self.attn(hidden.permute(1,0,2)))
        out = self.actFunc(self.outputAng(out))

        # outAcc = out[:,:,:3].permute(0,2,1)
        # outAng = out[:,:,3:].permute(0,2,1)
        out = out.permute(0,2,1)
        outAcc = self.batchnorm1(self.deConvAcc(out))
        
        outAcc = self.activationFunction1(outAcc)
        # outAcc = self.outputAcc(outAcc.permute(0,2,1)).permute(0,2,1)

        outAng = self.batchnorm2(self.deConvAng(outAcc))
        outAng = self.activationFunction2(outAng)
        # outAng = self.outputAng(outAng.permute(0,2,1)).permute(0,2,1)

        # outAcc = torch.cat((outAcc,outAng),1)
        out = self.outputAcc(outAng.permute(0,2,1)).permute(0,2,1)

        return out

    def forward(self, x, inidices1, inidices2):    
        outputReconstruction = self.decoder(x, inidices1, inidices2)
        
        return outputReconstruction
        
class VAE(nn.Module):
    def __init__(self, config):
        super(VAE, self).__init__()
        torch.manual_seed(0)

        self.encoder = VAE_Encoder(config).to(device)
        self.decoder = VAE_Decoder(config).to(device)

        self.logScale = nn.Parameter(torch.tensor([0.0]))

    def calculateDecoderGuassianLikelihood(self, reconstructionOutput, inputBatch):
        
        # Scale value ->  torch.exp(self.logScale)
        # Mean value -> reconstructionOutput
        dist = torch.distributions.Normal(reconstructionOutput,torch.exp(self.logScale))
        logPxz = dist.log_prob(inputBatch)

        return logPxz.sum(dim=(1,2))

    def calculateOutput(self,mu,var):
        std = torch.exp(var/2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z

    def forward(self, x):

        torch.manual_seed(0)
        lstmOut, latentVector, mu, log_var, inidices1, inidices2 = self.encoder(x)
        lstmOut = self.calculateOutput(mu,log_var)
        output = self.decoder(lstmOut, inidices1, inidices2)
        reconLikelihood = self.calculateDecoderGuassianLikelihood(output,x)

        return output.float(), lstmOut.float(), latentVector.float(), mu.float(), log_var.float(), reconLikelihood.float()