import sys

import torch
from torch import nn
from torch.distributions import Independent,Normal

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
        self.mu = nn.Linear(12,12)
        nn.init.xavier_normal_(self.mu.weight, gain=nn.init.calculate_gain('linear'))

        self.sigma = nn.Linear(12,12)
        nn.init.xavier_normal_(self.sigma.weight, gain=nn.init.calculate_gain('relu'))

        # Activation functions 
        self.activationFunction = nn.ReLU()
        self.activationFunction1 = nn.ReLU()
        self.activationFunction2 = nn.ReLU()

        self.lin_act1 = nn.ReLU()
        self.lin_act2 = nn.ReLU()

        self.mu_act = nn.Softplus()
        self.sigm_act = nn.Softplus()

        # self.maxpool1 = nn.AvgPool1d(3)
        # self.maxpool2 = nn.AvgPool1d(3)

        self.batchnorm1 = nn.BatchNorm1d(6)
        self.batchnorm2 = nn.BatchNorm1d(6)


    def encoder(self,inputSignals):
        # Run feature selector, concatenate outputs and run attention
        outAcc = self.activationFunction(self.batchnorm1(self.convAcc(inputSignals)))
        outAng = self.activationFunction1(self.batchnorm2(self.convAng(outAcc)))

        out = self.activationFunction2(self.attn(outAng.permute(0,2,1)))

        self.LSTM1.flatten_parameters()
        out, (hidden_state, cell_state) = self.LSTM1(out)

        last_lstm_layer_hidden_state = hidden_state.permute(1,0,2)
        last_lstm_layer_hidden_state = self.lin_act1(self.lin1(last_lstm_layer_hidden_state))
        out = self.lin_act2(self.lin2(last_lstm_layer_hidden_state))

        mu = self.mu(out)
        log_var = self.sigm_act(self.sigma(out))

        return out, mu, log_var

    def forward(self, x):    
        
        latentVector,mu,log_var = self.encoder(x)

        return latentVector, mu, log_var

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

        # Attention layer
        self.attn = nn.Linear(
            in_features=12,
            out_features=6
        )
        nn.init.xavier_normal_(self.attn.weight, gain=nn.init.calculate_gain('relu'))

        self.sig = nn.Linear(
            in_features=6,
            out_features=6
        )
        nn.init.xavier_normal_(self.sig.weight, gain=nn.init.calculate_gain('relu'))

        self.mu = nn.Linear(
            in_features=6,
            out_features=6
        )
        nn.init.xavier_normal_(self.mu.weight, gain=nn.init.calculate_gain('relu'))

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

        self.mu_act = nn.Softplus()
        self.sigm_act = nn.Softplus()

        self.batchnorm1 = nn.BatchNorm1d(6)
        self.batchnorm2 = nn.BatchNorm1d(6)

    def decoder(self,latentVector):

        # latentVector = latentVector.unsqueeze(1).repeat(1, 124, 1)

        latentVector = self.lin_act1(self.lin1(latentVector))
        latentVector = self.lin_act2(self.lin2(latentVector))

        self.LSTM1.flatten_parameters()
        out,(hidden,cell) = self.LSTM1(latentVector)

        out = self.activationFunction(self.attn(hidden.permute(1,0,2)))
        # out = self.actFunc(self.outputAng(out))

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
        mu = self.mu(outAng.permute(0,2,1)).permute(0,2,1)
        sig = self.sigm_act(self.sig(outAng.permute(0,2,1)).permute(0,2,1))

        return mu, sig

    def forward(self, x):    
        mu, sig = self.decoder(x)
        
        return mu, sig
        
class NVAE(nn.Module):
    def __init__(self, config):
        super(NVAE, self).__init__()
        torch.manual_seed(0)

        self.encoder = VAE_Encoder(config).to(device)
        self.decoder = VAE_Decoder(config).to(device)

        # self.log_scale = nn.Parameter(torch.tensor([0.0]))

    def calculateDecoderGuassianLikelihood(self, inputBatch, mu_dec, var_dec):
        # Scale value ->  torch.exp(self.logScale)
        # Mean value -> reconstructionOutput
        std = torch.exp(var_dec)
        dist = Normal(loc=mu_dec, scale=std)
        logPxz = dist.log_prob(inputBatch)

        out = dist.sample()

        return out,logPxz.sum(dim=(2,1))

    def calculateOutput(self, mu, var):
        std = torch.exp(var/2)
        q = torch.distributions.Independent(Normal(loc=mu, scale=std),1)
        z = q.rsample()

        return z
    
    def forward(self, x):

        torch.manual_seed(0)
        latentVector,mu,log_var = self.encoder(x)
        lstmOut = self.calculateOutput(mu,log_var)
        mu_dec, sig_dec = self.decoder(lstmOut)
        output, reconLikelihood = self.calculateDecoderGuassianLikelihood(x, mu_dec, sig_dec)

        return output, lstmOut, latentVector, mu, log_var, reconLikelihood