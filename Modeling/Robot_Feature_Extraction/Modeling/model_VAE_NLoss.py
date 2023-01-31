import sys
import random

import numpy as np

import torch
from torch import nn
from torch.distributions import Independent,Normal

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)


        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.transpose(0,1).contiguous()   # (timesteps, samples, output_size)

        return y

class VAE_Encoder(nn.Module):
    def __init__(self, config):
        super(VAE_Encoder, self).__init__()

        self.cst = config
        self.num_directions = 2 if self.cst['lstm_dir'] else 1

        # Convolutional layers for feature extraction
        # self.convBlocks = nn.Sequential()
        self.conv1 = torch.nn.Conv1d(self.cst["input_dim"], self.cst["conv_hidden"][0], 
                                            kernel_size=self.cst["conv_kernel"][0])
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
        self.batch1 = torch.nn.BatchNorm1d(self.cst["conv_hidden"][0])
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv1d(self.cst["conv_hidden"][0], self.cst["conv_hidden"][1], 
                                            kernel_size=self.cst["conv_kernel"][1])
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu'))
        self.batch2 = torch.nn.BatchNorm1d(self.cst["conv_hidden"][1])
        self.relu2 = torch.nn.ReLU()
        # next_dim = self.cst["input_dim"]
        # for i in range(self.cst['conv_blocks_num']):
        #     self.convBlocks.add_module("conv_"+str(i), torch.nn.Conv1d(next_dim, self.cst["conv_hidden"][i], 
        #                                                                 kernel_size=self.cst["conv_kernel"][i]))
        #     if self.cst["batchnorm"]:
        #         self.convBlocks.add_module("batchnorm_"+str(i), torch.nn.BatchNorm1d(self.cst["conv_hidden"][i]))

        #     self.convBlocks.add_module("relu_"+str(i), torch.nn.ReLU())
        #     next_dim = self.cst["conv_hidden"][i]

        # LSTM layers
        self.LSTM1 = nn.LSTM(
            input_size = self.cst["conv_hidden"][1],
            hidden_size = self.cst["lstm_hidden_dim"],
            num_layers = self.cst['num_lstm_layers'],
            batch_first=True,
            bidirectional=self.cst['lstm_dir']
        )

        # Latent vector representation
        self.mu = nn.Linear(self.cst["lstm_hidden_dim"]*self.cst['num_lstm_layers'],self.cst["z_dim"])
        # nn.init.xavier_uniform_(self.mu.weight, gain=nn.init.calculate_gain('linear'))

        self.sigma = nn.Linear(self.cst["lstm_hidden_dim"]*self.cst['num_lstm_layers'],self.cst["z_dim"])
        # nn.init.xavier_uniform_(self.sigma.weight, gain=nn.init.calculate_gain('linear'))

        self.flat = nn.Flatten()
        self.act_sigm = nn.Softplus()

    def _flatten(self, h, batch_size):
        return h.transpose(0,1).contiguous().view(batch_size, -1)

    def init_hidden(self, batch_size):
        self.hidden = (torch.zeros(self.cst['num_lstm_layers'] * self.num_directions,
                        batch_size, self.cst['lstm_hidden_dim']).to(device),
                    torch.zeros(self.cst['num_lstm_layers'] * self.num_directions, 
                        batch_size, self.cst['lstm_hidden_dim']).to(device))

    def encoder(self,inputSignals):

        # Run convolutional feature extraction
        convOut = self.relu1(self.batch1(self.conv1(inputSignals)))
        convOut = self.relu2(self.batch2(self.conv2(convOut))).permute(0,2,1)
        # convOut = self.convBlocks(inputSignals).permute(0,2,1)
        # print(self.convBlocks.conv_0.weight.size())
        # utilities.vis_latent_space(convOut,self.convBlocks.conv_0.weight)
        # utilities.vis_latent_space(convOut,self.convBlocks.conv_1.weight)

        # self.LSTM1.flatten_parameters()
        _, hidden = self.LSTM1(convOut)
        # utilities.vis_latent_space_lstm(self.hidden)

        out = self._flatten(hidden[0], inputSignals.shape[0])
        # out = self.flat(hidden[0].permute(1,0,2))
        # out =  torch.cat([self._flatten(self.hidden[0], len(inputSignals)), 
        #                     self._flatten(self.hidden[1], len(inputSignals))], 1)
    
        mu = self.mu(out)
        log_var = self.act_sigm(self.sigma(out))

        return out, mu, log_var

    def forward(self, x):    
        
        latentVector,mu,log_var = self.encoder(x)

        return latentVector, mu, log_var

class VAE_Decoder(nn.Module):
    def __init__(self, config):
        super(VAE_Decoder, self).__init__()

        # Decoder config
        self.cst = config
        self.num_directions = 2 if self.cst['lstm_dir'] else 1

        # Convolutional layers for feature extraction
        self.conv1 = torch.nn.ConvTranspose1d(self.cst["lstm_hidden_dim"], self.cst["conv_hidden"][0], 
                                            kernel_size=self.cst["conv_kernel"][0])
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
        self.batch1 = torch.nn.BatchNorm1d(self.cst["conv_hidden"][0])
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.ConvTranspose1d(self.cst["conv_hidden"][0], self.cst["conv_hidden"][1], 
                                            kernel_size=self.cst["conv_kernel"][1])
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu'))
        self.batch2 = torch.nn.BatchNorm1d(self.cst["conv_hidden"][1])
        self.relu2 = torch.nn.ReLU()
        # self.convBlocks = nn.Sequential()
        # next_dim = self.cst["lstm_hidden_dim"]
        # for i in range(self.cst['conv_blocks_num']):
        #     self.convBlocks.add_module("conv_"+str(i), torch.nn.ConvTranspose1d(next_dim, self.cst["conv_hidden"][i], 
        #                                                                 kernel_size=self.cst["conv_kernel"][i]))
        #     if self.cst["batchnorm"]:
        #         self.convBlocks.add_module("batchnorm_"+str(i), torch.nn.BatchNorm1d(self.cst["conv_hidden"][i]))

        #     self.convBlocks.add_module("relu_"+str(i), torch.nn.Softplus())
        #     next_dim = self.cst["conv_hidden"][i]

        self.outputLayer = TimeDistributed(nn.Linear(self.cst["conv_hidden"][1],self.cst["output_dim"]),batch_first=True)

        # LSTM layers
        self.LSTM1 = nn.LSTM(
            input_size = self.cst["input_dim"],
            hidden_size = self.cst["lstm_hidden_dim"],
            num_layers = self.cst['num_lstm_layers'],
            batch_first=True,
            bidirectional=self.cst['lstm_dir']
        )

    def get_seq_len(self):

        l_dim = 60
        for kernel in self.cst["conv_kernel"][::-1]:
            l_dim = ((l_dim+2*self.cst["padding"]-self.cst["dilation"]*(kernel-1)-1)/self.cst["stride"])+1

        return int(l_dim)

    def init_hidden(self, batch_size):
        self.hidden = (torch.zeros(self.cst['num_lstm_layers'] * self.num_directions,
                        batch_size, self.cst['lstm_hidden_dim']).to(device),
                    torch.zeros(self.cst['num_lstm_layers'] * self.num_directions, 
                        batch_size, self.cst['lstm_hidden_dim']).to(device))
    
    def _unflatten_hidden(self, X, batch_size):
        X_split = torch.split(X, int(X.shape[1]/2), dim=1)
        return (self._unflatten(X_split[0], batch_size), self._unflatten(X_split[1], batch_size))

    def _unflatten(self, X, batch_size):
        # (batch_size, num_directions*num_layers*hidden_dim)    ==>
        # (batch_size, num_directions * num_layers, hidden_dim) ==>
        # (num_layers * num_directions, batch_size, hidden_dim) ==>
        return X.view(batch_size, self.cst['num_lstm_layers'] * self.num_directions, 
                        self.cst['lstm_hidden_dim']).transpose(0, 1).contiguous()

    def decoder(self,latentVector):
        
        # latentVector = latentVector.unsqueeze(1).repeat(1, 124, 1)
        # latentVector = latentVector.repeat(1,10,1)
        latentVector = latentVector.unsqueeze(1).repeat(1,self.get_seq_len(),1)

        # self.LSTM1.flatten_parameters()
        out,(hidden,cell) = self.LSTM1(latentVector)

        convOut = self.relu1(self.batch1(self.conv1(out.permute(0,2,1))))
        convOut = self.relu2(self.batch2(self.conv2(convOut)).permute(0,2,1))
        output = self.outputLayer(convOut).permute(0,2,1)

        return output

    def forward(self, x):    
        mu = self.decoder(x)
        
        return mu
        
class NVAE(nn.Module):
    def __init__(self, config, device=torch.device("cpu")):
        super(NVAE, self).__init__()

        self.encoder = VAE_Encoder(config['encoder']).to(device)
        self.decoder = VAE_Decoder(config['decoder']).to(device)

        self.log_scale = nn.Parameter(torch.tensor([0.0]))

    def calculateDecoderGuassianLikelihood(self, inputBatch, mu_dec):
        # Scale value ->  torch.exp(self.logScale)
        # Mean value -> reconstructionOutput
        std = torch.exp(self.log_scale)
        dist = Normal(loc=mu_dec, scale=std)
        logPxz = dist.log_prob(inputBatch)

        # out = dist.sample()
        return logPxz.sum(dim=(1,2))

    def calculateOutput(self, mu, var):
        std = torch.exp(var/2)
        q = Normal(loc=mu, scale=std)
        z = q.rsample()
        # utilities.vis_lat_var(z)
        # utilities.close_script()
        return z
    
    def forward(self, x):

        latentVector,mu,log_var = self.encoder(x)
        lstmOut = self.calculateOutput(mu,log_var)
        mu_dec = self.decoder(lstmOut)
        reconLikelihood = self.calculateDecoderGuassianLikelihood(x, mu_dec)
        # output, reconLikelihood = self.calculateReconstructLoss(x, mu_dec)

        return mu_dec, lstmOut, latentVector, mu, log_var, reconLikelihood