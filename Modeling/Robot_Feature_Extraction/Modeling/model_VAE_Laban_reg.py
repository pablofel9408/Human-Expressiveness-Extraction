import sys
import random

import numpy as np

import torch
from torch import nn
from torch.distributions import Independent,Normal

import utilities
from .model_VAE_NLoss_Last_Hidden import VAE_Encoder, VAE_Decoder

#Previous best: model_LVAE_-25.874275537184726_0.pth

class LabanReg(nn.Module):
    def __init__(self, config):
        super(LabanReg, self).__init__()

        # Laban Regressor config
        self.cst = config

        # Layers for prediction
        self.linear1 = torch.nn.Linear(config["input_dim"],config["hidden_dims"][0])
        self.relu1 = torch.nn.ReLU()
        
        self.linear2 = torch.nn.Linear(config["hidden_dims"][0],config["output_dim"])
        self.relu2 = torch.nn.ReLU()

        self.linear3 = torch.nn.Linear(config["hidden_dims"][1],config["output_dim"])
        self.relu3 = torch.nn.ReLU()

        self.linear4 = torch.nn.Linear(config["hidden_dims"][2],config["output_dim"])
        self.relu4 = torch.nn.ReLU()

        self.flatten = nn.Flatten()
    
    def forward(self, x):
        
        out = self.flatten(x)
        out = self.relu1(self.linear1(out))
        out = self.linear2(out)
        # out = self.linear3(out)
        # out = self.relu4(self.linear4(out))

        return out

class NVAE_LabReg(nn.Module):
    def __init__(self, config, device=torch.device("cpu")):
        super(NVAE_LabReg, self).__init__()

        self.dev = device
        self.encoder = VAE_Encoder(config['encoder']).to(device)
        self.decoder = VAE_Decoder(config['decoder']).to(device)
        self.laban_reg = LabanReg(config["latent_reg"]).to(device)

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
        laban_out = self.laban_reg(lstmOut)
        # laban_out = torch.zeros(lstmOut.size()[0],5).to(self.dev)
        mu_dec = self.decoder(lstmOut)
        reconLikelihood = self.calculateDecoderGuassianLikelihood(x, mu_dec)
        # output, reconLikelihood = self.calculateReconstructLoss(x, mu_dec)

        return mu_dec, lstmOut, latentVector, mu, log_var, reconLikelihood, laban_out