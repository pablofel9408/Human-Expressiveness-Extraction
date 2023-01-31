from tempfile import tempdir
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf
from torch.nn.modules.padding import ConstantPad1d
import constants

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.LSTM1 = nn.LSTM(
            input_size = 64,
            hidden_size = 128,
            num_layers = 1,
            batch_first=True
        )

        self.attn = nn.Linear(
            in_features=24,
            out_features=64
        )

        self.convAcc = nn.Conv1d(
            in_channels=3,
            out_channels=12,
            kernel_size=3
        )

        self.convAng = nn.Conv1d(
            in_channels=3,
            out_channels=12,
            kernel_size=3
        )

        self.avgPool = nn.AvgPool1d(3,stride=2)

        # self.actFunc = nn.ReLU()
        # self.attActFunc = nn.Softmax(dim=2)
        if config['activation_function'] != 'linear':
            self.actFunc = config['activation_function']
        else:
            self.actFunc = nn.Linear(
                                in_features=config['convolutional_layer'],
                                out_features=config['convolutional_layer']
                            )

        if config['attn_activation_function'] != 'linear':
            self.attActFunc = config['attn_activation_function']
        else:
            self.attActFunc = 'linear'

        #self.attActFunc = nn.ReLU()

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample

    def forward(self, x):    
        # Split inputs
        inputAcc = x[:,3:,:]
        inputAngVel = x[:,:3,:]

        # Run feature selector, concatenate outputs and run attention
        outAcc = self.actFunc(self.convAcc(inputAcc))
        outAng = self.actFunc(self.convAcc(inputAngVel))
        outAcc = outAcc.permute(0,2,1)
        outAng = outAng.permute(0,2,1)
        out = torch.cat((outAng,outAcc),2)
        print(out.shape)
        # out = torch.flatten(out)

        if constants.printInNetwork:
            print('Shape before attention:', out.shape)

        if self.attActFunc=='linear':
            out = self.attn(out)
        else:
            out = self.attActFunc(self.attn(out))
            #out = out.permute(0,2,1)
            #out = self.avgPool(out)
            #out = out.permute(0,2,1)
            print(out.shape)
        # LSTM Encoding
        out, (hidden_state, cell_state) = self.LSTM1(out)  
        last_lstm_layer_hidden_state = hidden_state[-1,:,:]
        # last_lstm_layer_hidden_state = last_lstm_layer_hidden_state.view(-1,2,128)
        # get `mu` and `log_var`
        # mu = last_lstm_layer_hidden_state[:, 0, :] # the first feature values as mean
        # log_var = last_lstm_layer_hidden_state[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        # z = self.reparameterize(mu, log_var)

        # Print shapes of middle variables if need to check
        if constants.printInNetwork:
            print('Shape after convolution Acc:', outAcc.shape)
            print('Shape after convolution:', outAng.shape)
            print('Shape after LSTM:', out.shape)
            print('Shape hidden state:', last_lstm_layer_hidden_state.shape)

        return last_lstm_layer_hidden_state, out


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.LSTM1 = nn.LSTM(
            input_size = 128,
            hidden_size = 64,
            num_layers = 1,
            batch_first=True
        )

        self.attn = nn.Linear(
            in_features=64,
            out_features=24
        )

        self.deConvAcc = nn.ConvTranspose1d(
            in_channels=12,
            out_channels=6,
            kernel_size=4,
        )

        self.deConvAng = nn.ConvTranspose1d(
            in_channels=12,
            out_channels=6,
            kernel_size=4
        )    

        self.actFunc = nn.ReLU()
        self.attActFunc = nn.ReLU()
    
        # if config['attn_activation_function'] != 'linear':
        #     self.attActFunc = config['attn_activation_function']
        # else:
        #     self.attActFunc = 'linear'

        # if config['activation_function'] != 'linear':
        #     self.actFunc = config['activation_function']
        # else:
        #     self.actFunc = nn.Linear(
        #                         in_features=config['input_channels'],
        #                         out_features=config['input_channels']
                            # )
        self.output = nn.Linear(
                                in_features=12,
                                out_features=6
                            )
        #self.attActFunc = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        x = x.unsqueeze(1).repeat(1, 126, 1)
        x, (hidden_state, cell_state) = self.LSTM1(x)
        out = self.attActFunc(self.attn(x))
        outAcc = out[:,:,:12].permute(0,2,1)
        outAng = out[:,:,12:].permute(0,2,1)

        if constants.printInNetwork:
            print('Acc Shape:', outAcc.shape)
            print('Ang Shape:', outAng.shape)

        outAcc = self.actFunc(self.deConvAcc(outAcc))
        outAng = self.actFunc(self.deConvAng(outAng))

        if constants.printInNetwork:
            print('Acc Shape Before Cat:', outAcc.shape)
            print('Ang Shape Before Cat:', outAng.shape)

        out = torch.cat((outAng,outAcc),1)
        out = out.permute(0,2,1)
        out = self.output(out)
        out = out.permute(0,2,1)

        if constants.printInNetwork:
            print('Output shape', out.shape)

        #out = self.sigmoid(out)
        return out 

class LSTM_AE(nn.Module):
    def __init__(self, config):
        super(LSTM_AE, self).__init__()

        self.encoder = Encoder(config).to(device)
        self.decoder = Decoder(config).to(device)

    def forward(self, x):
        torch.manual_seed(0)
        latent, out = self.encoder(x)
        output = self.decoder(latent)
        return output, latent, out
