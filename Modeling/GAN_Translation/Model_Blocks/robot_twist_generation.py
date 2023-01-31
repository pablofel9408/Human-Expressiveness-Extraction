from doctest import OutputChecker
import torch 
import torch.nn as nn

import utilities

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

class Twist_Generation(nn.Module):
    def __init__(self, config):
        super(Twist_Generation, self).__init__()

        self.cst = config
        self.p1d = (3, 3)

        self.lstm = nn.LSTM(input_size = self.cst["input_dim2"],
                            hidden_size = self.cst["lstm_hidden_dim"],
                            num_layers = self.cst['num_lstm_layers'],
                            batch_first=True,
                            bidirectional=self.cst['lstm_dir'])

        # Convolutional layers for feature extraction
        self.conv1 = torch.nn.ConvTranspose1d(self.cst["input_dim2"], self.cst["conv_hidden"][0], 
                                            kernel_size=self.cst["conv_kernel"][0])
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.batch1 = torch.nn.BatchNorm1d(self.cst["conv_hidden"][0])
        self.relu1 = torch.nn.LeakyReLU()

        self.conv2 = torch.nn.ConvTranspose1d(self.cst["conv_hidden"][0], self.cst["conv_hidden"][1], 
                                            kernel_size=self.cst["conv_kernel"][1])
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        # self.batch2 = torch.nn.BatchNorm1d(self.cst["conv_hidden"][1])
        self.relu2 = torch.nn.LeakyReLU()

        self.output_layer1 = nn.Linear(self.cst["input_dim2"],self.cst["input_dim2"])
        nn.init.xavier_uniform_(self.output_layer1.weight, gain=nn.init.calculate_gain('sigmoid'))
        self.output_layer2 = nn.Linear(self.cst["conv_hidden"][-1],self.cst["output_dim"])
        nn.init.xavier_uniform_(self.output_layer2.weight, gain=nn.init.calculate_gain('sigmoid'))

        self.multihead = nn.MultiheadAttention(self.cst["input_dim"],num_heads=6,batch_first=True)
        self.layer_norm = nn.LayerNorm([50,self.cst["input_dim"]])

        self.multihead_2 = nn.MultiheadAttention(self.cst["input_dim"],num_heads=6,batch_first=True)
        self.layer_norm_2 = nn.LayerNorm([50,self.cst["input_dim"]])
        self.lin_layer1 = nn.Linear(self.cst["input_dim"],self.cst["input_dim"])
        nn.init.xavier_uniform_(self.lin_layer1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.lin_layer2 = nn.Linear(self.cst["input_dim"],self.cst["input_dim"])
        nn.init.xavier_uniform_(self.lin_layer2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.relu_lin = nn.LeakyReLU()
        self.relu_lin2 = nn.LeakyReLU()

        self.layer_norm3 = nn.LayerNorm([50,self.cst["input_dim"]])
        self.layer_norm_4 = nn.LayerNorm([50,self.cst["input_dim"]])
        self.lin_layer3 = nn.Linear(self.cst["input_dim"],self.cst["input_dim"])
        nn.init.xavier_uniform_(self.lin_layer3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.lin_layer4 = nn.Linear(self.cst["input_dim"],self.cst["input_dim"])
        nn.init.xavier_uniform_(self.lin_layer4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.relu_lin3 = nn.LeakyReLU()
        self.relu_lin4 = nn.LeakyReLU()

        self.sigmoid = nn.Sigmoid()
        self.sigmoi2 = nn.Sigmoid()
        self.leaky3 = nn.LeakyReLU()

    def forward(self, cont_att, loc_att_rob, loc_att_hum, multihead=False):

        cont_att=cont_att[0]
        loc_att_rob=loc_att_rob[0]
        loc_att_hum=loc_att_hum[0]

        # out_cat = torch.cat((loc_att_rob,loc_att_hum), dim=2)
        out_cat = loc_att_rob + self.cst["lambda_gain"]*loc_att_hum
        # out_cat = torch.nn.functional.pad(out_cat,(0,0,5,5))
        # print(out_cat.size())
        
        # out_cat = torch.mul(loc_att_rob,loc_att_hum)
        # out_cat = torch.nn.functional.pad(out_cat.permute(0,2,1), 
        #                                     self.p1d,"constant", 0).permute(0,2,1)
        # out_cat = out_cat.mean(-1).unsqueeze(dim=-1)
        out_att,_ = self.multihead(out_cat,out_cat,out_cat)
        out_norm = self.layer_norm3(out_cat + out_att)
        out = self.relu_lin3(self.lin_layer3(out_norm))
        out = self.relu_lin4(self.lin_layer4(out))
        out_att1 = self.layer_norm_4(out + out_norm)

        # out_1,_ = self.multihead_2(out_att1,out_att1,out_att1)
        # out_norm1 = self.layer_norm(out_1 + out_att1)
        # out1 = self.relu_lin(self.lin_layer1(out_norm1))
        # out1 = self.relu_lin2(self.lin_layer2(out1))
        # out_att2 = self.layer_norm_2(out1 + out_norm1)
        # out = self.leaky2(self.ll(out))
      
        out_att1,(_,_) = self.lstm(out_att1)

        out3 = self.relu1(self.conv1(out_att1.permute(0,2,1)))
        out2 = self.relu2(self.conv2(out3)).permute(0,2,1)
        # out = self.leaky3(self.output_layer1(out_att2))
        out = self.sigmoid(self.output_layer2(out2)).permute(0,2,1)

        return out