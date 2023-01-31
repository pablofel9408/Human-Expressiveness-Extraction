from re import A
from turtle import forward
import torch 
import torch.nn as nn
import torch.nn.functional as F
import utilities
import matplotlib.pyplot as plt

class Content_Attn(nn.Module):
    def __init__(self, config):
        super(Content_Attn, self).__init__()

        self.W_b = nn.Linear(config["seq_len"],config["seq_len"],bias=False)
        nn.init.xavier_uniform_(self.W_b.weight, gain=nn.init.calculate_gain('tanh'))
        self.W_q = nn.Linear(config["seq_len"],config["feat_dim"],bias=False)
        nn.init.xavier_uniform_(self.W_q.weight, gain=nn.init.calculate_gain('tanh'))
        self.W_v = nn.Linear(config["seq_len"],config["feat_dim"],bias=False)
        nn.init.xavier_uniform_(self.W_v.weight, gain=nn.init.calculate_gain('tanh'))

        self.w_hq = nn.Linear(config["feat_dim"],1,bias=False)
        nn.init.xavier_uniform_(self.w_hq.weight, gain=nn.init.calculate_gain('sigmoid'))

        self.tanh = nn.Tanh()
        self.tanh1 = nn.Tanh()
    
    def forward(self, z_robot,z_human):
        
        aux = torch.matmul(self.W_b.weight,z_human)
        C = self.tanh(torch.bmm(torch.transpose(z_robot,dim0=1,dim1=2),aux))
        aux2 = torch.matmul(self.W_v.weight,z_human)
        aux3 = torch.matmul(self.W_q.weight,z_robot)
        H_q = self.tanh1(aux3 + torch.bmm(aux2,torch.transpose(C,1,2)))
        a_q = F.softmax(torch.matmul(self.w_hq.weight, H_q),dim=2)
        attn = torch.bmm(z_robot,torch.transpose(a_q,1,2))

        return attn,a_q,C

class Location_Attn(nn.Module):
    def __init__(self, config):
        super(Location_Attn, self).__init__()

        # self.w_a = nn.Linear(config["seq_len"],1, bias=False)
        self.w_a = nn.Parameter(torch.randn(1,config["seq_len"],30))
        # nn.init.xavier_uniform_(self.w_a.weight, gain=nn.init.calculate_gain('sigmoid'))
    
    def forward(self, x):
        
        a_q = F.softmax(F.leaky_relu(self.w_a * x, 0.2),dim=2)
        attn = torch.matmul(x,torch.transpose(a_q,1,2))

        return attn, a_q

class Multihead_Attn(nn.Module):
    def __init__(self, config):
        super(Multihead_Attn, self).__init__()

        self.att_w = nn.MultiheadAttention(config["feat_dim"],num_heads=config["num_heads"],batch_first=True)
        self.layer_norm = nn.LayerNorm([config["seq_len"],config["feat_dim"]])
        self.layer_norm2 = nn.LayerNorm([config["seq_len"],config["feat_dim"]])
        self.linear = nn.Linear(config["feat_dim"],config["feat_dim"])
        # nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.linear2 = nn.Linear(config["feat_dim"],config["feat_dim"])
        # nn.init.xavier_uniform_(self.linear2.weight, gain=nn.init.calculate_gain('leaky_relu'))

        self.relu = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
    
    def forward(self, z_robot,z_human):
        
        out, attn_weights = self.att_w(z_robot,z_human,z_human)
        out1 = self.layer_norm(out + z_robot)
        out = self.relu(self.linear(out1))
        out = self.relu2(self.linear2(out))
        out = self.layer_norm2(out + out1)
        
        # out = out.mean(-1)

        return out, attn_weights


class Attention_Mech(nn.Module):
    def __init__(self, config):
        super(Attention_Mech, self).__init__()

        self.multihead_flag = config["multihead"]
        if not config["multihead"]:
            self.content = Content_Attn(config["content"])
            self.location_rob = Location_Attn(config["location"])
            self.location_hum = Location_Attn(config["location"])
        else:
            self.multihead = Multihead_Attn(config["multihead_config"])
            self.multihead_self1 = Multihead_Attn(config["multihead_config"])
            self.multihead_self2 = Multihead_Attn(config["multihead_config"])

    def forward(self, z_robot,z_human):
        
        if not self.multihead_flag:
            c_hat = self.content(z_robot,z_human)
            p_hat_robot = self.location_rob(z_robot)
            p_hat_human = self.location_hum(z_human)
            return c_hat, p_hat_robot, p_hat_human
        else:
            c_hat = self.multihead(z_robot,z_human)
            p_hat_human = self.multihead_self1(z_human,z_human)
            p_hat_robot = self.multihead_self2(z_robot,z_robot)
            return c_hat, p_hat_robot, p_hat_human



        