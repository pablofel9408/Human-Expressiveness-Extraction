import torch 
import torch.nn as nn
import torch.nn.functional as F

from .attention_mechanism import Attention_Mech
from .robot_twist_generation import Twist_Generation
import utilities
from Robot_Feature_Extraction.Modeling.model_VAE_Laban_reg import NVAE_LabReg
from Robot_Feature_Extraction.Modeling.model_VAE_NLoss_Last_Hidden import NVAE

class GAN_Translation_Disc(nn.Module):
    def __init__(self, config):
        super(GAN_Translation_Disc, self).__init__()

        self.cst = config

        # Layers for prediction
        if config["hidden_dim"]:
            self.layers = nn.ModuleList([nn.Linear(config["input_size"], config["hidden_dim"][0])])
            self.layers.extend([nn.Linear(config["hidden_dim"][i-1], config["hidden_dim"][i]) for i in range(1, len(config["hidden_dim"]))])
            self.layers.append(nn.Linear(config["hidden_dim"][-1], config["output_dim"]))
        else:
            self.layers = nn.ModuleList([nn.Linear(config["input_size"], config["output_dim"])])

        for n,layer in enumerate(self.layers):
            if n >= len(self.layers):
                break
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.layers[-1].weight, gain=nn.init.calculate_gain('sigmoid'))

        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

        self.drpout = nn.Dropout(0.3)
    
    def forward(self, x):
        y = self.flatten(x)
        
        if self.cst["hidden_dim"]:
            for i in range(len(self.layers)-1):
                y = F.leaky_relu(self.layers[i](y))
                y = self.drpout(y)
        y = self.sigmoid(self.layers[-1](y))
        return y

class GAN_Translation_Gen(nn.Module):
    def __init__(self, config, device=torch.device("cpu")):
        super(GAN_Translation_Gen, self).__init__()

        self.config = config

        self.VAE_human = NVAE_LabReg(config['human_vae']["model_config"]).to(device)
        self.VAE_robot = NVAE(config["robot_vae"]["model_config"]).to(device)
        self.attn_mech = Attention_Mech(config["attention_mechanism"]).to(device)
        self.twist_gen = Twist_Generation(config["twist_generation"]).to(device)
    
    def forward(self, x_robot, x_human, x_human_neutral):

        _,mu,log_var = self.VAE_human.encoder(x_human)
        z_human = self.VAE_human.calculateOutput(mu, log_var)

        z_human = z_human - x_human_neutral

        _,mu_rob,log_var_rob = self.VAE_robot.encoder(x_robot)
        z_robot = self.VAE_robot.calculateOutput(mu_rob,log_var_rob)

        cont_att, loc_att_robot, loc_att_human = self.attn_mech(z_robot,z_human)
        out_hat_twist = self.twist_gen(cont_att, loc_att_robot,loc_att_human, 
                                        self.config["attention_mechanism"]["multihead"])

        _,mu_hat_r,log_var_hat_r = self.VAE_robot.encoder(out_hat_twist)
        z_robot_hat = self.VAE_robot.calculateOutput(mu_hat_r, log_var_hat_r)

        out_hat_twist_human = out_hat_twist.clone()
        new_acc = torch.diff(out_hat_twist_human[:,:3,:]) / 0.012
        out_hat_twist_human[:,:3,:] = torch.cat((new_acc,new_acc[:,:,-1].unsqueeze(2)),dim=2) 
        _,mu_hat,log_var_hat = self.VAE_human.encoder(out_hat_twist_human)
        z_human_hat = self.VAE_human.calculateOutput(mu_hat, log_var_hat)

        return out_hat_twist, z_human, z_robot, mu, log_var, \
                z_human_hat, z_robot_hat, mu_hat, log_var_hat, \
                    cont_att, loc_att_human, loc_att_robot