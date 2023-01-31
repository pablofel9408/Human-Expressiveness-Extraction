import numpy as np

import torch 

from Robot_Feature_Extraction.Modeling.model_VAE_Laban_reg import NVAE_LabReg

class LatentGeneration():
    def __init__(self, config) -> None:
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.cst = config

        self.model = NVAE_LabReg(config["model_config"]["generator"]["human_vae"]["model_config"]).to(self.dev)
        checkpoint = torch.load(self.cst["pretrained_model_path_Human_VAE"], map_location=torch.device(self.dev))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def latent_encoding(self, x_human):
        x_human = torch.tensor(x_human).to(self.dev)
        x_human = x_human.permute(0,2,1)

        _,mu,log_var = self.model.encoder(x_human.float())
        z_human = self.model.calculateOutput(mu, log_var)
        z_human = z_human.clone().detach().cpu().numpy()
        return z_human

