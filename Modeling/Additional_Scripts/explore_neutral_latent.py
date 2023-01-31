import torch
from Robot_Feature_Extraction.Modeling.model_VAE_Laban_reg import NVAE_LabReg

import matplotlib.pyplot as plt
import numpy as np
import utilities
from construct_laban_qualities import Laban_Dict
import seaborn as sns

robot_flag = True

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Config human
dirpath = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Robot_Feature_Extraction\\Best Model Weights\\Last_Hidden\\model_LVAE_-25.874275537184726_0.pth"
config = {"encoder":{"input_dim":6,
                    "conv_blocks_num":2,
                    "conv_hidden":[9,12],
                    "conv_kernel":[7,5],
                    "batchnorm": True,
                    "lstm_hidden_dim": 25,
                    "num_lstm_layers":3,
                    "lstm_dir":False,
                    "z_dim": 30},

        "decoder":{"input_dim":30,
                    "output_dim":6,
                    "conv_blocks_num":2,
                    "conv_hidden":[12,9],
                    "conv_kernel":[5,7],
                    "batchnorm": True,
                    "lstm_hidden_dim": 25,
                    "num_lstm_layers":3,
                    "lstm_dir":False,
                    "padding":0,
                    "dilation":1,
                    "stride":1},

        "latent_reg":{"input_dim":1500,
                "output_dim":5,
                "hidden_dims":[500,100,20]}
}
model = NVAE_LabReg(config).to(dev)

checkpoint = torch.load(dirpath, map_location=torch.device(dev))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()