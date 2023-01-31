import torch
from Robot_Feature_Extraction.Modeling.model_VAE_NLoss_Last_Hidden import NVAE
from Robot_Feature_Extraction.Modeling.model_VAE_Laban_reg import NVAE_LabReg

import matplotlib.pyplot as plt
import numpy as np
import utilities
from construct_laban_qualities import Laban_Dict
import seaborn as sns

robot_flag = True

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if robot_flag:
    
    # Config robot
    dirpath = "C:\\Users\\posorio\\Documents\\Expressive movement\\Modeling\\Robot_Feature_Extraction\\Best Model Weights\\Last_Hidden\\model_NVAE_-105.62657083047402_0.pth"
    config = {  "encoder": {"input_dim":6,
                                    "conv_blocks_num":2,
                                    "conv_hidden":[15,27],
                                    "conv_kernel":[7,5],
                                    "batchnorm": True,
                                    "lstm_hidden_dim": 30,
                                    "num_lstm_layers":4,
                                    "lstm_dir":False,
                                    "z_dim": 30},

                "decoder":{"input_dim":30,
                            "output_dim":6,
                            "conv_blocks_num":2,
                            "conv_hidden":[27,15],
                            "conv_kernel":[5,7],
                            "batchnorm": True,
                            "lstm_hidden_dim": 30,
                            "num_lstm_layers":4,
                            "lstm_dir":False,
                            "padding":0,
                            "dilation":1,
                            "stride":1}
            }
    model = NVAE(config).to(dev)
else:

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
p = torch.distributions.Normal(torch.zeros(10000,50,30), torch.ones(10000,50,30))
z = p.rsample().to(dev)
# print(z.size())
sample = model.decoder(z)
print(sample.size())
sample = sample.detach().clone().cpu().numpy()
for i in range(np.shape(sample)[0]):
    sample[i] = utilities.filter_signal_2(sample[i])

laban_estm = Laban_Dict(sample)
df = laban_estm.start_process(name="generate_fake_human_data_laban", human=robot_flag)
cols = df.columns
print(df.columns)
fig, axes = plt.subplots(nrows=4, ncols=1,sharex=False, sharey=False)
for n,ax in enumerate(axes):
    sns.histplot(df, x=cols[n],ax=ax, bins=100)
fig.suptitle('Expressive Qualities Fake Human Dataset')
plt.show()
# sample = np.squeeze(sample, axis=0)

# time_samp = np.linspace(0,len(sample[0,0,:]),len( sample[0,0,:]))

# for k in range(np.shape(sample)[0]):
#     fig5, axs5 = plt.subplots(6, 1)

#     axs5[0].plot(time_samp, sample[k,0,:],linewidth=2.0)
#     axs5[1].plot(time_samp, sample[k,1,:],linewidth=2.0)
#     axs5[2].plot(time_samp, sample[k,2,:],linewidth=2.0)
#     axs5[3].plot(time_samp, sample[k,3,:],linewidth=2.0)
#     axs5[4].plot(time_samp, sample[k,4,:],linewidth=2.0)
#     axs5[5].plot(time_samp, sample[k,5,:],linewidth=2.0)

# plt.show()

