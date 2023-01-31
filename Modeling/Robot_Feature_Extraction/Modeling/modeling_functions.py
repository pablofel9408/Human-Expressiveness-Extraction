import sys 

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchsummary import torchsummary
from torch.utils.data import DataLoader

import utilities

class samplingDataLoader(Dataset):
    def __init__(self) -> None:
        pass

def covariance_estimate(latent_target, mean):

    delta = (latent_target-mean).unsqueeze(2)
    cov = (1 / (len(latent_target)-1))*torch.sum(torch.bmm(delta, torch.transpose(delta,1,2)),dim=0)
    return cov

def mahalanobis_loss(latent_x,latent_target,squared=False):
    
    mean = torch.mean(latent_target,dim=0)
    mean = mean.unsqueeze(0).repeat(len(latent_target),1)
    cov = covariance_estimate(latent_target,mean)

    # cov = torch.cov(torch.transpose(latent_target,0,1))
    # print(cov)
    # print(cov.size())
    # try:
    #     print(torch.linalg.cholesky(cov))
    # except:
    #     print("not positive definite")
    # cov = torch.inverse(cov).unsqueeze(0).repeat(len(latent_target),1,1)
    # cov = cov.unsqueeze(dim=0).repeat(len(latent_target),1,1)

    delta = (latent_x - mean).unsqueeze(2)
    d_squared = torch.bmm(torch.transpose(delta,1,2),torch.bmm(cov,delta))
    d = d_squared if not squared else torch.sqrt(d_squared)

    return torch.mean(d)

