import torch
import numpy as np

# a = torch.rand((2,2,1))
# print(a)
# print(a.size())
# b = torch.rand(2,2,1)
# print(b)
# print(b.size())

# mu = torch.randn([5,128,12])
# log_sig = torch.randn([5,128,12])
# indep = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=torch.exp(log_sig)), 1)
# print(indep.batch_shape)

# a = torch.randn([5,128,12])
# print(indep.log_prob(a))
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler, RandomSampler
import tqdm 

class CustomDataset(Dataset):
    def __init__(self, input_data):
        self.data = input_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data[idx]
        return label


# latent_x = torch.randn((4,2))
# indices = torch.randint(0, 3, (10,))
# print(indices)
# print(latent_x)
# dataloader_classic = DataLoader(CustomDataset(latent_x),batch_size=2)
# dataloader = DataLoader(CustomDataset(indices),shuffle=False,batch_size=2)
# for n,aa in tqdm.tqdm(enumerate(dataloader)):
#     print(aa)
#     xx = latent_x[aa]
#     print(xx)

#     if n % 5==0 and n!=0:
#         break

# for n,aa in tqdm.tqdm(enumerate(dataloader_classic)):
#     print(aa)
#     break

def covariance_estimate(latent_target, mean):

    delta = (latent_target-mean).unsqueeze(2)
    cov = (1 / (len(latent_target)-1))*torch.sum(torch.bmm(delta, torch.transpose(delta,1,2)),dim=0)
    return cov

def mahalanobis_loss(latent_x,latent_target,squared=False):
    
    mean = torch.mean(latent_target,dim=0)
    mean = mean.unsqueeze(0).repeat(len(latent_target),1)
    cov = covariance_estimate(latent_target,mean)

    # cov = torch.cov(torch.transpose(latent_target,0,1))
    print(cov)
    print(cov.size())
    try:
        print(torch.linalg.cholesky(cov))
    except:
        print("not positive definite")
    cov = torch.inverse(cov).unsqueeze(0).repeat(len(latent_target),1,1)
    # cov = cov.unsqueeze(dim=0).repeat(len(latent_target),1,1)

    delta = (latent_x - mean).unsqueeze(2)
    d_squared = torch.bmm(torch.transpose(delta,1,2),torch.bmm(cov,delta))
    d = d_squared if not squared else torch.sqrt(d_squared)

    return torch.mean(d)

aa = torch.randn((2,3))
bb = torch.randn((2,3))
d = mahalanobis_loss(aa,bb)
print(d)