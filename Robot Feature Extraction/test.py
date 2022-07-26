import torch
import numpy as np

# a = torch.rand((2,2,1))
# print(a)
# print(a.size())
# b = torch.rand(2,2,1)
# print(b)
# print(b.size())

mu = torch.randn([5,128,12])
log_sig = torch.randn([5,128,12])
indep = torch.distributions.Independent(torch.distributions.Normal(loc=mu, scale=torch.exp(log_sig)), 1)
print(indep.batch_shape)

a = torch.randn([5,128,12])
print(indep.log_prob(a))