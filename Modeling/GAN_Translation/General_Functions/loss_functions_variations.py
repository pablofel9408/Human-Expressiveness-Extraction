import inspect
import torch 
from torch import linalg as LA

def kl_divergence(z_hat=torch.zeros((1,2)), z=torch.zeros((1,2)), 
                    log_var=torch.zeros((1,2)), log_var_hat=torch.zeros((1,2)),
                    mu_hat=torch.zeros((1,2)), mu=torch.zeros((1,2))):

    std = torch.exp(log_var/2)
    std_hat = torch.exp(log_var_hat/2)
    # print("Standard deviation shape:", std.shape)
    # print("Mean shape:", mu.shape)
    # print("Sample shape:", z.shape)
    # p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    # q = torch.distributions.Normal(mu, std)
    
    p = torch.distributions.Normal(mu_hat, std_hat)
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_p_zh_xh = p.log_prob(z_hat)
    # print("Log probability in q of z given x shape:", log_qzx.shape)
    # print("Log probability of z in p shape:", log_pz.shape)

    # kl
    # kl = log_qzx.exp()*(log_qzx - log_p_zh_xh)
    # kl = (log_qzx - log_p_zh_xh).sum(dim=(2,1)).mean()
    kl = torch.distributions.kl.kl_divergence(p,q).mean()
    
    # print("Log probability difference shape:", kl.shape)
    # kl = kl.squeeze(2).sum(-1)
    # kl = kl.sum(dim=(1,2))
    # kl = kl.sum(dim=(2,1,0)) / log_qzx.size(0)
    # print(kl)
    # print("Kullback divergance shape:", kl.shape)
    return kl, log_qzx, log_p_zh_xh

def local_isometry(z_hat=torch.zeros((1,2,3)), z=torch.zeros((1,2,3)), 
                    x_hat=torch.zeros((1,2,4)), x=torch.zeros((1,2,4)), alpha=0.5):

    norm_z = LA.matrix_norm(z_hat-z, dim=(2,1))
    norm_x = LA.matrix_norm(x_hat-x,dim=(1,2))
    isometry = alpha*(LA.norm((norm_z-norm_x)))

    return isometry

def local_isometry_mse(z_hat=torch.zeros((1,2,3)), z=torch.zeros((1,2,3)), 
                    x_hat=torch.zeros((1,2,4)), x=torch.zeros((1,2,4)), alpha=0.5):

    norm_z = torch.nn.MSELoss(reduction="none")(z_hat,z).mean(dim=(2,1))
    # norm_z = LA.matrix_norm(z_hat-z, dim=(2,1)).mean()
    norm_x = torch.nn.MSELoss(reduction="none")(x_hat,x).mean(dim=(1,2))
    isometry = alpha*LA.norm(norm_z-norm_x)

    return isometry

def change_ratio(z=torch.zeros((1,2,3)), x_hat=torch.zeros((1,2,4)), 
                    x=torch.zeros((1,2,4)), z_hat=torch.zeros((1,2,4))):

    huber_out = torch.nn.HuberLoss(reduction='none')(x_hat,x)
    huber_out = torch.sum(huber_out,dim=2)
    huber_out = torch.mean(huber_out,dim=1)
    # norm1 = torch.nn.MSELoss(reduction='mean')(z_hat,z)
    norm1 = LA.norm(z_hat - z, dim=1, ord=2)
    norm1 = torch.mean(norm1,dim=1)
    
    output = huber_out / norm1 
    # output = ratio.mean()
    
    return -output.mean()

def covariance_estimate(latent_target, mean):

    delta = (latent_target-mean).unsqueeze(2)
    cov = (1 / (len(latent_target)-1))*torch.sum(torch.bmm(delta, torch.transpose(delta,1,2)),dim=0)
    return cov

def mahalanobis_loss(z_hat=torch.zeros((1,2)), z=torch.zeros((1,2)),squared=False):
    
    mean = torch.mean(z,dim=0)
    mean = mean.unsqueeze(0).repeat(len(z),1)
    cov = covariance_estimate(z,mean)

    # cov = torch.cov(torch.transpose(latent_target,0,1))
    # print(cov)
    # print(cov.size())
    # try:
    #     print(torch.linalg.cholesky(cov))
    # except:
    #     print("not positive definite")
    # cov = torch.inverse(cov).unsqueeze(0).repeat(len(latent_target),1,1)
    # cov = cov.unsqueeze(dim=0).repeat(len(latent_target),1,1)

    delta = (z_hat - mean).unsqueeze(2)
    d_squared = torch.bmm(torch.transpose(delta,1,2),torch.bmm(cov,delta))
    d = d_squared if not squared else torch.sqrt(d_squared)

    return d

def mean_squared_error(z_hat=torch.zeros((1,2)), z=torch.zeros((1,2))):
    return torch.nn.MSELoss()(z_hat,z)

def cosine_similarity(z_hat=torch.zeros((1,2)), z=torch.zeros((1,2))):
    return torch.mean(torch.sum(torch.nn.CosineSimilarity(dim=-1, eps=1e-6)(z_hat, z),dim=-1))

def calc_l1_norm(loss, model, l1_lambda = 0.001):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return loss + l1_lambda * l1_norm

def funct_match(label, **kwargs):

    switcher = {
                'cosine similarity': cosine_similarity,
                'mse': mean_squared_error,
                "kld": kl_divergence,
                "mahalanobis": mahalanobis_loss,
                "local isometry": local_isometry,
                "local isometry mse": local_isometry_mse,
                "change ratio": change_ratio
            }
    filtered_mydict = {k: v for k, v in kwargs.items() if k in [p.name for p in inspect.signature(switcher[label]).parameters.values()]}

    return switcher.get(label, "nothing")(**filtered_mydict)