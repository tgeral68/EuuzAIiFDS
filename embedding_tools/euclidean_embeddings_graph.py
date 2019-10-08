import tqdm
import torch
from torch import nn

from function_tools import poincare_function, poincare_module, distribution_function, euclidean_function
from function_tools import distribution_function as df
from embedding_tools import losses
from torch import optim


class EuclideanEmbedding(nn.Module):
    def __init__(self, n_exemple, size=10,  cuda=False, lr=1e-2, verbose=True, negative_distribution=None,
                optimizer_method=optim.SGD):
        super(EuclideanEmbedding, self).__init__()
        self.cuda = cuda
        self.N = n_exemple
        self.W = nn.Embedding(n_exemple, size, sparse=True)
        if(self.cuda):
            self.W.cuda()
        self.optimizer = optimizer_method(self.W.parameters(), lr=lr)
        self.verbose = verbose
        self.d = euclidean_function.distance
        self.dim = size
        if(negative_distribution is None):
            self.n_dist = torch.distributions.Categorical(torch.ones(self.N)/self.N)
        else:
            self.n_dist = negative_distribution

    def forward(self, x):
        return self.W(x)

    def get_PoincareEmbeddings(self):
        return self.W(torch.arange(0, self.N, device=self.W.weight.data.device)).detach()

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    
    def fit(self, dataloader, alpha=1.0, beta=1.0, gamma=0.0, pi=None, mu=None, sigma=None, max_iter=100,
            negative_sampling=5):
        
        if(pi is None):
            gamma = 0.0

        else:
            if(self.cuda):
                pi = pi.cuda()
                sigma = sigma.cuda()
                mu = mu.cuda()
        progress_bar = tqdm.trange(max_iter) if(self.verbose) else range(max_iter)
        for i in progress_bar:
            loss_value1, loss_value2, loss_value3, loss_pdf3 = 0,0,0,0
            for example, neigbhors, walks in dataloader:
                self.optimizer.zero_grad()
                # obtain negatives examples sampled according to the given distribution
                with torch.no_grad():
                    negative = self.n_dist.sample(sample_shape=(walks.size(0), walks.size(1), negative_sampling))
                # set variables to cuda device
                if(self.cuda):
                    example = example.cuda()
                    neigbhors = neigbhors.cuda()
                    walks = walks.cuda()
                    negative = negative.cuda()
                # get the needed embeddings
                r_example = example.unsqueeze(1).expand_as(neigbhors)
                embed_source, embed_neigbhor = self.W(r_example), self.W(neigbhors)
                embed_source_rw  = self.W(walks)
                embed_source_rw, embed_context_rw = embed_source_rw[:,:,0], embed_source_rw[:,:,1]
                embed_negative = self.W(negative)
                # computing O1 loss
                loss_o1 = losses.SGDLoss.O1(embed_source, embed_neigbhor, distance=self.d).mean()
                # computing O2 loss
                loss_o2 = losses.SGDLoss.O2(embed_source_rw, embed_context_rw, embed_negative, distance=self.d).mean()
                # computing total loss
                loss = alpha * loss_o1 + beta * loss_o2 
                # if we want to use the prior loss
                if(gamma > 0):
                    def nfunc(sigma):
                        return df.euclidean_norm_factor(sigma, self.dim)
                    r_example = self.W(example).squeeze()
                    pi_z = pi[example].squeeze()
                    loss_o3 = losses.SGDLoss.O3(r_example, pi_z, mu, sigma, distance=self.d, zeta_f=nfunc).mean()
                    # print("loss o3 size ->", loss_o3)
                    loss += gamma * loss_o3.mean()
                    loss_value3 = loss_o3.sum(-1).mean().item()
                    loss_pdf3 = torch.exp(-loss_o3.mean()).item()

                loss_value1 = loss_o1.item()
                loss_value2 = loss_o2.item()
                loss.backward()
                self.optimizer.step()
            if(self.verbose):
                progress_bar.set_postfix({"O1":loss_value1, "O2":loss_value2, "O3":loss_value3, "PDF":loss_pdf3})