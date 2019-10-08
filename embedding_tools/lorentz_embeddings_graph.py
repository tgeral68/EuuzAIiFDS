import tqdm
import torch
from torch import nn
from torch import optim
from function_tools import torch_function, lorentz_function, lorentz_module
from optim_tools import optimizer


class RiemannianEmbedding(nn.Module):
    def __init__(self, n_exemple, cuda=False, lr=1e-2, verbose=True):
        super(RiemannianEmbedding, self).__init__()
        self.cuda = cuda
        self.N = n_exemple
        self.W = lorentz_module.LorentzEmbedding(n_exemple, 3)
        if(self.cuda):
            self.W.cuda()
        self.optimizer = optimizer.LorentzModelSGD(self.W.parameters(), lr=lr)
        self.verbose = verbose
        self.d = lorentz_function.lorentzian_distance
    def forward(self, x):
        return self.W(x)
    def get_lorentzEmbeddings(self):
        return self.W.l_embed.weight.data
    def get_PoincareEmbeddings(self):
        return lorentz_function.lorentzToPoincare(self.W.l_embed.weight.data)

    def fit(self, dataloader, alpha=1.0, beta=1.0, gamma=0.0, pi=None, mu=None, sigma=None, max_iter=100):
        progress_bar = tqdm.trange(max_iter) if(self.verbose) else range(max_iter)
        for i in progress_bar:
            loss_value1, loss_value2 = 0,0
            for example, neigbhors, walks in dataloader:
                self.optimizer.zero_grad()
                if(self.cuda):
                    example = example.cuda()
                    neigbhors = neigbhors.cuda()
                    walks = walks.cuda()
                r_example = example.unsqueeze(1).expand_as(neigbhors)
                me, mw = self.W(r_example), self.W(neigbhors)
                # print(neigbhors, print(r_example))
                # print(-self.d(me, mw))
                # print("ldsqjiqsdf")
                loss_o1 = -(torch.log(torch.exp(-self.d(me, mw)))).sum(-1).sum(-1).mean()

                # print(loss_o1)
                # O_2        
                r_example = example.unsqueeze(1).expand_as(walks)
                me, mw = self.W(r_example), self.W(walks)
                positive_d = (self.d(me, mw))
                # print(positive_d)
                me = me.expand(walks.size(0), walks.size(1),  5, mw.size(-1)).contiguous()
                negative = (torch.rand(walks.size(0), walks.size(1), 5) * self.N)
                if(self.cuda):
                    negative = negative.cuda()
                negative = self.W(negative.long())

                negative_d = self.d(me, negative)
                # print(negative_d)
                loss_o2 = torch.log( 1 + (torch.exp(-(negative_d - positive_d.expand_as(negative_d)))).sum(-1)).mean()
                # print(loss_o2)
                if(gamma > 0):
                    r_example = self.W(example).squeeze()
                    p_example = pi.unsqueeze(0).expand(len(example), len(mu))
                    loss_o3 = (-torch.log( weighted_gmm_pdf(p_example, r_example, mu, sigma, self.d))).mean()
                loss = alpha * loss_o1 + beta * loss_o2 

                loss_value1 = loss_o1.item()
                loss_value2 = loss_o2.item()
                loss.backward()
                self.optimizer.step()
            if(self.verbose):
                progress_bar.set_postfix({"loss":beta *loss_value2})