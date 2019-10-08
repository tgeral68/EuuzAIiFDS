import torch

from torch import nn
# Projecting data in R^n to the poincaré ball (||x||<1)
# if an examples is outside trhe circle we divide by the vector
# by norm 
EPS = 1e-4
class HyperbolicProjection(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        try :
            assert((x.data.sum() == x.data.sum()).sum()>=1)
            x_norm = x.norm(2,-1)
            # print(x_norm.max())
            x_norm[x_norm>=1.0] *= (1+1e-4)
            x_norm[x_norm<1.0] = 1
            normalized_result = x/(x_norm.unsqueeze(-1).expand_as(x))
            assert((normalized_result.data.sum() == normalized_result.data.sum()).sum()>=1)
        except:
            print(x.size())
            print(x.norm(2,-1))
            print(x.norm(2,-1).max())
            print(x.norm(2,-1).min())
        return normalized_result

    @staticmethod
    def backward(ctx, grad_output):
        try:
            assert((grad_output.mean() == grad_output.mean())>=1)
        except:
            print(grad_output)
            exit(2)
        return grad_output

class HyperbolicProjectionModule(nn.Module):
    def __init__(self):
        super(HyperbolicProjectionModule, self).__init__()
    def forward(self, x):
        return HyperbolicProjection.apply(x)    

def arc_cosh(x):
    return torch.log(x + torch.sqrt(x**2-1))

def hyperProj(x):
    return HyperbolicProjection.apply(x)

class ArcTanh(nn.Module):
    def __init__(self):
        super(ArcTanh, self).__init__()
    def forward(self, x):
        return 0.5 * torch.log((1+x)/(1-x))

class ArcSinh(nn.Module):
    def __init__(self):
        super(ArcSinh, self).__init__()
    def forward(self, x):
        return  torch.log(x + torch.sqrt(x**2 + 1))

def arcSinh(x):
    return ArcSinh()(x)

class ExpMapZero(nn.Module):
    def __init__(self):
        super(ExpMapZero, self).__init__()
    def forward(self, x):
        norm_x = x.norm(2,-1, keepdim=True).expand_as(x)
        return hyperProj((torch.tanh(norm_x)) * (x/norm_x))
    

def expMapZero(x):
    return ExpMapZero()(x)

class LogMapZero(nn.Module):
    def __init__(self):
        super(LogMapZero, self).__init__()
    def forward(self, x):
        norm_x = x.norm(2,-1, keepdim=True).expand_as(x)
        return hyperProj(((arcTanh(norm_x))) * (x/norm_x))

def logMapZero(x):
    return LogMapZero()(x)

class LogMap(nn.Module):
    def __init__(self):
        super(LogMap, self).__init__()
    def forward(self,k,  x):
        kpx = addh(-k,x)
        norm_kpx = kpx.norm(2,-1, keepdim=True).expand_as(kpx)
        norm_k = k.norm(2,-1, keepdim=True).expand_as(kpx)

        return (1-norm_k)* ((arcTanh(norm_kpx))) * (kpx/norm_kpx)

def logMap(k, x):
    return LogMap()(k, x)

class ExpMap(nn.Module):
    def __init__(self):
        super(ExpMap, self).__init__()
    def forward(self,k,  x):
        # print(k.size())
        # print(x.size())
        norm_k = k.norm(2,-1, keepdim=True).expand_as(k)
        lambda_k = 2/(1-norm_k)
        norm_x = x.norm(2,-1, keepdim=True).expand_as(x)
        direction = x/norm_x
        factor = torch.tanh((lambda_k * norm_x)/2)
        return addh(k,direction*factor)

def expMap(k, x):
    return ExpMap()(k, x)


def nonLinearityH(x, nl_func):
    return expMapZero(nl_func(logMapZero(x)))

class NonLinearityH(nn.Module):
    def __init__(self, nl_module):
        super(NonLinearityH, self).__init__()        
        self.nl_module = nl_module
    
    def forward(self, x):
        return nonLinearityH(x, self.nl_module)

class AddH(nn.Module):
    def __init__(self):
        super(AddH, self).__init__()        
    def forward(self, x, y):
        nx = torch.sum(x ** 2, dim=-1, keepdim=True).expand_as(x)
        ny = torch.sum(y ** 2, dim=-1, keepdim=True).expand_as(x)
        xy = (x * y).sum(-1, keepdim=True).expand_as(x)
        return ((1 + 2*xy+ ny)*x + (1-nx)*y)/(1+2*xy+nx*ny)

def addh(x,y):
    return AddH()(x,y)

class BAddH(nn.Module):
    def __init__(self):
        super(BAddH, self).__init__()
    def forward(self, B, x):
        # B = K \times n
        # x = Batch \times n
        # return Batch X K X n
        nB = torch.sum(B ** 2, dim=-1, keepdim=True).expand_as(B).unsqueeze(0).expand(x.size(0), B.size(0),B.size(1))
        nx = torch.sum(x ** 2, dim=-1, keepdim=True).expand_as(x).unsqueeze(1).expand(x.size(0), B.size(0),B.size(1))
        Bb = B.unsqueeze(0).expand(x.size(0), B.size(0),B.size(1))
        xb = x.unsqueeze(1).expand(x.size(0), B.size(0),B.size(1))
        xy = (xb * Bb).sum(-1, keepdim=True).expand_as(xb)
        return ((1 + 2*xy+ nx)*Bb + (1-nB)*xb)/(1+2*xy+nB*nx)        
    @staticmethod
    def test():
        b = torch.rand(10,5)
        x = torch.rand(25,5)
        m = BAddH()
        res = m(b,x)
        print(res)
        print(res.size())

class BiasH(nn.Module):
    def __init__(self, size):
        super(BiasH, self).__init__()              
        self.b = nn.Parameter(torch.randn(1, size)/(size))

    def forward(self, x):
        return addh(x, self.b.expand_as(x))
 