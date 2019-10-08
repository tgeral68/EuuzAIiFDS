import numpy as np
import math
import cmath
def aphi(sigma):
    return math.pow(sigma,2)+math.pow(sigma,4) + \
        ( ( ((math.pow(sigma,3))*math.sqrt(2)*math.exp(-(math.pow(sigma,2))/2)))/(math.sqrt(math.pi)*math.erf(sigma/(math.sqrt(2)))) )

sigma_pos = np.linspace(0.01,10, 1600) 
sigma_inverse = np.array([aphi(sigma_pos[i].item()) for i in range(len(sigma_pos))])

class RiemannianFunction(object):
    PI_CST = pow((2*math.pi),2/3)
    SQRT_CST = math.sqrt(2)
    ERF_CST = 8.0/(3.0*np.pi)*(np.pi-3.0)/(4.0-np.pi)
 
    @staticmethod
    def riemannian_distance(x, y):
        a = abs((y-x)/(1-x.conjugate()*y))
        num = 1 + a
        den = 1 - a
        return 0.5 * np.log(num/den)
        # error function 
    @staticmethod
    def erf(x):
        return np.sign(x)*np.sqrt(1-np.exp(-x*x*(4/np.pi+RiemannianFunction.ERF_CST*x*x)/(1+RiemannianFunction.ERF_CST*x**2)))
    
    @staticmethod
    def normalization_factor(sigma):
        return RiemannianFunction.PI_CST * sigma * np.exp((sigma**2)/2)*RiemannianFunction.erf(sigma/(RiemannianFunction.SQRT_CST))

    @staticmethod
    def phi(value):
        if((sigma_inverse>value).sum().item() < 1):
            print("Too low variance....")
            return sigma_pos[0]
        return sigma_pos[(sigma_inverse>value).nonzero()[0][0]]

    # log map similar to the one use in the previous code 
    @staticmethod
    def log(z, y):
        q = ((y-z)/(1-z.conjugate()*y))
        return (1 - np.abs(z) **2) * np.arctanh(np.abs(q)) * (q/np.abs(q))

    # no change only need on one component
    @staticmethod
    def exp(z, v):
        v_square = abs(v)/(1-abs(z)*abs(z))

        theta = np.angle(v)

        numerator = (z + cmath.exp(1j * theta)) * cmath.exp(2 * v_square) + (z - cmath.exp(1j * theta))
        denominator = (1 + z.conjugate() * cmath.exp(1j * theta)) * cmath.exp(2 * v_square) + (1 - z.conjugate() * cmath.exp(1j * theta))
        result1 = numerator / denominator

        result = result1.real + result1.imag * 1j

        return result