from utils import *
from class_htv import *
from class_interpolation import *


class Loss():

    def __init__(self, mask, y, lmbda=1e-6):
        '''
        Sets the scaling factor and the input image. Creates the 
        necessary classes for the forward interpolation and the HTV.
        '''
        #parameters
        self.y = y
        self.lmbda = lmbda

        #classes
        self.F = ForwardInterpolation(mask)
        self.reg = HTV()

    def calc_mse(self, x):
        '''
        Calculates the mean squared error between the input image and the
        forward interpolation of the solution.
        '''
        return ((torch.abs(self.y - self.F.H(x)) ** 2).sum() / 2).item()

    def calc_reg(self, x,l1_type = 'l1_row'):
        '''
        Calculates the regularization loss 
        '''
        return (self.reg.eval(x,l1_type) * self.lmbda)

    def calc_loss(self, x, l1_type='l1_row'):
        '''
        Calculates the final loss function as the sum of 
        the MSE and the regularization loss. 
        '''
        return self.calc_mse(x) + self.calc_reg(x,l1_type)


