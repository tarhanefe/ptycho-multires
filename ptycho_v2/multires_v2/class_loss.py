from ptycho_v2.tools_v2.utils import *
from ptycho_v2.multires_v2.class_interpolation import *


class Loss():
    def __init__(self,linOperator, y):
        pass
        '''
        Sets the scaling factor and the input image. Creates the 
        necessary classes for the forward interpolation and the HTV.
        '''
        self.y = y
        self.F = ForwardInterpolation(linOperator)

    def calc_mse(self, x):
        '''
        Calculates the mean squared error between the input image and the
        forward interpolation of the solution.
        '''
        return ((torch.abs(self.y - self.F.H_power(x))**2).sum() / 2).item()

    def calc_loss(self, x):    
        '''
        Calculates the final loss function as the sum of 
        the MSE and the regularization loss. 
        '''
        return self.calc_mse(x) 


