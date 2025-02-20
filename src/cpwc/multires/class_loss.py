from src.cpwc.tools.utils import *
from src.cpwc.multires.class_interpolation import *


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
        return ((torch.abs(torch.sqrt(self.y) - torch.sqrt(self.F.H_power(x)))**2).sum() / 2).item()
        #return ((torch.abs(self.y - self.F.H_power(x))**2).sum() / 2).item() !!! OLD LOSS

    def calc_loss(self, x):    
        '''
        Calculates the final loss function as the sum of 
        the MSE and the regularization loss. 
        '''
        return self.calc_mse(x) 


