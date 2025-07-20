from src.cpwc.tools.utils import *
from src.cpwc.multires.class_interpolation import *
from src.cpwc.multires.class_htv import *


class Loss():
    def __init__(self,linOperator, y,lmbda = 1e-6):
        pass
        '''
        Sets the scaling factor and the input image. Creates the 
        necessary classes for the forward interpolation and the HTV.
        '''
        self.y = y
        self.lmbda = lmbda

        #classes
        self.F = ForwardInterpolation(linOperator)
        self.reg = HTV()

    def calc_mse(self, x):
        '''
        Calculates the mean squared error between the input image and the
        forward interpolation of the solution.
        '''
        return ((torch.abs(torch.sqrt(self.y) - torch.sqrt(self.F.H_power(x)))**2).sum() / 2).item()
        #return ((torch.abs(self.y - self.F.H_power(x))**2).sum() / 2).item() !!! OLD LOSS


    def calc_reg(self, x,l1_type = 'l1_row'):
        '''
        Calculates the regularization loss 
        '''
        return (self.reg.eval(x,l1_type) * self.lmbda)
    
    def calc_loss(self, x,l1_type = 'l1_row'):    
        '''
        Calculates the final loss function as the sum of 
        the MSE and the regularization loss. 
        '''
        return self.calc_mse(x) + self.calc_reg(x,l1_type)
    
    def calc_improvement(self, x1, x2):
        '''
        Calculates the improvement of the loss function between two solutions.
        '''
        return (self.calc_loss(x1) - self.calc_loss(x2)) / self.calc_loss(x2) * 100.0

