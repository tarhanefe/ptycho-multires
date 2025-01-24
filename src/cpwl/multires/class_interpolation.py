from cpwl.tools.utils import *
from cpwl.multires.class_multires import *
import torch.nn.functional as func

class ForwardInterpolation(MultiRes):

    #s is the local scale and S the global scale
    def __init__(self, linOperator):
        self.linOperator = linOperator

    def H(self, x):

        return self.linOperator.apply_linop(Iter(self.up, x, self.S - self.loc["s"]))

    def Ht(self, x):

        return Iter(self.up_t, self.linOperator.apply_linopT(x), self.S - self.loc["s"])
    
    def H_power(self,x):
        return self.linOperator.apply(Iter(self.up, x, self.S - self.loc["s"]))

