from cpwc.tools.utils import *
from cpwc.multires.class_multires import *

class ForwardInterpolation(MultiRes):
    def __init__(self, linOperator):
        self.linOperator = linOperator

    def H(self, x):
        return self.linOperator.apply_linop(x)

    def Ht(self, x):
        return self.linOperator.apply_linopT(x)
    
    def H_power(self,x):
        return self.linOperator.apply(x)
    

