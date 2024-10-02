from utils import *
from class_multires import *
import torch.nn.functional as func

class ForwardInterpolation(MultiRes):

    #s is the local scale and S the global scale
    def __init__(self, mask):
        self.mask = mask

    def H(self, x):

        return self.mask.apply_linop(Iter(self.up, x, self.S - self.loc["s"]))

    def Ht(self, x):

        return Iter(self.up_t, self.mask.apply_linopT(x), self.S - self.loc["s"])