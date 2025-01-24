import torch.nn.functional as F
import torch
import numpy as np
from ptycho_v2.tools_v2.utils import *


class MultiRes():

    device = "cuda"
    S = None
    size = None

    loc = {"s": 0,
           "S-s": None,
           "sigma_U": None}

    @classmethod
    def set_locals(self, scale=None, mod="reset"):

        if mod == "reset":
            if not scale is None:
                self.loc["s"] = scale
                self.loc["S-s"] = self.S - self.loc["s"]
                self.loc["sigma_U"] = np.sqrt(2.5) ** self.loc["S-s"]

        if mod == "update":
            if not scale is None:
                self.loc["s"] += scale
                self.loc["S-s"] = self.S - self.loc["s"]
                self.loc["sigma_U"] = np.sqrt(2.5) ** self.loc["S-s"]

    @classmethod
    def set_scale(self, S):
        self.S = S
        self.size = 2 ** S
        self.set_locals(scale=S)

    @classmethod
    def set_device(self, device):
        '''
        This function sets the device (CPU or GPU) for the class
        '''
        self.device = device

    def __init__(self, S, device):
        '''
        Initialize the class with the scale and device.
        '''
        self.set_device(device)
        self.set_scale(S)


    def up(self, x):
        m = torch.ones(2, 2).to(self.device)
        x = torch.kron(x, m)
        return x / 2

    def down(self, x):
        return x[:, :, 0::2, 0::2] * 2

