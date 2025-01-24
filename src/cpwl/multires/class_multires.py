import torch.nn.functional as F
import torch
import numpy as np
from ptycho.tools.utils import *


class MultiRes():

    device = "cuda"
    # The filter is the structure of the simplex.
    '''
    [0 0.5 0.5
     0.5 1 0.5
     0.5 0.5 0]

     Which denotes the coefficients for each spline to go to a coarser scale 
    '''
    filter = torch.Tensor([[[[0., 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 0.]]]]).to(device).double()
    S = None
    size = None

    # local variables for multiresolution purpose
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

        '''
        Input: x // shape:  (1,1,width,height)  // type: torch.float64

        Output: x_up_m // shape : (1,1,2*width+1,2*height+1) // type: torch.float64

        Upsampling by applying the transpose convolution operation 
        with the filter: 
        
        [[0, 0.5, 0.5],
         [0.5, 1, 0.5],
         [0.5, 0.5, 0]]

         This operation acts like a classical zero padded upsampling
         followed by a convolution with the LPF filter. More spesfically 
         A sinc interpolator. 
        '''
        return cmpx_conv_transpose2d(x, self.filter, padding=0, stride=2)

    def down(self, x):
        '''
        Input : x // shape:  (1,1,width,height)  // type: torch.float64

        Output: shape : (1,1,width+1/2,height+1/2) // type: torch.float64

        Simple downsampling operation for the multiresolution images
        '''
        # Downsampling operator for the multiresoliution images
        return x[:, :, 0::2, 0::2]
    
    def up_t(self, x):
        '''
        Input: x // shape:  (1,1,width,height)  // type: torch.float64

        Output: x_down_m // shape : (1,1,width-1/2,height-1/2) // type: torch.float64

        This function will downsample the image with interpolating it with a 
        LPF. This gives the function parameters in a coarser scale. 
        '''
        x_down_m = cmpx_conv2d(x, self.filter, padding=0, stride=2)
        return x_down_m

