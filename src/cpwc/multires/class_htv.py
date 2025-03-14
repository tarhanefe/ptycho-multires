import torch
import torch.nn.functional as F
import numpy as np
from src.cpwc.tools.utils import *
from src.cpwc.multires.class_multires import *

class HTV(MultiRes):

    def __init__(reg):
    
        reg.f1 = 2 * torch.Tensor([[[[1., -1.,], [-1., 1.]]]]).double().to(reg.device)
        reg.sigma_L = 8 #may not be suitable for the transpose


    def L(reg, x):
        '''
        Input: x // shape: (1, 1, width,height)// type: torch.float64

        Output: L // shape: (1, 3, width+2, height+2) // type: torch.float64

        Calculates the convolution of the input x with the filters f1, f2, f3 that are 
        being used for the calculation of the hessian.
        '''
        Lx1 = cmpx_conv2d(cmpx_pad(x, (0, 1, 0, 1)), reg.f1)

        return Lx1

    def Lt(reg, y):
        '''
        Input: y // shape: (1, 3, width+2, height+2) // type: torch.float64

        Output: Lt // shape: (1, 1, width,height)// type: torch.float64
        
        Calculates the multiplication with the adjoint of the hessian matrix L.
        '''
        Lt1y = cmpx_conv_transpose2d(y[:, :, :, :], reg.f1)[:, :, :-1, :-1]

        return Lt1y

    def eval(reg, x,l1_type = 'l1_row'):
        '''
        Input: x // shape: (1, 1, width, height) // type: torch.float64

        Output: x // type: float
        After obtatining the hessian matrix L, this function calculates the l1 norm of the matrix L.
        '''
        #evaluates regularization without the lambda
        if l1_type == "l1_row":
            return reg.L(x).abs().sum().item() #!!
        if l1_type == "l1_all":
            return torch.view_as_real(reg.L(x)).abs().sum().item()

    def grad(reg, y, iter_in, lmbda, tau, toi=1e-4):
        '''
        Applies the the algorithm 3 which deals with the gradient of the HTV.
        '''
        v_k = torch.zeros((1, 1,  2 ** reg.loc["s"], 2 ** reg.loc["s"]), requires_grad=False, device=reg.device).double()
        u_k = v_k.clone().detach()

        n, t_k, ukp1 = 0, 1, None
        alpha = (reg.sigma_L ** -2)

        loss_ = lambda x: (((y - reg.Lt(x)) ** 2).sum() / 2).item()  #tried -- -+ ++ +- #best -- +-
        error = torch.inf
        while n < iter_in and abs(error) > toi:
            
            Lpc = -reg.L(y - reg.Lt(v_k))
            u_kp1 = complex_clamp(v_k - alpha * Lpc, -lmbda * tau, lmbda * tau)

            error = calc_error(u_k, u_kp1, norm1=loss_)
            u_k, v_k, t_k = fista_fast(t_k, u_kp1, u_k)
            n += 1
        print(n)
        return y - reg.Lt(u_k)


