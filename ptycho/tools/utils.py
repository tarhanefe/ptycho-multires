import torch
import numpy as np
import torch.nn.functional as F

def Iter(func, x, iter):
    '''
    This function will apply the 'func' to the input x 'iter' times. 
    ex: 
    Iter(f(.),x,3) = f(f(f(x)))
    Iter(f(.),x,0) = x
    '''
    if iter > 0:
        return Iter(func, func(x), iter-1)
    else:
        return x

def calc_error(c_k, c_kp1, norm1):
    '''
    This function calculates the error between the two input tensors c_k and c_kp1.
    The error is calculated as the relative difference between the norms of the two tensors.
    
    '''
    if c_k is None or c_kp1 is None:
        return torch.inf

    else:
        loss1, loss2 = norm1(c_k), norm1(c_kp1)
        if loss2 != 0:
            return abs(loss1 - loss2) / loss2
        else:
            return torch.inf

def pad_t( x):
    '''
    Input: y // shape: (1, 1, width,height)// type: torch.float64

    Output: shape: (1, 1, width-1, height-1) // type: torch.float64

    This function cuts out the last row and columns of the input 
    tensor x. 
    '''
    return x[:, :, :-1, :-1]

def pad(y):
    '''
    Input: y // shape: (1, 1, width,height)// type: torch.float64

    Output: shape: (1, 1, width+1, height+1) // type: torch.float64

    This function adds a row and column of zeros to the input tensor y.
    But only to the right and bottom of the tensor.
    '''
    return F.pad(y, (0, 1, 0, 1))

def get_device(array):
    ''''
    This function returns the device of the input tensor array. If the input 
    is on the CPU, the function returns "cpu". If the input is on the GPU, the
    function returns "gpu".
    '''

    if torch.get_device(array) == -1:
        return "cuda"
    elif torch.get_device(array) == 0:
        return "cuda"

def fista_fast(t_k, c_kp1, c_k):
    '''
    Used for calculating the necessary variables c[k+1], 
    d[k+1] and t[k+1] using t[k], c[k+1] and c[k]for the 
    Algorithm 2 in the paper. 
    '''
    t_kp1 = (1 + np.sqrt(4 * t_k ** 2 + 1)) / 2
    d_kp1 = c_kp1 + (t_k - 1) / t_kp1 * (c_kp1 - c_k)
    return c_kp1, d_kp1, t_kp1

def norm_fro(x):
    '''
    Input : x // shape: (1, 1, width,height) // type: torch.float64

    Output : float

    This function calculates the Frobenius norm of the input tensor x.
    And returns the maximum of the norm and 1e-10 to avoid numerical instability. 
    '''
    return max(torch.norm(x, p="fro").item(), 1e-10)

def soft(x, l):
    '''
    Input : x // shape: (1, 1, width,height) // type: torch.float64

    Output : x // shape: (1, 1, width,height) // type: torch.float64

    This function zeros out the elements in
    x that has absolute value less than l.
    Used for the clipping step in the FISTA algorithm.

    '''
    return F.threshold(x.abs(), l, 0) * torch.sgn(x)

def complexify(i):
    if i.dtype == torch.complex64 or i.dtype == torch.complex128:
        return i
    elif (len(i.shape) == 4):
        tens = A = torch.stack([i, torch.zeros_like(i)], dim=-1)
        return torch.view_as_complex(tens)
    else:
        return torch.view_as_complex(i)
    

def cmpx_conv2d(input, filter, padding=0, stride=1):
    input = complexify(input)
    re_input = input.real
    imag_input = input.imag
    re_output = F.conv2d(re_input, filter, padding=padding, stride=stride) 
    imag_output = F.conv2d(imag_input, filter, padding=padding, stride=stride)
    return torch.view_as_complex(torch.stack([re_output, imag_output], dim=-1))

def cmpx_conv_transpose2d(input, filter , padding=0, stride=1):
    input = complexify(input)    
    re_input = input.real
    imag_input = input.imag
    re_output = F.conv_transpose2d(re_input, filter, padding=padding, stride=stride)
    imag_output = F.conv_transpose2d(imag_input, filter, padding=padding, stride=stride)
    return torch.view_as_complex(torch.stack([re_output, imag_output], dim=-1))

def cmpx_pad(input, pad):
    input = complexify(input)
    re_input = input.real
    imag_input = input.imag
    re_output = F.pad(re_input, pad)
    imag_output = F.pad(imag_input, pad)
    return torch.view_as_complex(torch.stack([re_output, imag_output], dim=-1))

def cmpx_pad_t(input, pad):
    input = complexify(input)
    re_input = input.real
    imag_input = input.imag
    re_output = pad_t(re_input)
    imag_output = pad_t(imag_input)
    return torch.view_as_complex(torch.stack([re_output, imag_output], dim=-1))

def complex_clamp(input, min=None, max=None):
    # Separate the real and imaginary parts
    real_part = input.real
    imag_part = input.imag
    
    # Apply torch.clamp to both the real and imaginary parts
    real_clamped = torch.clamp(real_part, min=min, max=max)
    imag_clamped = torch.clamp(imag_part, min=min, max=max)
    
    # Recombine the clamped real and imaginary parts into a complex tensor
    return torch.complex(real_clamped, imag_clamped)


def scrapper(x):
    if type(x) != list:
        return x
    else: 
        return scrapper(x[0])
    
