import torch
import numpy as np
from cpwc.tools.linop_base import BaseLinOp

## 1D classes
class LinOpMatrix(BaseLinOp):
    def __init__(self, matrix):
        self.forward_matrix = matrix
        self.in_shape = (matrix.shape[1],)
        self.out_shape = (matrix.shape[0],)
    
    def apply(self, x):
        return torch.einsum("ij, ...j -> ...i", self.forward_matrix, x)
        
    def applyT(self, x):
        return torch.einsum("ij, ...j -> ...i", self.forward_matrix.T, x)

class LinOpFFT(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return torch.fft.fft(x, norm="ortho")  # dim=-1 by default

    def applyT(self, x):
        return torch.fft.ifft(x, norm="ortho")

class LinOpRoll(BaseLinOp):
    def __init__(self, shifts, pad_zeros=False):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.shifts = int(np.round(shifts))
        self.pad_zeros = pad_zeros

    def apply(self, x):
        x = torch.roll(x, shifts=self.shifts, dims=-1)

        if self.pad_zeros:
            if self.shifts < 0:
                x[..., self.shifts:] = 0
            elif self.shifts > 0:
                x[..., 0:self.shifts] = 0
        return x

    def applyT(self, x):
        x = torch.roll(x, shifts=-self.shifts, dims=-1)

        if self.pad_zeros:
            if self.shifts > 0:
                x[..., -self.shifts:] = 0
            elif self.shifts < 0:
                x[..., 0:-self.shifts] = 0
        return x

class LinOpCrop(BaseLinOp):
    def __init__(self, in_shape, out_shape, fourier_origin=False):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.fourier_origin = fourier_origin

    def apply(self, x):
        if not self.fourier_origin:
            size = x.shape[-1]
            start = int(size//2 - self.out_shape//2)
            return x[..., start:start+self.out_shape]
        else:
            size = x.shape[-1]
            start = int(np.ceil(self.out_shape / 2))
            end = size - int(np.floor(self.out_shape / 2))
            return torch.cat((x[..., :start], x[..., end:]), dim=-1)

    def applyT(self, x):
        pad_size = self.in_shape - self.out_shape

        if not self.fourier_origin:
            if self.in_shape%2 == 1:
                x = torch.nn.functional.pad(
                    x, (int(np.floor(pad_size/2)), int(np.ceil(pad_size/2))),
                    mode='constant')
            else:
                x = torch.nn.functional.pad(
                    x, (int(np.ceil(pad_size/2)), int(np.floor(pad_size/2))),
                    mode='constant')
        else:
            mid_idx = int(np.ceil(self.out_shape / 2))
            x = torch.cat((x[..., :mid_idx], torch.zeros(x.shape[0], x.shape[1], pad_size), x[..., mid_idx:]), dim=-1)
        return x
    
# class LinOp_RealPartExpand(BaseLinOp):
#     def __init__(self, LinOp:BaseLinOp):
#         self.LinOp = LinOp
#         self.in_shape = (2*LinOp.in_shape[0],)
#         self.out_shape = LinOp.out_shape
    
#     def apply(self, x):
#         return torch.real(self.LinOp.apply(x[0:self.LinOp.in_shape[0]])) - \
#             torch.imag(self.LinOp.apply(x[-self.LinOp.in_shape[0]:]))

#     def applyT(self, x):
#         return torch.cat((torch.real(self.LinOp.applyT(x)), torch.imag(self.LinOp.applyT(x))), dim=0)

## 2D classes
class LinOpFFT2(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return torch.fft.fft2(x, norm="ortho")

    def applyT(self, x):
        return torch.fft.ifft2(x, norm="ortho")
    
class LinOpIFFT2(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return torch.fft.ifft2(x, norm="ortho")

    def applyT(self, x):
        return torch.fft.fft2(x, norm="ortho")


class LinOpRoll2(BaseLinOp):
    def __init__(self, shifts_dim0, shifts_dim1, pad_zeros=True):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        self.shifts_dim0 = int(shifts_dim0)
        self.shifts_dim1 = int(shifts_dim1)
        self.pad_zeros = pad_zeros

    def apply(self, x):
        x = torch.roll(x, (self.shifts_dim0, self.shifts_dim1), dims=(-2, -1))

        if self.pad_zeros:
            if self.shifts_dim0 < 0:
                x[..., self.shifts_dim0:, :] = 0
            elif self.shifts_dim0 > 0:
                x[..., 0:self.shifts_dim0 , :] = 0
            if self.shifts_dim1 < 0:
                x[..., :, self.shifts_dim1:] = 0
            elif self.shifts_dim1 > 0:
                x[..., :, 0:self.shifts_dim1] = 0
        return x
        
    def applyT(self, x):
        x = torch.roll(x, shifts=(-self.shifts_dim0, -self.shifts_dim1), dims=(-2, -1))

        if self.pad_zeros:
            if self.shifts_dim0 > 0:
                x[..., -self.shifts_dim0:, :] = 0
            elif self.shifts_dim0 < 0:
                x[..., 0:-self.shifts_dim0 , :] = 0
            if self.shifts_dim1 > 0:
                x[..., :, -self.shifts_dim1:] = 0
            elif self.shifts_dim1 < 0:
                x[..., :, 0:-self.shifts_dim1] = 0
        return x
    
class LinOpCrop2(BaseLinOp):
    def __init__(self, in_shape, out_shape, fourier_origin=False):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.fourier_origin = fourier_origin

    def apply(self, x):
        v_size, h_size = x.shape[-2:]
        if not self.fourier_origin:
            v_start = int(v_size//2 - self.out_shape[0]//2)
            h_start = int(h_size//2 - self.out_shape[1]//2)
            return x[..., v_start:v_start+self.out_shape[0], h_start:h_start+self.out_shape[1]]
        else:
            v_start = int(np.ceil(self.out_shape[0] / 2))
            v_end = v_size - int(np.floor(self.out_shape[0] / 2))
            h_start = int(np.ceil(self.out_shape[1] / 2))
            h_end = h_size - int(np.floor(self.out_shape[1] / 2))
            return torch.cat(
                (torch.cat((x[..., :v_start, :h_start], x[..., :v_start, h_end:]), dim=-1),
                torch.cat((x[..., v_end:, :h_start], x[..., v_end:, h_end:]), dim=-1)), 
                dim=-2)
    
    def applyT(self, x):
        v_pad_size = self.in_shape[0] - self.out_shape[0]
        h_pad_size = self.in_shape[1] - self.out_shape[1]

        if not self.fourier_origin:
            if self.in_shape[0]%2 == 1:
                x = torch.nn.functional.pad(
                    x, (0, 0, int(np.floor(v_pad_size/2)), int(np.ceil(v_pad_size/2))),
                    mode='constant')
            else:
                x = torch.nn.functional.pad(
                    x, (0, 0, int(np.ceil(v_pad_size/2)), int(np.floor(v_pad_size/2))),
                    mode='constant')
            if self.in_shape[1]%2 == 1:
                x = torch.nn.functional.pad(
                    x, (int(np.floor(h_pad_size/2)), int(np.ceil(h_pad_size/2)), 0, 0),
                    mode='constant')
            else:
                x = torch.nn.functional.pad(
                    x, (int(np.ceil(h_pad_size/2)), int(np.floor(h_pad_size/2)), 0, 0),
                    mode='constant')
        else:
            v_mid_idx = int(np.ceil(self.out_shape[0] / 2))
            h_mid_idx = int(np.ceil(self.out_shape[1] / 2))
            x = torch.cat(
                (torch.cat((x[..., :v_mid_idx, :h_mid_idx], torch.zeros(x.shape[0], x.shape[1], v_mid_idx, h_pad_size), x[..., :v_mid_idx, h_mid_idx:]), dim=-1),
                torch.zeros(x.shape[0], x.shape[1], v_pad_size, self.in_shape[1]),
                torch.cat((x[..., v_mid_idx:, :h_mid_idx], torch.zeros(x.shape[0], x.shape[1], self.out_shape[0]-v_mid_idx, h_pad_size), x[..., v_mid_idx:, h_mid_idx:]), dim=-1)),
                dim=-2)
        return x
    
## Dimensionless 
class LinOpCat(BaseLinOp):
    def __init__(self, LinOpList):
        self.LinOpList = LinOpList
        self.in_shape = max([linop.in_shape for linop in LinOpList])
        self.out_shape = tuple(np.sum(
            np.array([(linop.out_shape if linop.out_shape>(0,) else self.in_shape)
            for linop in LinOpList]), axis=0))

    def apply(self, x):
        return torch.cat(tuple(linop.apply(x) for linop in self.LinOpList), dim=1)

    def applyT(self, x):
        return torch.sum(torch.stack([linop.applyT(x[:, idx, :, :]) for idx, linop in enumerate(self.LinOpList)], dim=1), dim=1).unsqueeze(1)

class LinOpIdentity(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        
    def apply(self, x):
        return x
    
    def applyT(self, x):
        return x
    
class LinOpFlip(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return torch.flip(x, dims=(-1,))

    def applyT(self, x):
        return torch.flip(x, dims=(-1,))
    
class LinOpFlip2(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return torch.flip(x, dims=(-2, -1,))

    def applyT(self, x):
        return torch.flip(x, dims=(-2, -1,))

class LinOpFFTShift(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return torch.fft.fftshift(x)

    def applyT(self, x):
        return torch.fft.ifftshift(x)

class LinOpMul(BaseLinOp):
    def __init__(self, coefficients):
        self.coefficients = coefficients
        self.in_shape = coefficients.shape
        self.out_shape = coefficients.shape

    def apply(self, x):
        return self.coefficients * x

    def applyT(self, x):
        return self.coefficients.conj() * x

class LinOpReal(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)
        
    def apply(self, x):
        return torch.real(x)

    def applyT(self, x):
        return x

class LinOpImag(BaseLinOp):
    def __init__(self):
        self.in_shape = (-1,)
        self.out_shape = (-1,)

    def apply(self, x):
        return torch.imag(x)

    def applyT(self, x):
        return 1j * x
