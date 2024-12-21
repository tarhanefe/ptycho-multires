from abc import abstractmethod
import torch
import numpy as np
from ptycho_v2.tools_v2.linop import LinOpMul, LinOpFFT2,LinOpRoll2, LinOpCrop2, LinOpCat, BaseLinOp
from ptycho_v2.tools_v2.phaseretrieval import PhaseRetrievalBase
from ptycho_v2.tools_v2.u_ptychography import generate_shifts, get_overlap_img


class Ptychography(PhaseRetrievalBase):
    def __init__(self,max_scale = 9,max_probe_size = 128 ,max_shift = 32,device="cuda"):
        self.max_scale = max_scale
        self.max_shift = max_shift
        self.max_probe_size = max_probe_size
        self.device = device
        self.in_shape = None  # To be determined adaptively
        self.shifts = None
        self.probe = None
        self.linop = None
        self.initialized = False  # Tracks if parameters have been initialized

    
    def initialize(self, in_shape):
        """Initialize the parameters adaptively based on input size."""
        self.in_shape = in_shape 
        self.scale = int(np.log2(self.in_shape[0]))
        self.n_copies = 2**(self.max_scale - self.scale)
        self.probe_radius_temp = int(self.max_probe_size / self.n_copies)
        self.shift_amount_temp = int(self.max_shift / self.n_copies)
        self.probe = self.construct_probe(probe_radius=self.probe_radius_temp)

        self.shifts = generate_shifts(self.in_shape[-1], self.shift_amount_temp)
        self.n_img = len(self.shifts)
        self.renormalize_probe()
        self.init_multipliers()
        self.linop = self.build_lin_op()
        self.initialized = True

    def build_lin_op(self) -> BaseLinOp:
        op_fft2 = LinOpFFT2()
        op_probe = LinOpMul(self.probe)
        if self.in_shape == self.probe.shape:
            return LinOpCat([
                op_fft2 @ op_probe @ 
                LinOpRoll2(self.shifts[i_probe, 0], self.shifts[i_probe, 1])
                for i_probe in range(self.n_img)
            ])
        else:
            return LinOpCat([
                op_fft2 @ op_probe @
                LinOpCrop2(self.in_shape, self.probe.shape) @
                LinOpRoll2(self.shifts[i_probe, 0], self.shifts[i_probe, 1])
                for i_probe in range(self.n_img)
            ])
    
    def construct_probe(self, probe_radius=10):
        shape = (self.in_shape[0], self.in_shape[1])
        probe = torch.zeros(shape)
        probe[self.in_shape[0] // 2 - probe_radius // 2 : self.in_shape[0] // 2 + probe_radius // 2,
              self.in_shape[1] // 2 - probe_radius // 2 : self.in_shape[1] // 2 + probe_radius // 2] = 1
        return probe.to(self.device)
    
    def copy(self,base_matrix, n):
        shared_matrix = base_matrix.repeat(1, 1, n, n)
        return shared_matrix

    def renormalize_probe(self):
        self.overlap_img = get_overlap_img(probe=self.probe, shifts=self.shifts, n_dim=2)
        mean_val = torch.sqrt(torch.mean(self.overlap_img))
        self.probe = self.probe / mean_val

    def apply(self, x):
        if self.initialized and x.shape[-2:] != self.in_shape:
            self.initialize(x.shape[-2:])
        elif not self.initialized:
            self.initialize(x.shape[-2:])  # Initialize based on input size
        else:
            pass
        x = self.linop.apply(x)
        x = self.copy(x,self.n_copies)
        x = torch.abs(x * self.multipliers)**2
        return x

    def apply_linop(self, x):
        if self.initialized and x.shape[-2:] != self.in_shape:
            self.initialize(x.shape[-2:])
        elif not self.initialized:
            self.initialize(x.shape[-2:])  # Initialize based on input size
        else:
            pass
        x = super().apply_linop(x)
        x = self.copy(x,self.n_copies)
        x = x * self.multipliers
        return x

    def apply_linopT(self, y):
        y = y * self.multipliers.conj()
        y = self.copyT(y, self.in_shape[0])
        y = super().apply_linopT(y)
        return y
    
    def init_multipliers(self):
        multiplier = 2**(-2 * self.scale)
        vec = torch.arange(0, 2**self.max_scale) * (2**(-self.scale))
        sinc_exp = torch.sinc(vec) * torch.exp(-1j * np.pi * vec)
        result = sinc_exp.view(-1, 1) @ sinc_exp.view(1, -1)
        self.multipliers = result * multiplier 
        self.multipliers = self.multipliers.to(self.device)

    def restart(self):
        self.initialized = False

    def copyT(self,images: torch.Tensor, patch_size: int) -> torch.Tensor:
        _, batch_size, height, _ = images.shape
        n = height
        grid_size = n // patch_size  
        patches = images.reshape(1, batch_size, grid_size, patch_size, grid_size, patch_size)
        patches = patches.sum(dim=(2, 4))

        return patches