from abc import abstractmethod
import torch
import numpy as np
from ptycho_v2.tools_v2.linop import LinOpFFT, LinOpRoll, LinOpMul, LinOpFFT2,LinOpIFFT2,LinOpRoll2, LinOpCrop2, LinOpCat, BaseLinOp
from ptycho_v2.tools_v2.phaseretrieval import PhaseRetrievalBase
from ptycho_v2.tools_v2.u_ptychography import generate_shifts, get_probe_diameter, get_overlap_img,generate_shifts_v2
from ptycho_v2.tools_v2.u_czt import custom_fft2


class Ptychography(PhaseRetrievalBase):
    def __init__(self, probe_radius=None, fov=None, n_img:int=25, device="cuda"):
        self.device = device
        self.in_shape = None  # To be determined adaptively
        self.probe_radius = probe_radius
        self.n_img = n_img
        self.fov = fov
        self.shifts = None
        self.probe = None
        self.linop = None
        self.initialized = False  # Tracks if parameters have been initialized

    
    def initialize(self, in_shape):
        """Initialize the parameters adaptively based on input size."""
        self.in_shape = in_shape
        self.probe = self.construct_probe(probe_radius=self.probe_radius)
        self.scale = int(np.log2(self.in_shape[0]))
        self.shifts = generate_shifts_v2(size=self.in_shape, probe_radius=self.probe_radius)
        self.n_img = len(self.shifts)
        self.renormalize_probe()
        self.linop = self.build_lin_op()
        self.init_multipliers()
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
        
    def renormalize_probe(self):
        overlap_img = get_overlap_img(probe=self.probe, shifts=self.shifts, n_dim=2)
        mean_val = torch.sqrt(torch.mean(overlap_img))
        self.probe = self.probe / mean_val

    def apply(self, x):
        if not self.initialized:
            self.initialize(x.shape[-2:])  # Initialize based on input size
        return torch.abs(self.linop.apply(x) * self.multipliers)**2

    def apply_linop(self, x):
        if not self.initialized:
            self.initialize(x.shape[-2:])  # Initialize based on input size
        return super().apply_linop(x) * self.multipliers

    def apply_linopT(self, x):
        if not self.initialized:
            self.initialize(x.shape[-2:])  # Initialize based on input size
        return super().apply_linopT(x) * self.multipliers.T.conj()
    
    def init_multipliers(self):
        multiplier = 2**(-2 * self.scale)
        vec = torch.arange(0, 2**self.scale) * (2**(-self.scale))
        sinc_exp = torch.sinc(vec) * torch.exp(-1j * np.pi * vec)
        result = sinc_exp.view(-1, 1) @ sinc_exp.view(1, -1)
        self.multipliers = result * multiplier
        self.multipliers = self.multipliers.to(self.device)

    def restart(self):
        self.initialized = False