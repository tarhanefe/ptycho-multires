from abc import abstractmethod
import torch
import numpy as np
from src.cpwc.tools.linop import (
    LinOpMul, LinOpFFT2, LinOpRoll2, LinOpCrop2,
    LinOpCat, BaseLinOp, LinOpFFTShift2D
)
from src.cpwc.tools.phaseretrieval import PhaseRetrievalBase
from src.cpwc.tools.u_ptychography import generate_shifts, get_overlap_img

class Ptychography(PhaseRetrievalBase):
    def __init__(self, max_scale=9, min_scale=4,
                 max_probe_size=128, max_shift=32, device="cuda"):
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.max_probe_size = max_probe_size
        self.max_shift = max_shift
        self.device = device
        self.in_shape = None
        self.shifts = None
        self.probe = None
        self.linop = None
        self.initialized = False

    def initialize(self, in_shape):
        self.in_shape = in_shape
        self.scale = int(np.log2(self.in_shape[0]))
        self.n_copies = 2 ** (self.max_scale - self.scale)
        self.probe_radius_temp = int(self.max_probe_size / self.n_copies)
        self.shift_amount_temp = int(self.max_shift / self.n_copies)
        self.probe = self.construct_probe(self.probe_radius_temp)

        self.shifts = generate_shifts(self.in_shape[-1], self.shift_amount_temp)
        self.n_img = len(self.shifts)
        self.renormalize_probe()
        self.init_multipliers()
        self.linop = self.build_lin_op()
        self.initialized = True

    def build_lin_op(self) -> BaseLinOp:
        op_fft2   = LinOpFFT2()
        op_probe  = LinOpMul(self.probe)
        parts = []
        for i in range(self.n_img):
            roll = LinOpRoll2(*self.shifts[i])
            if self.in_shape == self.probe.shape:
                parts.append(op_fft2 @ op_probe @ roll)
            else:
                parts.append(
                    op_fft2 @ op_probe @
                    LinOpCrop2(self.in_shape, self.probe.shape) @
                    roll
                )
        return LinOpCat(parts)

    def construct_probe(self, probe_radius=10):
        h, w = self.in_shape
        probe = torch.zeros((h, w), dtype=torch.complex64, device=self.device)
        sr, er = h//2 - probe_radius//2, h//2 + probe_radius//2
        sc, ec = w//2 - probe_radius//2, w//2 + probe_radius//2
        probe[sr:er, sc:ec] = 1
        return probe

    def renormalize_probe(self):
        overlap = get_overlap_img(probe=self.probe,
                                  shifts=self.shifts, n_dim=2)
        mean_val = torch.sqrt(torch.mean(overlap))
        self.probe = self.probe / mean_val

    def init_multipliers(self):
        vec   = torch.arange(-2**(self.max_scale-1),
                             2**(self.max_scale-1),
                             device=self.device)
        exp   = torch.exp(-1j * np.pi * (vec * 2**(-self.scale)))
        sinc  = torch.sinc(vec * 2**(-self.scale))
        se    = sinc * exp
        mat   = se.view(-1,1) @ se.view(1,-1)
        mult  = 2**(self.max_scale - self.scale)
        self.multipliers = mat * mult

    def apply(self, x):
        if (not self.initialized) or (x.shape[-2:] != self.in_shape):
            self.initialize(x.shape[-2:])
        out = super().apply_linop(x)
        if self.scale == self.max_scale:
            out = LinOpFFTShift2D().apply(out)
        out = self.copy(out, self.n_copies)
        return torch.abs(out * self.multipliers)**2

    def apply_linop(self, x):
        if (not self.initialized) or (x.shape[-2:] != self.in_shape):
            self.initialize(x.shape[-2:])
        out = super().apply_linop(x)
        if self.scale == self.max_scale:
            out = LinOpFFTShift2D().apply(out)
        out = self.copy(out, self.n_copies)
        return out * self.multipliers

    def apply_linopT(self, y):
        if not self.initialized:
            raise RuntimeError("Ptychography not initialized")
        y = y * self.multipliers.conj()
        y = self.copyT(y, self.in_shape[0])
        if self.scale == self.max_scale:
            y = LinOpFFTShift2D().applyT(y)
        return super().apply_linopT(y)

    def copy(self, base, n):
        return base.repeat(1, 1, n, n)

    def copyT(self, images: torch.Tensor, patch_size: int) -> torch.Tensor:
        _, b, h, _ = images.shape
        grid = h // patch_size
        patches = images.reshape(1, b, grid, patch_size, grid, patch_size)
        return patches.sum(dim=(2, 4))

    def restart(self):
        self.initialized = False

