import torch
import numpy as np
from cpwl.tools.phaseretrieval import FourierFilterPR2, PhaseRetrievalBase
from cpwl.tools.u_electron_microscopy import get_ctf, get_envelope
from cpwl.tools.linop import LinOpFFT2


class zernike(FourierFilterPR2):
    def __init__(self, zernike_mask_radius=5, fov_radius=None, param=None, device="cuda"):
        fourier_filter = torch.ones(param.shape, param.shape, dtype=torch.complex64).to(device)
        center = param.shape // 2
        x = np.linspace(0, param.shape-1, param.shape)
        xx, yy = np.meshgrid(x, x, indexing="ij")
        r2 = torch.Tensor(((xx-center)**2 + (yy-center)**2))
        mask = torch.fft.ifftshift(r2 <= zernike_mask_radius**2)
        fourier_filter[mask] = 1j
        envelope = get_envelope(param=param, spatial_incoherence=False, temporal_incoherence=True).to(device)
        fourier_filter *= envelope
        self.fourier_filter = fourier_filter

        if fov_radius is not None:
            center = param.shape // 2
            x = np.linspace(0, param.shape-1, param.shape)
            xx, yy = np.meshgrid(x, x, indexing="ij")
            r2 = torch.Tensor(((xx-center)**2 + (yy-center)**2))
            self.object_mask = (r2 <= (fov_radius / param.pix_size)**2).to(device)
        else:
            self.object_mask = None

        super().__init__(fourier_filter=self.fourier_filter, object_mask=self.object_mask)
        
        self.device = device

class ctf_forward(FourierFilterPR2):
    def __init__(self, param=None, perfect=False, device="cuda"):
        ctf_filter = get_ctf(param=param, device=device, perfect=perfect)
        self.fourier_filter = ctf_filter.to(device)
        super().__init__(fourier_filter=self.fourier_filter)

class fourier_forward(PhaseRetrievalBase):
    def __init__(self, param=None, device="cuda"):
        self.linop = LinOpFFT2()
        envelope = get_envelope(param=param, spatial_incoherence=False, temporal_incoherence=True).to(device)
        self.linop = envelope * self.linop
        self.device = device
