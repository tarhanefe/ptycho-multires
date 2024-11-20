from abc import abstractmethod
import torch
import numpy as np
from ptycho_v2.tools_v2.linop import LinOpFFT, LinOpRoll, LinOpMul, LinOpFFT2,LinOpIFFT2,LinOpRoll2, LinOpCrop2, LinOpCat, BaseLinOp
from ptycho_v2.tools_v2.phaseretrieval import PhaseRetrievalBase
from ptycho_v2.tools_v2.u_ptychography import generate_shifts, get_probe_diameter, get_overlap_img,generate_shifts_v2
from ptycho_v2.tools_v2.u_czt import custom_fft2


class Ptychography(PhaseRetrievalBase):
    def __init__(self, in_shape=None, probe_radius=None,fov=None, n_img:int=25, device="cuda"):
        self.device = device
        self.in_shape = in_shape
        self.probe_radius = probe_radius
        self.probe = self.construct_probe(probe_radius=probe_radius)                              
        self.n_img = n_img
        self.fov = fov
        # self.shifts = generate_shifts(size=in_shape, n_img=n_img,
        #                                 probe_diameter=probe_diameter, fov=fov, n_dim=2)
    
        self.shifts = generate_shifts_v2(size=in_shape, probe_radius=self.probe_radius)
        self.n_img = len(self.shifts)
        self.renormalize_probe()
        self.linop = self.build_lin_op()

    def build_lin_op(self) -> BaseLinOp:
        op_fft2 = LinOpFFT2()
        op_probe = LinOpMul(self.probe)
        if self.in_shape == self.probe.shape:
            return LinOpCat([
                op_fft2 @ op_probe @ 
                LinOpRoll2(self.shifts[i_probe,0], self.shifts[i_probe,1])
                for i_probe in range(self.n_img)
            ])
        else:
            return LinOpCat([
                op_fft2 @ op_probe @
                LinOpCrop2(self.in_shape, self.probe.shape) @
                LinOpRoll2(self.shifts[i_probe,0], self.shifts[i_probe,1])
                for i_probe in range(self.n_img)
            ])
    
    def construct_probe(self,probe_radius=10):
        # shape = (self.in_shape[0] + 1, self.in_shape[1] + 1)
        # probe = torch.zeros(shape)
        # probe[self.in_shape[0]//2 + 1-probe_radius//2:self.in_shape[0]//2+probe_radius//2 + 1,
        #       self.in_shape[1]//2 + 1-probe_radius//2:self.in_shape[1]//2+probe_radius//2 + 1] = 1
        shape = (self.in_shape[0], self.in_shape[1])
        probe = torch.zeros(shape)
        probe[self.in_shape[0]//2 -probe_radius//2:self.in_shape[0]//2+probe_radius//2 ,
                self.in_shape[1]//2 -probe_radius//2:self.in_shape[1]//2+probe_radius//2 ] = 1
            
        return probe.to(self.device)
        
    def renormalize_probe(self):
        overlap_img = get_overlap_img(probe=self.probe, shifts=self.shifts, n_dim=2)
        mean_val = torch.sqrt(torch.mean(overlap_img))
        self.probe = self.probe / mean_val

    def init_multipliers(self):
        return None