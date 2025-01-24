from abc import abstractmethod
import torch
import numpy as np
from src.cpwl.tools.linop import LinOpFFT, LinOpRoll, LinOpMul, LinOpFFT2,LinOpIFFT2,LinOpRoll2, LinOpCrop2, LinOpCat, BaseLinOp
from src.cpwl.tools.phaseretrieval import PhaseRetrievalBase
from src.cpwl.tools.u_ptychography import generate_shifts, get_probe_diameter, get_overlap_img,generate_shifts_v2
from src.cpwl.tools.u_czt import custom_fft2


class Ptychography(PhaseRetrievalBase):
    def __init__(self, in_shape=100, probe=None, shifts=None, n_img:int=10, 
                 threshold=0.1, device="cuda"):
        self.in_shape = in_shape
        self.device = device
        if probe is not None:
            self.probe = probe
        else:
            self.probe = self.construct_probe()
        if shifts is not None:
            self.shifts = shifts
            self.n_img = len(shifts)
        else:
            self.n_img = n_img
            probe_diameter = get_probe_diameter(self.probe, threshold=threshold, n_dim=1)
            self.shifts = generate_shifts(size=in_shape, n_img=n_img, 
                                          probe_diameter=probe_diameter, n_dim=1)
        self.linop = self.build_lin_op()

    def build_lin_op(self) -> BaseLinOp:
        op_fft = LinOpFFT()
        op_probe = LinOpMul(self.probe)
        return LinOpCat([
            op_fft @ op_probe @ LinOpRoll(self.shifts[i_probe])
            for i_probe in range(self.n_img)
        ])
    
    def construct_probe(self):
        probe_dia = self.in_shape // 5
        probe = torch.zeros(self.in_shape)
        probe[self.in_shape//2-probe_dia//2:self.in_shape//2+probe_dia//2] = 1
        return probe.to(self.device)
    

class Ptychography2(PhaseRetrievalBase):
    def __init__(self, in_shape=None, probe=None, shifts=None, 
                 probe_type=None, probe_radius=None, defocus_factor=0.5,  # probe parameters
                 fov=None, threshold=0.1, n_img:int=25, device="cuda"):
        self.device = device
        if probe is not None:
            self.probe = probe
            self.in_shape = in_shape if in_shape is not None else probe.shape
        else:
            self.in_shape = in_shape
            self.probe_type = probe_type
            self.probe_radius = probe_radius
            self.defocus_factor = defocus_factor
            self.probe = self.construct_probe(type=probe_type, 
                                              probe_radius=probe_radius, 
                                              defocus_factor=defocus_factor)
        if shifts is not None:
            self.shifts = shifts
            self.n_img = len(shifts)
        else:
            self.n_img = n_img
            self.threshold = threshold
            self.fov = fov
            probe_diameter = 0  # get_probe_diameter(self.probe, threshold=threshold, n_dim=2)
            self.shifts = generate_shifts(size=in_shape, n_img=n_img,
                                          probe_diameter=probe_diameter, fov=fov, n_dim=2)
            self.shifts = generate_shifts_v2(size=in_shape,probe_radius=self.probe_radius)

        self.renormalize_probe()
        self.linop = self.build_lin_op() #!!!!!!!!!!!!!!

    def build_lin_op(self) -> BaseLinOp:
        op_fft2 = LinOpFFT2()
        op_ifft2 = LinOpIFFT2()
        op_probe = LinOpMul(self.probe)
        if self.in_shape == self.probe.shape:
            return LinOpCat([
                op_fft2 @ op_probe @ 
                LinOpRoll2(self.shifts[i_probe,0], self.shifts[i_probe,1]) @ op_ifft2
                for i_probe in range(self.n_img)
            ])
        else:
            return LinOpCat([
                op_fft2 @ op_probe @
                LinOpCrop2(self.in_shape, self.probe.shape) @
                LinOpRoll2(self.shifts[i_probe,0], self.shifts[i_probe,1]) @ op_ifft2
                for i_probe in range(self.n_img)
            ])
    
    def build_single_lin_op(self, i_probe=None) -> BaseLinOp:
        op_fft2 = LinOpFFT2()
        op_probe = LinOpMul(self.probe)
        if self.in_shape == self.probe.shape:
            if i_probe is None:  # return the operator without shifts
                return op_fft2 @ op_probe
            else:
                return op_fft2 @ op_probe @ \
                    LinOpRoll2(self.shifts[i_probe,0], self.shifts[i_probe,1])
        else:
            if i_probe is None:
                return op_fft2 @ op_probe @ \
                    LinOpCrop2(self.in_shape, self.probe.shape)
            else:
                return op_fft2 @ op_probe @ \
                    LinOpCrop2(self.in_shape, self.probe.shape) @ \
                    LinOpRoll2(self.shifts[i_probe,0], self.shifts[i_probe,1])
        
    def single_apply(self, x, i_probe=None):
        single_lin_op = self.build_single_lin_op(i_probe)
        return torch.abs(single_lin_op.apply(x))**2
    
    def get_fourier_probe(self, type, defocus_factor):
        # TODO: can be compressed
        if type == 'defocus pupil':
            x = torch.arange(self.in_shape[0], dtype=torch.float64)
            y = torch.arange(self.in_shape[1], dtype=torch.float64)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            
            # Construct pupil
            pupil = torch.zeros(self.in_shape)
            pupil[torch.sqrt((xx-self.in_shape[0]//2)**2 + (yy-self.in_shape[1]//2)**2) 
                < self.in_shape[0]//2] = 1
            
            # Add defocus
            defocus_filter = ((xx-self.in_shape[0]//2)**2 + (yy-self.in_shape[1]//2)**2) / \
                (self.in_shape[0]//2)**2
            pupil = pupil * torch.exp(1j * 2 * np.pi * defocus_filter * defocus_factor)
        elif type == 'random pupil':
            x = torch.arange(self.in_shape[0], dtype=torch.float64)
            y = torch.arange(self.in_shape[1], dtype=torch.float64)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            
            # Construct pupil
            mask = torch.zeros(self.in_shape)
            mask[torch.sqrt((xx-self.in_shape[0]//2)**2 + (yy-self.in_shape[1]//2)**2) 
                < self.in_shape[0]//2] = 1
            pupil = mask * torch.exp(1j * 2 * np.pi * torch.rand(self.in_shape))
        
        elif type == 'square pupil':
            x = torch.arange(self.in_shape[0], dtype=torch.float64)
            y = torch.arange(self.in_shape[1], dtype=torch.float64)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            
            # Construct pupil
            pupil = torch.zeros(self.in_shape)
            pupil[torch.abs(xx-self.in_shape[0]//2) < self.in_shape[0]//4] = 1
            pupil[torch.abs(yy-self.in_shape[1]//2) < self.in_shape[1]//4] = 1
        self.pupil = pupil  # TODO: clean this :)
        return pupil
        
    def construct_probe(self, type='disk', 
                        probe_radius=10, defocus_factor=0, speckle_size=2):
        if type == 'disk' or type is None:
            x = torch.arange(self.in_shape[0], dtype=torch.float64)
            y = torch.arange(self.in_shape[1], dtype=torch.float64)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            probe = torch.zeros(self.in_shape)
            probe[torch.sqrt((xx-self.in_shape[0]//2)**2 + (yy-self.in_shape[1]//2)**2) 
                < probe_radius] = 1
        elif type == 'defocus pupil':
            pupil = self.get_fourier_probe(type, defocus_factor)

            # Fourier transform
            probe = custom_fft2(pupil, k_start=-np.pi/probe_radius, k_end=np.pi/probe_radius, 
                                fftshift_input=True, include_end=True)
            # torch.fft.fftshift(
            #     torch.fft.fft2(torch.fft.ifftshift(pupil), norm='ortho'))
        elif type == 'random pupil':
            pupil = self.get_fourier_probe(type, defocus_factor)
            
            # Fourier transform
            probe = custom_fft2(pupil, k_start=-np.pi/speckle_size, k_end=np.pi/speckle_size, 
                                fftshift_input=True, include_end=True)
            probe *= torch.exp(- ((xx-self.in_shape[0]//2)**2 + (yy-self.in_shape[1]//2)**2)
                               / 2 / (probe_radius/2)**2
                               )
        if type == "square":
            probe = torch.zeros(self.in_shape)
            probe[self.in_shape[0]//2-probe_radius//2:self.in_shape[0]//2+probe_radius//2,
                  self.in_shape[1]//2-probe_radius//2:self.in_shape[1]//2+probe_radius//2] = 1
        return probe.to(self.device)
        
    def renormalize_probe(self):
        overlap_img = get_overlap_img(probe=self.probe, shifts=self.shifts, n_dim=2)
        mean_val = torch.sqrt(torch.mean(overlap_img))
        self.probe = self.probe / mean_val

class Ptychography2_v2(PhaseRetrievalBase):
    def __init__(self, in_shape=None, probe=None, shifts=None, 
                 probe_type=None, probe_radius=None, defocus_factor=0.5,  # probe parameters
                 fov=None, threshold=0.1, n_img:int=25, device="cuda"):
        self.device = device
        if probe is not None:
            self.probe = probe
            self.in_shape = in_shape if in_shape is not None else probe.shape
        else:
            self.in_shape = in_shape
            self.probe_type = probe_type
            self.probe_radius = probe_radius
            self.defocus_factor = defocus_factor
            self.probe = self.construct_probe(type=probe_type, 
                                              probe_radius=probe_radius, 
                                              defocus_factor=defocus_factor)
        if shifts is not None:
            self.shifts = shifts
            self.n_img = len(shifts)
        else:
            self.n_img = n_img
            self.threshold = threshold
            self.fov = fov
            probe_diameter = 0  # get_probe_diameter(self.probe, threshold=threshold, n_dim=2)
            self.shifts = generate_shifts(size=in_shape, n_img=n_img,
                                         probe_diameter=probe_diameter, fov=fov, n_dim=2)
        
            self.shifts = generate_shifts_v2(size=in_shape, probe_radius=self.probe_radius)
            self.n_img = len(self.shifts)
        self.renormalize_probe()
        self.linop = self.build_lin_op() #!!!!!!!!!!!!!!

    def build_lin_op(self) -> BaseLinOp:
        op_fft2 = LinOpFFT2()
        op_ifft2 = LinOpIFFT2()
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
    
    def build_single_lin_op(self, i_probe=None) -> BaseLinOp:
        op_fft2 = LinOpFFT2()
        op_probe = LinOpMul(self.probe)
        if self.in_shape == self.probe.shape:
            if i_probe is None:  # return the operator without shifts
                return op_fft2 @ op_probe
            else:
                return op_fft2 @ op_probe @ \
                    LinOpRoll2(self.shifts[i_probe,0], self.shifts[i_probe,1])
        else:
            if i_probe is None:
                return op_fft2 @ op_probe @ \
                    LinOpCrop2(self.in_shape, self.probe.shape)
            else:
                return op_fft2 @ op_probe @ \
                    LinOpCrop2(self.in_shape, self.probe.shape) @ \
                    LinOpRoll2(self.shifts[i_probe,0], self.shifts[i_probe,1])
        
    def single_apply(self, x, i_probe=None):
        single_lin_op = self.build_single_lin_op(i_probe)
        return torch.abs(single_lin_op.apply(x))**2
    
    def get_fourier_probe(self, type, defocus_factor):
        # TODO: can be compressed
        if type == 'defocus pupil':
            x = torch.arange(self.in_shape[0], dtype=torch.float64)
            y = torch.arange(self.in_shape[1], dtype=torch.float64)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            
            # Construct pupil
            pupil = torch.zeros(self.in_shape)
            pupil[torch.sqrt((xx-self.in_shape[0]//2)**2 + (yy-self.in_shape[1]//2)**2) 
                < self.in_shape[0]//2] = 1
            
            # Add defocus
            defocus_filter = ((xx-self.in_shape[0]//2)**2 + (yy-self.in_shape[1]//2)**2) / \
                (self.in_shape[0]//2)**2
            pupil = pupil * torch.exp(1j * 2 * np.pi * defocus_filter * defocus_factor)
        elif type == 'random pupil':
            x = torch.arange(self.in_shape[0], dtype=torch.float64)
            y = torch.arange(self.in_shape[1], dtype=torch.float64)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            
            # Construct pupil
            mask = torch.zeros(self.in_shape)
            mask[torch.sqrt((xx-self.in_shape[0]//2)**2 + (yy-self.in_shape[1]//2)**2) 
                < self.in_shape[0]//2] = 1
            pupil = mask * torch.exp(1j * 2 * np.pi * torch.rand(self.in_shape))
        
        elif type == 'square pupil':
            x = torch.arange(self.in_shape[0], dtype=torch.float64)
            y = torch.arange(self.in_shape[1], dtype=torch.float64)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            
            # Construct pupil
            pupil = torch.zeros(self.in_shape)
            pupil[torch.abs(xx-self.in_shape[0]//2) < self.in_shape[0]//4] = 1
            pupil[torch.abs(yy-self.in_shape[1]//2) < self.in_shape[1]//4] = 1

        self.pupil = pupil  # TODO: clean this :)
        return pupil
        
    def construct_probe(self, type='disk', 
                        probe_radius=10, defocus_factor=0, speckle_size=2):
        if type == 'disk' or type is None:
            x = torch.arange(self.in_shape[0], dtype=torch.float64)
            y = torch.arange(self.in_shape[1], dtype=torch.float64)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            probe = torch.zeros(self.in_shape)
            probe[torch.sqrt((xx-self.in_shape[0]//2)**2 + (yy-self.in_shape[1]//2)**2) 
                < probe_radius] = 1
        elif type == 'defocus pupil':
            pupil = self.get_fourier_probe(type, defocus_factor)

            # Fourier transform
            probe = custom_fft2(pupil, k_start=-np.pi/probe_radius, k_end=np.pi/probe_radius, 
                                fftshift_input=True, include_end=True)
            # torch.fft.fftshift(
            #     torch.fft.fft2(torch.fft.ifftshift(pupil), norm='ortho'))
        elif type == 'random pupil':
            pupil = self.get_fourier_probe(type, defocus_factor)
            
            # Fourier transform
            probe = custom_fft2(pupil, k_start=-np.pi/speckle_size, k_end=np.pi/speckle_size, 
                                fftshift_input=True, include_end=True)
            probe *= torch.exp(- ((xx-self.in_shape[0]//2)**2 + (yy-self.in_shape[1]//2)**2)
                               / 2 / (probe_radius/2)**2
                               )
        elif type == "square":
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