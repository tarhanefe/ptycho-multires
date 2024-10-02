import torch
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

def initialize_physical_params(shape=101, pix_size=1, device="cpu", perfect=False):
    param = lambda : None

    # Camera parameters
    param.pix_size = pix_size  # in Angstroms
    param.shape = shape
    param.device = device

    # Fourier coefficients
    param.fourier_k_range1D = np.fft.fftfreq(param.shape) / param.pix_size
    param.fourier_kx, param.fourier_ky = np.meshgrid(param.fourier_k_range1D, 
                                                     param.fourier_k_range1D, indexing="ij")
    param.k_square = param.fourier_kx**2 + param.fourier_ky**2

    # Microscope parameters
    param.E0 = 300e3  # 300 keV electron microscope
    param.m = 9.109383e-31
    param.e = 1.602177e-19
    param.c = 299792458
    param.h = 6.62607e-34
    param.lambd = param.h / np.sqrt(2*param.m*param.e*param.E0) \
        / np.sqrt(1 + param.e*param.E0/2/param.m/param.c**2) * 1e10  # in Angstrom
    param.sigma = (2*np.pi/param.lambd/param.E0) \
        *(param.m*param.c**2+param.e*param.E0)/(2*param.m*param.c**2+param.e*param.E0)
    
    param.defocus = -3000  # in Angstrom (1000 = 0.1 um)
    param.Cs = 2.3e4 # in Angstrom (2.3 um)
    param.Cc = 2.8e7 # in Angstrom (2.8 mm)
    # n_scherzer = 1
    # param.defocus = - np.sqrt((2*n_scherzer-0.5)*param.Cs*param.lambd)
    
    if not perfect:
        param.beam_convergence = 1/5.31e1  # 0.1e-3  # in rad
        param.objective_instability = 5e-7  # current instability dI/I of the objective lens
        param.voltage_instability = 5e-7  # current instability dE/E of the voltage source
        param.energy_spread = 0.3  # energy spread of the electron beam in eV
    else:
        param.beam_convergence = 0
        param.objective_instability = 0
        param.voltage_instability = 0
        param.energy_spread = 0

    return param

def get_proj(param=None, sigma=1):
    """
    Load the projection data from the .mat file and return it as a torch tensor.
    :param sigma: sigma value for the gaussian filter, in Angstrom
    :param pix_size: pixel size in Angstrom
    """
    if param is None:
        param = initialize_physical_params()
    shape = param.shape
    pix_size = param.pix_size
    data = loadmat("potential.mat")
    phase = data["output"]
    orig_pix_size = 0.25

    result = np.sum(phase, axis=2)

    # Blur
    result = gaussian_filter(result, sigma=sigma / orig_pix_size)

    # Downsample
    result = result[::int(pix_size / orig_pix_size), ::int(pix_size / orig_pix_size)]
    
    # Crop empty edges
    result = crop(result, shape=shape)

    return torch.tensor(result, dtype=torch.float32).to(param.device)

def crop(input_img, shape):
    current_size = input_img.shape[0]
    if current_size > shape:
        middle = int(current_size / 2)
        start = int(middle - shape / 2)
        stop = start + shape
        input_img = input_img[start:stop, start:stop]
    else:
        # pad
        if (shape - current_size) % 2 == 1:
            pad_size = int((shape - current_size) / 2)
            input_img = np.pad(input_img, (pad_size, pad_size+1), mode="constant")
        else:
            pad_size = int((shape - current_size) / 2)
            input_img = np.pad(input_img, (pad_size, pad_size), mode="constant")
        # print("Warning: input image is smaller than the desired size.")
    return input_img

def get_ctf(param=None, device="cpu", perfect=False):
    if param is None:
        param = initialize_physical_params(device=device, perfect=perfect)
    else:
        device = param.device
    defocus_term = np.pi * param.lambd * param.k_square * param.defocus
    spherical_term = 0.5 * np.pi * param.lambd**3 * param.k_square**2 * param.Cs
    ctf_phase_mask = defocus_term + spherical_term
    ctf_phase_mask = torch.tensor(ctf_phase_mask, dtype=torch.complex64).to(device)

    ctf_envelope = get_envelope(param=param)
    ctf_filter = ctf_envelope * torch.exp(1j * ctf_phase_mask)
    return ctf_filter
    # return ctf_envelope, ctf_phase_mask

def get_envelope(param, spatial_incoherence=True, temporal_incoherence=True):
    k_square = torch.Tensor(param.k_square)
    ctf_envelope = torch.ones_like(k_square, dtype=torch.complex64)
    if spatial_incoherence:
        ctf_envelope *= torch.exp(- (np.pi * param.beam_convergence)**2 * (
            param.Cs * param.lambd**3 * k_square**1.5 -
            param.defocus * param.lambd * k_square**0.5)**2)
    if temporal_incoherence:
        Delta = param.Cc * np.sqrt(
            (2 * param.objective_instability)**2 + 
            (param.voltage_instability)**2 + 
            (param.energy_spread / param.E0)**2
        )
        ctf_envelope *= torch.exp(- (np.pi * param.lambd * Delta)**2 * k_square**2 / 2)
    # ctf_envelope = np.exp(- param.k_square**2 / (0.3337 * 0.5**4))  # From Colin's codes

    return ctf_envelope.to(param.device)

def get_ring_average(data, param, delta_radius=1, fftshift=True, return_x_axis=False):
    if fftshift:
        data = torch.fft.fftshift(data)
    size = data.shape[0]
    center = size//2
    ring_average = torch.empty(center).to(param.device)
    x = np.linspace(0, size, size, dtype=np.uint)
    xx, yy = np.meshgrid(x, x, indexing="ij") 
    r2 = ((xx-center)**2 + (yy-center)**2)
    for radius in range(center):
        mask = (r2 >= radius**2) * (r2 < (radius+delta_radius)**2)
        ring_average[radius] = data[mask].sum() / mask.sum()
    if return_x_axis:
        x_axis_ticks = np.sqrt(param.k_square[0, :center]) * 10  # in nm^-1
        return ring_average, x_axis_ticks
    else:
        return ring_average
    