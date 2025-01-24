import numpy as np
import torch
from src.cpwl.tools.structured_transforms import hartley_transform
from tqdm import tqdm

def compute_fi(phase_img, forward_operator, sequential=False, eps=1e-5):
    """
    Compute the Fisher Information matrix for a given phase image and forward model of the imaging system.
    :param phase_img: the phase image as a torch tensor
    :param forward_function: the forward model of the imaging system
    :return: the Fisher Information matrix
    """
    in_size = phase_img.shape[2] * phase_img.shape[3]
    if not sequential:
        f = lambda x: forward_operator.apply(torch.exp(1j * x.type(torch.complex64)))

        # Compute the measured intensity
        measured_intensity = f(phase_img)
        out_size = measured_intensity.shape[1] * measured_intensity.shape[2] * measured_intensity.shape[3]

        # Vectorized computation of Jacobian
        jacobian = torch.func.jacrev(f)(phase_img).reshape(out_size, in_size)

        # Fisher Information matrix
        renorm_factor = 1/(measured_intensity+eps).reshape(out_size)
        return (jacobian.T * renorm_factor) @ jacobian
    else:
        # Previous version
        n_img = forward_operator.n_img
        FIm = torch.zeros(in_size, in_size).to(phase_img.device)
        for i_probe in tqdm(range(n_img)):
            f = lambda x: forward_operator.single_apply(torch.exp(1j * x.type(torch.complex64)), i_probe)
            measured_intensity = f(phase_img)
            out_size = measured_intensity.shape[1] * measured_intensity.shape[2] * measured_intensity.shape[3]
            jacobian = torch.func.jacrev(f)(phase_img).reshape(out_size, in_size)
            renorm_factor = 1/(measured_intensity+eps).reshape(out_size).float()
            FIm += (jacobian.T * renorm_factor) @ jacobian
        return FIm
        # Fast approximate version
        # Compute first FIm without any shift
        print("\t Computing FIm without shifts")
        f = lambda x: forward_operator.single_apply(torch.exp(1j * x.type(torch.complex64)), i_probe=None)
        measured_intensity = f(phase_img)
        out_size = measured_intensity.shape[1] * measured_intensity.shape[2] * measured_intensity.shape[3]
        jacobian = torch.func.jacrev(f)(phase_img).reshape(out_size, in_size)
        renorm_factor = 1/(measured_intensity+eps).reshape(out_size).float()
        single_FIm = (jacobian.T * renorm_factor) @ jacobian
        single_FIm = single_FIm.reshape(phase_img.shape[2], phase_img.shape[3], phase_img.shape[2], phase_img.shape[3])

        # Sum the FIm for all the different shifts
        print("\t Computing FIm with shifts")
        n_img = forward_operator.n_img
        FIm = torch.zeros(in_size, in_size, dtype=torch.float32)
        for i_probe in tqdm(range(n_img)):
            shift_vec = forward_operator.shifts[i_probe, :]
            shifted_single_FIm = torch.roll(single_FIm, (-shift_vec[0], -shift_vec[1]), 
                                            dims=(2, 3))
            if -shift_vec[0] < 0:
                single_FIm[..., -shift_vec[0]:, :] = 0
            elif -shift_vec[0] > 0:
                single_FIm[..., 0:-shift_vec[0] , :] = 0
            if -shift_vec[1] < 0:
                single_FIm[..., :, -shift_vec[1]:] = 0
            elif -shift_vec[1] > 0:
                single_FIm[..., :, 0:-shift_vec[1]] = 0
            shifted_single_FIm = torch.roll(shifted_single_FIm, (-shift_vec[0], -shift_vec[1]), 
                                            dims=(0, 1))
            if -shift_vec[0] < 0:
                single_FIm[-shift_vec[0]:, :, ...] = 0
            elif -shift_vec[0] > 0:
                single_FIm[0:-shift_vec[0] , :, ...] = 0
            if -shift_vec[1] < 0:
                single_FIm[:, -shift_vec[1]:, ...] = 0
            elif -shift_vec[1] > 0:
                single_FIm[:, 0:-shift_vec[1], ...] = 0
            FIm += shifted_single_FIm.reshape(in_size, in_size)
        return FIm


def hartley_transform_fim(original_fim, size=None):
    """
    Compute the Hartley transform of the Fisher Information matrix.
    :param original_fim: the Fisher Information matrix
    :return: the Hartley transform of the Fisher Information matrix
    """
    if size is None:
        size0 = int(np.sqrt(original_fim.shape[0]))
        size1 = size0
    else:
        size0, size1 = size

    fi_matrix = original_fim.reshape(size0, size1, size0, size1)

    left_transform = hartley_transform(fi_matrix, dim=(0, 1), norm="ortho")
    right_transform = hartley_transform(left_transform, dim=(2, 3), norm="ortho")

    return right_transform.reshape(size0*size1, size0*size1)
