import torch
import numpy as np

from linop import LinOpRoll, LinOpRoll2, LinOpCrop, LinOpCrop2


def get_probe_diameter(probe, threshold=0.1, n_dim=1):
    assert n_dim in [1, 2], "n_dim must be 1 or 2"
    if n_dim == 1:
        return torch.count_nonzero(torch.abs(probe) > threshold)
    else:
        mid_idx = int(np.ceil(probe.shape[0]/2+0.5))
        return torch.count_nonzero(torch.abs(probe[mid_idx, :]) > threshold)

def generate_shifts(size, n_img, probe_diameter=10, fov=None, n_dim=1):
    assert n_dim in [1, 2], "n_dim must be 1 or 2"
    if fov is None:
        start_shift = -(size-probe_diameter)//2
        end_shift = (size-probe_diameter)//2
    else:
        start_shift = - fov // 2
        end_shift = fov // 2

    if n_dim == 1:
        return np.linspace(start_shift, end_shift, n_img).astype(int)
    else:
        assert int(np.sqrt(n_img))**2 == n_img, "n_img needs to be a perfect square"
        side_n_img = int(np.sqrt(n_img))
        shifts = np.linspace(start_shift, end_shift, side_n_img).astype(int)
        shifts_h, shifts_v = np.meshgrid(shifts, shifts, indexing='ij')
        return np.concatenate(
            [shifts_v.reshape(n_img, 1), shifts_h.reshape(n_img, 1)], axis=1)

def get_overlap_img(probe, shifts, in_shape=None, n_dim=1):
    assert n_dim in [1, 2], "n_dim must be 1 or 2"
    if in_shape is not None and in_shape != probe.shape:
        crop_op0 = LinOpCrop2 if n_dim == 2 else LinOpCrop
        op_crop = crop_op0(in_shape, probe.shape)
        probe = op_crop.applyT(probe)
    overlap_img = torch.zeros_like(probe, dtype=torch.float32)
    if n_dim == 1:
        for i_probe in range(len(shifts)):
            roll_linop  = LinOpRoll(-shifts[i_probe])
            overlap_img += torch.abs(roll_linop.apply(probe))**2
    else:
        for i_probe in range(len(shifts)):
            roll_linop  = LinOpRoll2(-shifts[i_probe,0], -shifts[i_probe,1])
            overlap_img += torch.abs(roll_linop.apply(probe))**2
    return overlap_img

def get_overlap_rate(probe, shifts, threshold=0.1, n_dim=1):
    assert n_dim in [1, 2], "n_dim must be 1 or 2"
    if n_dim == 1:
        probe_dia = torch.count_nonzero(torch.abs(probe) > threshold)
        step_size = np.abs(shifts[0]-shifts[1])
        return 1 - step_size / probe_dia
    else:  # to double-check
        mid_idx = int(np.ceil(probe.shape[0]/2+0.5))
        probe_dia = torch.count_nonzero(torch.abs(probe[mid_idx, :]) > threshold)
        
        probe_radius = probe_dia//2
        step_size = np.abs(shifts[0, 1] - shifts[1, 1])
        if step_size > 2*probe_radius:
            return 0
        else:
            circ_area = np.arccos(step_size/2/probe_radius) * probe_radius**2
            tri_area = step_size / 2 * np.sqrt(probe_radius**2 - (step_size/2)**2)
            overlap_rate = 2 * (circ_area - tri_area) / (np.pi * probe_radius**2)
            return overlap_rate
