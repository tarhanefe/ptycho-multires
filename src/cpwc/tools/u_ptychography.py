import torch
import numpy as np

from src.cpwc.tools.linop import LinOpRoll, LinOpRoll2, LinOpCrop, LinOpCrop2


def get_probe_diameter(probe, threshold=0.1, n_dim=1):
    assert n_dim in [1, 2], "n_dim must be 1 or 2"
    if n_dim == 1:
        return torch.count_nonzero(torch.abs(probe) > threshold)
    else:
        mid_idx = int(np.ceil(probe.shape[0]/2+0.5))
        return torch.count_nonzero(torch.abs(probe[mid_idx, :]) > threshold)

def generate_shifts(in_size, shift_amount):
    """
    Generates shifts for scanning based on the input size and shift amount.

    Parameters:
    - in_size (int): The size of the input image or region (assumes square region for simplicity).
    - shift_amount (int): Distance between consecutive shifts.

    Returns:
    - shifts (numpy array): Array of 2D shifts (vertical and horizontal).
    """
    # Calculate the range of shifts based on input size
    half_size = in_size // 2  # Half of the input size to define the range
    n_shifts = (2 * half_size // shift_amount) + 1  # Number of shifts along one axis

    # Generate shift values using linspace
    shifts = np.linspace(-half_size, half_size, n_shifts).astype(int)

    # Create a 2D grid of shifts
    shifts_h, shifts_v = np.meshgrid(shifts, shifts, indexing='ij')

    # Combine horizontal and vertical shifts into a single array
    return np.concatenate([shifts_v.reshape(-1, 1), shifts_h.reshape(-1, 1)], axis=1)

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
##############################################

def crop_nonzero_square(tensor):
    # Find the indices of non-zero elements
    nonzero_indices = torch.nonzero(tensor, as_tuple=False)
    
    # Find the min and max indices in both dimensions to determine the bounding box
    min_row, min_col = torch.min(nonzero_indices, dim=0).values
    max_row, max_col = torch.max(nonzero_indices, dim=0).values
    
    # Determine the side length of the square region
    side_length = max(max_row - min_row, max_col - min_col).item() + 1  # +1 to include the bounds
    
    # Crop the square region from the original tensor
    cropped_tensor = tensor[min_row:min_row + side_length, min_col:min_col + side_length]
    
    return cropped_tensor


def map_values_by_magnitude(array):
    # Flatten the array and find unique values
    unique_values = np.unique(array)
    
    # Sort unique values and assign labels based on their sorted order
    sorted_indices = np.argsort(unique_values)
    labels = {value: i + 1 for i, value in enumerate(unique_values[sorted_indices])}
    
    # Map each element in the array to its label
    mapped_array = np.vectorize(labels.get)(array)
    
    return mapped_array

def get_overlap_area(probe, shifts):
    overlap_img = torch.zeros_like(probe, dtype=torch.float32)
    dist = shifts[1,0]-shifts[0,0]
    selected_shifts = np.array([[0,dist],[0,0]])
    for i_probe in range(len(selected_shifts)):  
        roll_linop  = LinOpRoll2(-selected_shifts[i_probe,0], -selected_shifts[i_probe,1])
        overlap_img += torch.abs(roll_linop.apply(probe))**2
    return map_values_by_magnitude(overlap_img)
