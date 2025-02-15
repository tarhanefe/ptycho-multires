import pytest
import torch
import numpy as np
from src.cpwc.tools.ptychography import Ptychography
# (Import any other modules you need)

def obtain_mesurements():
    max_scale = 9
    min_scale = 6
    max_probe_size = 32 * 4
    max_shift = 32
    # Create an image tensor; no need to call torch.tensor() since image is already a tensor.
    image = torch.rand(1, 1, 2**max_scale, 2**max_scale)
    device = 'cuda'
    image_tensor = image.to(torch.complex64)
    x = torch.exp(1j * image_tensor)
    # Use the same dtype/device for the kernel matrix m
    m = torch.ones(2, 2, dtype=x.dtype, device=x.device)
    x8 = x[:, :, ::8, ::8]
    x4 = torch.kron(x8, m) / 2
    x2 = torch.kron(x4, m) / 2
    x = torch.kron(x2, m) / 2

    # Create four instances of the Ptychography operator
    linOperator1 = Ptychography(
        min_scale=min_scale,
        max_scale=max_scale,
        max_probe_size=max_probe_size,
        max_shift=max_shift,
        device=device
    )
    linOperator2 = Ptychography(
        min_scale=min_scale,
        max_scale=max_scale,
        max_probe_size=max_probe_size,
        max_shift=max_shift,
        device=device
    )
    linOperator4 = Ptychography(
        min_scale=min_scale,
        max_scale=max_scale,
        max_probe_size=max_probe_size,
        max_shift=max_shift,
        device=device
    )
    linOperator8 = Ptychography(
        min_scale=min_scale,
        max_scale=max_scale,
        max_probe_size=max_probe_size,
        max_shift=max_shift,
        device=device
    )

    # Apply the linear operators
    y = linOperator1.apply_linop(x.to(device))
    y2 = linOperator2.apply_linop(x2.to(device))
    y4 = linOperator4.apply_linop(x4.to(device))
    y8 = linOperator8.apply_linop(x8.to(device))
    return [y, y2, y4, y8]

# Define a fixture so that Pytest can inject the measurements into the test.
@pytest.fixture
def measurements():
    return obtain_mesurements()

def test_forward(measurements):
    """
    For every pair of measurements, verify that the amplitude on the 20th index
    (along axis=1) is nearly identical.
    """
    device = 'cuda'
    max_scale = 9
    min_scale = 6
    tol = 1e-10
    num_list = [int(i * 2**min_scale) for i in range(1, int(2**(max_scale-min_scale)))]
    matrix = torch.ones(2**max_scale, 2**max_scale).to(device).to(torch.complex64)
    matrix[num_list, :] = 0
    matrix[:, num_list] = 0
    for meas_a in measurements:
        for meas_b in measurements:
            amp_a = torch.abs(meas_a[:, 20, :, :]) * matrix
            amp_b = torch.abs(meas_b[:, 20, :, :]) * matrix
            # Instead of taking logarithms, we simply check that the difference is within tolerance.
            diff = (amp_a - amp_b).abs()
            mean_diff = diff.mean().item()

            assert mean_diff < tol, f"Mean difference {mean_diff} exceeds tolerance {tol}"