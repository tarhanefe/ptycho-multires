import numpy as np
import torch 

def get_ring_average(ground_truth, estimate, delta_radius=1):
    ground_truth = torch.fft.fftshift(torch.fft.fft2(ground_truth))
    estimate = torch.fft.fftshift(torch.fft.fft2(estimate))
    size = ground_truth.shape[0]
    center = size//2
    
    x = np.linspace(0, size, size, dtype=np.uint)
    xx, yy = np.meshgrid(x, x, indexing="ij") 
    r2 = ((xx-center)**2 + (yy-center)**2)

    ring_average = torch.empty(center)
    for radius in range(center):
        mask = (r2 >= radius**2) * (r2 < (radius+delta_radius)**2)
        vec1 = ground_truth[mask].flatten()
        vec2 = estimate[mask].flatten()
        ring_average[radius] = torch.abs(torch.dot(vec1, vec2.conj())) / (torch.norm(vec1) * torch.norm(vec2))
    return np.array(ring_average)


def F2fluxconverter(F):
    pixel_area = 0.25**2 
    sum_of_overlaps = 1.0
    flux = F / pixel_area * sum_of_overlaps
    return flux


