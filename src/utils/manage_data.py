import numpy as np
import torch
import matplotlib.pyplot as plt
from src.utils.get_image import get_image
from src.utils.metrics import get_ring_average

def unwrap_2d(phase):
    unwrapped_phase = np.unwrap(phase, axis=0)
    unwrapped_phase = np.unwrap(unwrapped_phase, axis=1)
    return unwrapped_phase

def extract_data(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):  
            result.extend(extract_data(item))
        else:  
            result.append(item)
    return result

def save_data(path,model,image_path,metrics,device = 'cuda',max_scale = 9,overlap = 75,spline_type = "cpwc",lr = 1e-2,lambda_ = 0.1,noise_type = "possion",noise = 0.1,loop = "mrgd"):
    image,image_tensor_ = get_image(image_path,max_scale = max_scale,device = device)
    mean_img = np.mean(image)
    if lr != None:
        file_name = "{}/{}_overlap{}_{}_lambda{}_noise_type{}_noise{}_lr{}".format(path,spline_type,overlap,loop,lambda_,noise_type,noise,lr)
    else: 
        file_name = "{}/{}_overlap{}_{}_lambda{}_noise_type{}_noise{}".format(path,spline_type,overlap,loop,lambda_,noise_type,noise)
    for i in metrics:
        if i == "loss":
            loss_data = extract_data(model.measures["loss"])
            np.save(file_name + "_loss.npy", loss_data)
        if i == "csim":
            cos_sim = extract_data(model.measures["csim"])
            np.save(file_name + "_csim.npy", cos_sim)
        if i == "image":
            phase = torch.angle(model.c_k[0,0,:,:].to('cpu'))
            phase = phase.numpy()
            phase = unwrap_2d(phase)
            phase += (mean_img-np.mean(phase)) 
            np.save(file_name + "_image.npy", phase)
        if i == "psnr":
            psnr = extract_data(model.measures["psnr"])
            np.save(file_name + "_psnr.npy", psnr)
        if i == "frc":
            frc = get_ring_average(image_tensor_[0,0,:,:], model.c_k[0,0,:,:],delta_radius=5)
            np.save(file_name + "_frc.npy", frc)
    return file_name

def calc_frc_x(delta_radius):
    fourier_k_range = np.fft.fftfreq(512) / 0.25
    fourier_k_range = fourier_k_range[0:256] * 10 
    fourier_k_range = fourier_k_range[1::delta_radius]
    return fourier_k_range