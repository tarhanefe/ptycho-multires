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

def save_data(model,image_path,metrics,device = 'cuda',max_scale = 9,overlap = 75,spline_type = "cpwc",lambda_ = 0.1,noise_type = "possion",noise = 0.1,loop = "mrgd"):
    image,image_tensor_ = get_image(image_path,max_scale = max_scale,device = device)
    mean_img = np.mean(image)
    for i in metrics:
        if i == "loss":
            loss_data = extract_data(model.measures["loss"])
            np.save("data/{}_overlap{}_{}_lambda{}_noise_type{}_noise{}_loss.npy".format(spline_type,overlap,loop,lambda_,noise_type,noise), loss_data)
        if i == "csim":
            cos_sim = extract_data(model.measures["csim"])
            np.save("data/{}_overlap{}_{}_lambda{}_noise_type{}_noise{}_csim.npy".format(spline_type,overlap,loop,lambda_,noise_type,noise), cos_sim)
        if i == "image":
            phase = torch.angle(model.c_k[0,0,:,:].to('cpu'))
            phase = phase.numpy()
            phase = unwrap_2d(phase)
            phase += (mean_img-np.mean(phase)) 
            np.save("data/{}_overlap{}_{}_lambda{}_noise_type{}_noise{}_phase.npy".format(spline_type,overlap,loop,lambda_,noise_type,noise), phase)
        if i == "psnr":
            psnr = extract_data(model.measures["psnr"])
            np.save("data/{}_overlap{}_{}_lambda{}_noise_type{}_noise{}_psnr.npy".format(spline_type,overlap,loop,lambda_,noise_type,noise), psnr)
        
        if i == "frc":
            frc = get_ring_average(image_tensor_[0,0,:,:], model.c_k[0,0,:,:])
            np.save("data/{}_overlap{}_{}_lambda{}_noise_type{}_noise{}_frc.npy".format(spline_type,overlap,loop,lambda_,noise_type,noise), frc)        