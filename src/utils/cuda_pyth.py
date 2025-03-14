import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import gc

def print_cuda_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                print(type(obj), obj.size())
        except Exception:
            pass
    return None

def empty_cuda():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except Exception:
            pass
    return None
