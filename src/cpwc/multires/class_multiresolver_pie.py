import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import time
from collections import deque
import cv2
from src.cpwc.multires.class_htv import *
from src.cpwc.multires.class_multires import *
from src.cpwc.multires.class_interpolation import *
from src.cpwc.multires.class_loss import *
from src.cpwc.tools.utils import *
from src.utils.manage_data import save_data,unwrap_2d,extract_data


class MultiResSolver():

    def __init__(self, multires, loss, I_in=None, I_out=None, tol=None, cycle=None, tol_in=None,LR = None,l1_type = 'l1_row',gt = None,early_stopping = 9,scaler = 4,tol_vals = [10,0.5,1000]):            
        self.minscale = loss.F.linOperator.min_scale
        self.maxscale = loss.F.linOperator.max_scale
        self.l1_type = l1_type
        self.multires = multires
        self.gt = gt
        self.history = tol_vals[0]
        self.percent_tol = tol_vals[1]
        self.max_iter = tol_vals[2]
        self.iterats_ = 0
        self.scaler = scaler
        self.iter_times = []
        self.loss = loss
        self.early_stopping = early_stopping
        self.LR = LR
        self.cycle = {"cycle": cycle, "I_in": I_in, "I_out": I_out, "tol": tol, "tol_in": tol_in}

        self.measures = {"loss": [], 
                         "mse": [], 
                         "reg": [], 
                         "rel_loss": [], 
                         "iters": [], 
                         "time": [], 
                         "csim": [],
                         "psnr":[],
                         "gt_mse":[],
                         "imp": []}

        self.loc = {"grid": 0,
                    "d_k": None}

        self.sols = [[] for i in range(self.loss.F.S)]

        self.shift = [torch.zeros((1, 1, 2 ** (i + 1) , 2 ** (i + 1)),
                                     device=self.multires.device,
                                     dtype=torch.double) for i in range(self.loss.F.S)]


        self.infos = lambda:  'Iter ' + str(self.measures["iters"][-1][-1]) + \
                              ', [loss,mse,reg,csim] : [' \
                              + str(np.round(self.measures["loss"][-1][-1], 7)) + ", " \
                              + str(np.round(self.measures["mse"][-1][-1], 7))+ ", " \
                              + str(np.round(self.measures["reg"][-1][-1], 7)) + ", " \
                              + str(np.round(self.measures["csim"][-1][-1], 7)) + ", " \
                              + "] "
#                              + str(np.round(self.measures["gt_mse"][-1][-1], 7)) + ", " \
#                              + str(np.round(self.measures["psnr"][-1][-1], 7)) + ", " \
#                              + str(np.round(self.measures["rel_loss"][-1][-1], 7)) + ", " \
#                              + str(self.LR) + "] "
                              
    #    self.gt = torch.tensor(gt).double().to(self.loss.F.device)


    def csim(self, gt, recon):
        gt = self.gt  
        rec = recon.clone()
        m = torch.ones(2, 2, dtype=rec.dtype, device=self.multires.device)
        max_scale = self.loss.F.linOperator.max_scale
        current_scale = self.multires.loc["s"]
        for _ in range(max_scale - current_scale):
            rec = torch.kron(rec, m)
        dot_product = torch.sum(gt * rec.conj())  
        norm_gt = torch.norm(gt)
        norm_rec = torch.norm(rec)
        return torch.abs(dot_product) / (norm_gt * norm_rec)

    
    def up_measures(self, x1=None, x2=None):
        if x1 is None:
            self.measures["iters"].append([])
            self.measures["loss"].append([])
            self.measures["mse"].append([])
            self.measures["reg"].append([])
            #self.measures["rel_loss"].append([])
            self.measures["csim"].append([])
            #self.measures["psnr"].append([])
            #self.measures["gt_mse"].append([])
            self.measures["imp"].append([])

        else:
            try:
                self.measures["iters"][-1].append(self.measures["iters"][-1][-1] + 1)
            except:
                self.measures["iters"][-1].append(0)
            self.measures["loss"][-1].append(self.loss.calc_loss(x1, l1_type= self.l1_type))
            #self.measures["loss"][-1].append(self.loss.calc_loss(x1))
            self.measures["mse"][-1].append(self.loss.calc_mse(x1))
            self.measures["reg"][-1].append(self.loss.calc_reg(x1,l1_type= self.l1_type))
            #self.measures["rel_loss"][-1].append(calc_error(x2, x1, norm1=self.loss.calc_loss))
            self.measures["csim"][-1].append(self.csim(self.gt, x1).cpu().numpy())
            #self.measures["psnr"][-1].append(self.psnr(self.gt, x1).cpu().numpy())
            #self.measures["gt_mse"][-1].append(self.gt_mse(self.gt, x1).cpu().numpy())
            if x2 is None:
                x2 = x1
            self.measures["imp"][-1].append(self.loss.calc_improvement(x1, x2))
    
    
    def print_time(self):
        tot = 0
        for t in self.measures["time"]:
            tot += t

        print("Times: ", self.measures["time"])
        print("Total time: ", tot)
    
    def estimate_LR(self):
        L = calcLiepschitz(self.loss.F.linOperator, self.c_k, self.loss.y, num_iterations=50, tol=1e-6, device=self.loss.F.device)
        return self.scaler / L
    

    def solve_scale(self):
        g, d_k =  self.loc["grid"], self.loc["d_k"]
        F, reg, loss = self.loss.F, self.loss.reg, self.loss
        t_k, c_kp1, self.c_k = 1, None, d_k
        self.up_measures()
        self.up_measures(self.c_k, c_kp1)
        self.measures["time"].append(-time.time())
        self.LR_ = self.LR[self.loc["grid"]]
        number_of_measurements = self.loss.y.shape[1]
        self._imp_deque = deque(maxlen=self.history)
        while (self.measures["iters"][-1][-1] < self.cycle["I_out"][g]) and (self.iterats_ < self.max_iter):
            c_kp1 = self.c_k.clone()
            self.psi = self.loss.F.H(c_kp1)
            self.I = self.loss.F.H_power(c_kp1)
            self.Psi = self.psi / (torch.abs(self.psi) + 1e-16)
            update = self.loss.F.Ht(self.Psi*torch.sqrt(self.loss.y) - self.psi)
            self.LR_ = self.estimate_LR()
            c_kp1 = c_kp1 + self.LR_ * update
            start_iter = time.time()
            self.up_measures(self.c_k, c_kp1)  # Update tracking measures
            if (self.measures["iters"][-1][-1] != self.maxscale) and (self.measures["iters"][-1][-1] > 0):
                self._imp_deque.append(self.measures['imp'][-1][-1])
                if (len(self._imp_deque) == self.history) :
                    self.avg_imp = sum(self._imp_deque) / self.history
                    print("Average improvement: ", self.avg_imp)
                    if self.avg_imp < self.percent_tol:
                        break
            self.c_k = c_kp1
            iter_time = time.time() - start_iter
            self.iter_times.append(iter_time)
            print('LR = ' + str(self.LR_))
            print(self.infos())
            #self.imsave()
            self.iterats_ += 1
            
            
        self.measures["time"][-1] += time.time()
        
        self.sols[self.multires.loc["s"] - 1] = self.c_k


    def solve_multigrid(self):
        self.final_sols = []
        F, multires = self.loss.F, self.multires
        for grid in range(len(self.cycle["cycle"])):
            multires.set_locals(scale=self.cycle["cycle"][grid], mod="update")
            self.s, size = multires.loc["s"], 2 ** multires.loc["s"]
            self.loc["grid"] = grid
            N = size **2 
            std = np.sqrt(2/N)
            self.loc["d_k"] = torch.exp(1j*torch.ones((1, 1, size ,size), dtype=torch.double, device=F.device) * 0)
            if self.cycle["cycle"][grid] == 0 and type(self.sols[self.s - 1]) == torch.Tensor:
                self.loc["d_k"] = self.sols[self.s - 1]
            elif self.cycle["cycle"][grid] == 1:
                self.loc["d_k"] = F.up(self.sols[self.s - 2])
            print('----------- s = ' + str(multires.loc["s"]) + ' -----------' )
            self.solve_scale()
            self.final_sols.append(self.sols[self.s - 1])
            if (self.s == self.early_stopping) and (self.cycle["cycle"][grid] == 1):
                break


    def imsave(self):
        phase = torch.angle(self.c_k[0,0,:,:].to('cpu'))
        phase = phase.numpy()
        phase = unwrap_2d(phase)
        phase = cv2.resize(phase, (2**self.early_stopping, 2**self.early_stopping), interpolation=cv2.INTER_CUBIC)
        plt.imsave(f"experiments/results/gif/img_{self.iterats_}.png", phase)