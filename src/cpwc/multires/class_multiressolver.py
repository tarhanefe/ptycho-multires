import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import time

from src.cpwc.multires.class_htv import *
from src.cpwc.multires.class_multires import *
from src.cpwc.multires.class_interpolation import *
from src.cpwc.multires.class_loss import *
from src.cpwc.tools.utils import *


class MultiResSolver():

    def __init__(self, multires, loss, I_in=None, I_out=None, tol=None, cycle=None, tol_in=None,LR = None,l1_type = 'l1_row',gt = None,early_stopping = 9):            
        self.minscale = loss.F.linOperator.min_scale
        self.maxscale = loss.F.linOperator.max_scale
        self.l1_type = l1_type
        self.multires = multires
        self.gt = gt
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
                         "gt_mse":[]}

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

#    def psnr(self, gt, recon):
#        gt = self.gt
#        rec = recon.clone()
#        m = torch.ones(2, 2, dtype=rec.dtype, device=self.multires.device)
#        max_scale = int(np.log2(gt.shape[-1]))
#        current_scale = self.multires.loc["s"]
#        
#        for _ in range(max_scale - current_scale):
#            rec = torch.kron(rec, m)
#        mse = torch.mean(torch.abs(gt - rec) ** 2)
#        if mse == 0:
#            return float('inf')
#        return - 10 * torch.log10(mse)
    
#    def gt_mse(self, gt, recon):
#        gt = self.gt
#        rec = recon.clone()
#        m = torch.ones(2, 2, dtype=rec.dtype, device=self.multires.device)
#        max_scale = int(np.log2(gt.shape[-1]))
#        current_scale = self.multires.loc["s"]
#        for _ in range(max_scale - current_scale):
#            rec = torch.kron(rec, m)
#        return torch.mean(torch.abs(gt - rec) ** 2)
    
    
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
    
    
    def print_time(self):
        tot = 0
        for t in self.measures["time"]:
            tot += t

        print("Times: ", self.measures["time"])
        print("Total time: ", tot)
    
    def calc_grad(self, x,loss_type = "rooted",conj = False):
        self.Ax = self.loss.F.H(x)
        self.Ax_pwr = self.loss.F.H_power(x)
        if loss_type == "rooted":
            self.Grad = self.loss.F.Ht(((self.Ax/(torch.abs(self.Ax)+1e-16))*(torch.sqrt(self.Ax_pwr) - torch.sqrt(self.loss.y))))
        elif loss_type == "amplitude":
            self.Grad = 2*self.loss.F.Ht(self.Ax*(self.Ax_pwr - self.loss.y))
        if conj:
            self.Grad = torch.conj(self.Grad)
        return self.Grad


#    def calc_grad(self,x):
#        self.Ax = self.loss.F.H(x)
#        self.Ax_pwr = self.loss.F.H_power(x)
#        Grad = self.loss.F.Ht((self.Ax/(torch.abs(self.Ax)+1e-8))*(torch.sqrt(self.Ax_pwr) - torch.sqrt(self.loss.y)))
#        return Grad

    def estimate_LR(self):
        L = calcLiepschitz(self.loss.F.linOperator, self.c_k, self.loss.y, num_iterations=50, tol=1e-6, device=self.loss.F.device)
        return 2.0 / L


    def solve_scale(self):

        g, d_k =  self.loc["grid"], self.loc["d_k"]
        F, reg, loss = self.loss.F, self.loss.reg, self.loss
        #F, loss = self.loss.F, self.loss
        t_k, c_kp1, self.c_k = 1, None, d_k
        self.up_measures()
        self.up_measures(self.c_k, c_kp1)
        self.measures["time"].append(-time.time())
        self.LR_ = self.LR[self.loc["grid"]]
        while self.measures["iters"][-1][-1] < self.cycle["I_out"][g]:
            start_iter = time.time()
            grad = self.calc_grad(self.c_k)
            self.LR_ = self.estimate_LR()
            inner_prox = d_k - grad * (self.LR_)
            c_kp1 = reg.grad(y=inner_prox,
                             iter_in=self.cycle["I_in"][g],
                             lmbda=loss.lmbda,
                             tau = self.LR_,
                             toi=self.cycle["tol_in"][g])            #if (self.loss.calc_loss(self.c_k) < 0.9*self.loss.calc_loss(c_kp1)):
            self.up_measures(self.c_k, c_kp1)  # Update tracking measures
            self.c_k, d_k, t_k = fista_fast(t_k, c_kp1, self.c_k)  # Perform FISTA update
            iter_time = time.time() - start_iter
            self.iter_times.append(iter_time)
            print('LR = ' + str(self.LR_))
            print(self.infos())
            
        self.measures["time"][-1] += time.time()
        

        self.sols[self.multires.loc["s"] - 1] = self.c_k



    def solve_multigrid(self):
        self.final_sols = []
        #F, reg, multires = self.loss.F, self.loss.reg, self.multires
        F, multires = self.loss.F, self.multires
        for grid in range(len(self.cycle["cycle"])):
            multires.set_locals(scale=self.cycle["cycle"][grid], mod="update")
            self.s, size = multires.loc["s"], 2 ** multires.loc["s"]
            self.loc["grid"] = grid
            N = size **2 
            std = np.sqrt(2/N)
            self.loc['d_k'] = torch.randn((1, 1, size ,size)).double().to(F.device) * std 
            if self.cycle["cycle"][grid] == 0 and type(self.sols[self.s - 1]) == torch.Tensor:
                self.loc["d_k"] = self.sols[self.s - 1]
            #goes on a finer scale
            elif self.cycle["cycle"][grid] == 1:
                self.loc["d_k"] = F.up(self.sols[self.s - 2])
            print('----------- s = ' + str(multires.loc["s"]) + ' -----------' )
            self.solve_scale()
            self.final_sols.append(self.sols[self.s - 1])
            if (self.s == self.early_stopping) and (self.cycle["cycle"][grid] == 1):
                break

