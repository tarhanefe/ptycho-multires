import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import time

from src.cpwc_v.multires.class_multires import *
from src.cpwc_v.multires.class_interpolation import *
from src.cpwc_v.multires.class_loss import *
from src.cpwc_v.tools.utils import *


class MultiResSolver():

    def __init__(self, multires, loss, I_in=None, I_out=None, tol=None, cycle=None, tol_in=None,LR = None,l1_type = 'l1_row',d0 = False,gt = None):            
        
        self.l1_type = l1_type
        self.multires = multires
        self.d0 = d0
        self.loss = loss

        self.LR = LR
        self.lr_list = []
        self.cycle = {"cycle": cycle, "I_in": I_in, "I_out": I_out, "tol": tol, "tol_in": tol_in}

        #self.measures = {"loss": [], "mse": [], "reg": [], "rel_loss": [], "iters": [], "time": []}
        self.measures = {"loss": [],
                         "mse": [], 
                         "rel_loss": [], 
                         "iters": [], 
                         "time": [],
                         "csim": []}
        self.loc = {"grid": 0,
                    "d_k": None}

        self.sols = [[] for i in range(self.loss.F.S)]

        self.shift = [torch.zeros((1, 1, 2 ** (i + 1) , 2 ** (i + 1)),
                                     device=self.multires.device,
                                     dtype=torch.double) for i in range(self.loss.F.S)]


        self.infos = lambda:  'Iter ' + str(self.measures["iters"][-1][-1]) + \
                              ', [loss, mse, reg, rel_loss, LR] : [' \
                              + str(np.round(self.measures["loss"][-1][-1], 7)) + ", " \
                              + str(np.round(self.measures["mse"][-1][-1], 7))+ ", " \
                              + str(np.round(self.measures["rel_loss"][-1][-1], 7)) + ", " \
                              + str(self.LR) + "] "
                              
        self.gt = torch.tensor(gt).double().to(self.loss.F.device)


    def cosine_similarity(self, gt, recon):
        gt = self.gt 
        rec = recon.clone()
        m = torch.ones(2, 2).to(self.multires.device)
        max_scale = int(np.log2(gt.shape[-1]))
        current_scale = self.multires.loc["s"]
        for i in range(max_scale - current_scale):
            rec = torch.kron(rec, m)
        rec += torch.mean(gt) - torch.mean(rec)
        return torch.abs(torch.sum(gt*rec)) / (torch.norm(gt) * torch.norm(rec))

    def up_measures(self, x1=None, x2=None):

        #time does not need an init or a particular update because it is not a list of list

        if x1 is None:
            self.measures["iters"].append([])
            self.measures["loss"].append([])
            self.measures["mse"].append([])
            #self.measures["reg"].append([])
            self.measures["rel_loss"].append([])
            self.measures["csim"].append([])

        else:
            try:
                self.measures["iters"][-1].append(self.measures["iters"][-1][-1] + 1)
            except:
                self.measures["iters"][-1].append(0)
            #self.measures["loss"][-1].append(self.loss.calc_loss(x1, l1_type= self.l1_type))
            self.measures["loss"][-1].append(self.loss.calc_loss(x1))
            self.measures["mse"][-1].append(self.loss.calc_mse(x1))
           # self.measures["reg"][-1].append(self.loss.calc_reg(x1,l1_type= self.l1_type))
            self.measures["rel_loss"][-1].append(calc_error(x2, x1, norm1=self.loss.calc_loss))
            self.measures["csim"][-1].append(self.cosine_similarity(self.gt, x1).cpu().numpy())

    
    
    def print_time(self):
        tot = 0
        for t in self.measures["time"]:
            tot += t

        print("Times: ", self.measures["time"])
        print("Total time: ", tot)
    
    def calc_grad(self,x):
        Grad = self.loss.F.Ht((self.loss.F.H(x)/(torch.abs(self.loss.F.H(x))+1e-8))*(torch.sqrt(self.loss.F.H_power(x)) - torch.sqrt(self.loss.y)))
        #Grad = self.loss.F.Ht(self.loss.F.H(x)*(self.loss.F.H_power(x) - self.loss.y)) ! OLD METHOD 
        return Grad

    def solve_scale(self):
        alpha_u = 1.01
        alpha_d = 0.8
        g, d_k =  self.loc["grid"], self.loc["d_k"]
        #F, reg, loss = self.loss.F, self.loss.reg, self.loss
        F, loss = self.loss.F, self.loss
        t_k, c_kp1, self.c_k = 1, None, d_k
        self.up_measures()
        self.up_measures(self.c_k, c_kp1)
        self.measures["time"].append(-time.time())
        while self.measures["iters"][-1][-1] < self.cycle["I_out"][g]:
            self.lr_list.append(self.LR)
            c_kp1 = d_k - self.calc_grad(self.c_k) * self.LR

            #if (self.loss.calc_loss(self.c_k) < 0.9*self.loss.calc_loss(c_kp1)):
            #    self.LR = alpha_d*self.LR
            #else: 
            #    self.LR = alpha_u*self.LR
            #    self.up_measures(self.c_k, c_kp1)
            #    self.c_k, d_k, t_k = fista_fast(t_k, c_kp1, self.c_k)
            #    print(self.infos())

            self.up_measures(self.c_k, c_kp1)  # Update tracking measures
            self.c_k, d_k, t_k = fista_fast(t_k, c_kp1, self.c_k)  # Perform FISTA update
            print(self.infos())
        self.measures["time"][-1] += time.time()
        self.sols[self.multires.loc["s"] - 1] = self.c_k


    def solve_multigrid(self):
        self.final_sols = []
        #F, reg, multires = self.loss.F, self.loss.reg, self.multires
        F, multires = self.loss.F, self.multires
        for grid in range(len(self.cycle["cycle"])):
            multires.set_locals(scale=self.cycle["cycle"][grid], mod="update")
            s, size = multires.loc["s"], 2 ** multires.loc["s"]
            self.loc["grid"] = grid
            N = size **2 
            std = np.sqrt(2/N)
            if self.d0 is False:
                d0 = torch.randn((1, 1, size ,size)).double().to(F.device) * std
            else: 
                d0 = self.d0
            self.loc['d_k'] = d0
            if self.cycle["cycle"][grid] == 0 and type(self.sols[s - 1]) == torch.Tensor:
                self.loc["d_k"] = self.sols[s - 1]
            #goes on a finer scale
            elif self.cycle["cycle"][grid] == 1:
                self.loc["d_k"] = F.up(self.sols[s - 2])

            print('----------- s = ' + str(multires.loc["s"]) + ' -----------' )
            self.solve_scale()
            self.final_sols.append(self.sols[s - 1])

