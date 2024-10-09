import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import time

from ptycho.multires.class_htv import *
from ptycho.multires.class_multires import *
from ptycho.multires.class_interpolation import *
from ptycho.multires.class_loss import *
from ptycho.tools.utils import *


# 0: stay on the same level 
# 1: refine 
# -1: coarsen 

#Create the composite upscale. 
#Create the shift in the regularization. 
#Unify the stopping conditions. 


class MultiResSolver():

    def __init__(self, multires, loss, I_in=None, I_out=None, tol=None, cycle=None, tol_in=None,LR = None,l1_type = 'l1_row'):
        self.l1_type = l1_type
        self.multires = multires

        self.loss = loss

        self.LR = LR

        self.cycle = {"cycle": cycle, "I_in": I_in, "I_out": I_out, "tol": tol, "tol_in": tol_in}

        self.measures = {"loss": [], "mse": [], "reg": [], "rel_loss": [], "iters": [], "time": []}

        self.loc = {"grid": 0,
                    "d_k": None}

        self.sols = [[] for i in range(self.loss.F.S)]

        self.shift = [torch.zeros((1, 1, 2 ** (i + 1) - 1, 2 ** (i + 1) - 1),
                                     device=self.multires.device,
                                     dtype=torch.double) for i in range(self.loss.F.S)]


        self.infos = lambda:  'Iter ' + str(self.measures["iters"][-1][-1]) + \
                              ', [loss, mse, reg, rel_loss, LR] : [' \
                              + str(np.round(self.measures["loss"][-1][-1], 7)) + ", " \
                              + str(np.round(self.measures["mse"][-1][-1], 7))+ ", " \
                              + str(np.round(self.measures["reg"][-1][-1], 7)) + ", " \
                              + str(np.round(self.measures["rel_loss"][-1][-1], 7)) + ", " \
                              + str(self.LR) + "] "


    def up_measures(self, x1=None, x2=None):

        #time does not need an init or a particular update because it is not a list of list

        if x1 is None:
            self.measures["iters"].append([])
            self.measures["loss"].append([])
            self.measures["mse"].append([])
            self.measures["reg"].append([])
            self.measures["rel_loss"].append([])

        else:
            try:
                self.measures["iters"][-1].append(self.measures["iters"][-1][-1] + 1)
            except:
                self.measures["iters"][-1].append(0)
            self.measures["loss"][-1].append(self.loss.calc_loss(x1, l1_type= self.l1_type))
            self.measures["mse"][-1].append(self.loss.calc_mse(x1))
            self.measures["reg"][-1].append(self.loss.calc_reg(x1,l1_type= self.l1_type))
            self.measures["rel_loss"][-1].append(calc_error(x2, x1, norm1=self.loss.calc_loss))

    def print_time(self):
        tot = 0
        for t in self.measures["time"]:
            tot += t

        print("Times: ", self.measures["time"])
        print("Total time: ", tot)
    
    def calc_grad(self,x):
        Ax = self.loss.F.H(x)
        residual = (self.loss.y - self.loss.F.H_power(x))
        Ax_residual = Ax*residual
        return self.loss.F.Ht(Ax_residual)

    def solve_scale(self):
        alpha_d = 0.8
        alpha_u = 1

        g, d_k =  self.loc["grid"], self.loc["d_k"]
        F, reg, loss = self.loss.F, self.loss.reg, self.loss

        t_k, c_kp1, c_k = 1, None, d_k
        self.up_measures()
        self.up_measures(c_k, c_kp1)
        self.measures["time"].append(-time.time())
        while self.measures["iters"][-1][-1] < self.cycle["I_out"][g] and self.measures["rel_loss"][-1][-1] > self.cycle["tol"][g]:
            inner_prox = d_k + self.calc_grad(d_k) * (self.LR*self.multires.loc["sigma_U"] ** -2)
            c_kp1 = reg.grad(y=inner_prox,
                             iter_in=self.cycle["I_in"][g],
                             lmbda=loss.lmbda,
                             tau = self.LR * self.multires.loc["sigma_U"] ** -2,
                             toi=self.cycle["tol_in"][g])
            
            if (self.loss.calc_loss(c_k, l1_type= self.l1_type) < 0.9*self.loss.calc_loss(c_kp1, l1_type= self.l1_type)):
                self.LR = alpha_d*self.LR
            else: 
                self.LR = alpha_u*self.LR
                self.up_measures(c_k, c_kp1)
                c_k, d_k, t_k = fista_fast(t_k, c_kp1, c_k)
                print(self.infos())


        self.measures["time"][-1] += time.time()
        self.sols[self.multires.loc["s"] - 1] = c_k

    def solve_scale_v2(self):

        alpha_d = 0.7
        alpha_u = 1.01

        g, d_k =  self.loc["grid"], self.loc["d_k"]
        F, reg, loss = self.loss.F, self.loss.reg, self.loss

        t_k, c_kp1, c_k = 1, None, d_k
        self.up_measures()
        self.up_measures(c_k, c_kp1)
        self.measures["time"].append(-time.time())
        while self.measures["iters"][-1][-1] < self.cycle["I_out"][g] and self.measures["rel_loss"][-1][-1] > self.cycle["tol"][g]:
            inner_prox = d_k + F.Ht(loss.y - F.H(d_k)) * (self.LR*self.multires.loc["sigma_U"] ** -2)
            c_kp1 = reg.grad(y=inner_prox,
                             iter_in=self.cycle["I_in"][g],
                             lmbda=loss.lmbda,
                             tau = self.LR * self.multires.loc["sigma_U"] ** -2,
                             toi=self.cycle["tol_in"][g])
            
            if (self.loss.calc_loss(c_k, l1_type= self.l1_type) < 0.9*self.loss.calc_loss(c_kp1, l1_type= self.l1_type)):
                self.LR = alpha_d*self.LR
            else: 
                self.LR = alpha_u*self.LR
                self.up_measures(c_k, c_kp1)
                c_k, d_k, t_k = fista_fast(t_k, c_kp1, c_k)
                print(self.infos())


        self.measures["time"][-1] += time.time()
        self.sols[self.multires.loc["s"] - 1] = c_k

    def solve_scale_v3(self):
        g, d_k =  self.loc["grid"], self.loc["d_k"]
        F, reg, loss = self.loss.F, self.loss.reg, self.loss

        t_k, c_kp1, c_k = 1, None, d_k

        self.up_measures()
        self.up_measures(c_k, c_kp1)
        self.measures["time"].append(-time.time())

        while self.measures["iters"][-1][-1] < self.cycle["I_out"][g] and self.measures["rel_loss"][-1][-1] > self.cycle["tol"][g]:

            inner_prox = d_k + F.Ht(loss.y - F.H(d_k)) * (self.LR*self.multires.loc["sigma_U"] ** -2)
            c_kp1 = reg.grad(y=inner_prox,
                             iter_in=self.cycle["I_in"][g],
                             lmbda=loss.lmbda,
                             tau=self.multires.loc["sigma_U"] ** -2,
                             toi=self.cycle["tol_in"][g])

            self.up_measures(c_k, c_kp1)
            print(self.infos())

            c_k, d_k, t_k = fista_fast(t_k, c_kp1, c_k)

        self.measures["time"][-1] += time.time()
        self.sols[self.multires.loc["s"] - 1] = c_k


    def solve_multigrid(self):
        #implements exact residual-based multigrid

        F, reg, multires = self.loss.F, self.loss.reg, self.multires

        for grid in range(len(self.cycle["cycle"])):

            #s stands for (local) scale
            multires.set_locals(scale=self.cycle["cycle"][grid], mod="update")
            s, size = multires.loc["s"], 2 ** multires.loc["s"]

            self.loc["grid"] = grid
            self.loc["d_k"] = torch.randn((1, 1, size - 1, size - 1)).double().to(F.device)

            #stay on the same scale
            if self.cycle["cycle"][grid] == 0 and type(self.sols[s - 1]) == torch.Tensor:
                self.loc["d_k"] = self.sols[s - 1]

            #goes on a finer scale
            elif self.cycle["cycle"][grid] == 1:
                self.loc["d_k"] = F.up(self.sols[s - 2])

            print('----------- s = ' + str(multires.loc["s"]) + ' -----------' )

            self.solve_scale()

