from abc import abstractmethod
import torch
import numpy as np
from linop import BaseLinOp, LinOpMatrix, LinOpMul, LinOpFFT, LinOpFFT2, LinOpIdentity

class PhaseRetrievalBase():
    def __init__(self, linop:BaseLinOp):
        self.linop = linop

    def apply(self, x):
        return torch.abs(self.linop.apply(x))**2

    def apply_linop(self, x):
        return self.linop.apply(x)

    def apply_linopT(self, x):
        return self.linop.applyT(x)

    @abstractmethod
    def build_lin_op(self):
        pass

class ExplicitPR(PhaseRetrievalBase):
    def __init__(self, A):
        self.A = A
        self.linop = self.build_lin_op()

    def build_lin_op(self) -> BaseLinOp:
        return LinOpMatrix(self.A)

class FourierFilterPR(PhaseRetrievalBase):
    def __init__(self, filter):
        self.filter = filter
        self.linop = self.build_lin_op()

    def build_lin_op(self) -> BaseLinOp:
        op_fft = LinOpFFT()
        op_filter = LinOpMul(self.filter)
        return op_fft.T() @ op_filter @ op_fft

class FourierFilterPR2(PhaseRetrievalBase):
    def __init__(self, fourier_filter, object_mask=None):
        self.fourier_filter = fourier_filter
        self.object_mask = object_mask
        self.linop = self.build_lin_op()

    def build_lin_op(self) -> BaseLinOp:
        if self.object_mask is not None:
            op_mask = LinOpMul(self.object_mask)
        else:
            op_mask = LinOpIdentity()
        op_fft2 = LinOpFFT2()
        op_filter = LinOpMul(self.fourier_filter)
        return op_fft2.T() @ op_filter @ op_fft2 @ op_mask
