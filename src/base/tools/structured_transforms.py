import torch

def hartley_transform(matrix, dim=(-2, -1), norm="ortho"):
    """
    Compute the Hartley transform of a matrix.
    :param matrix: the matrix to transform
    :param dim: the dimensions to transform
    :param norm: the normalization to apply
    :return: the Hartley transform of the matrix
    """

    fft_res = torch.fft.fft2(matrix, dim=dim, norm=norm)
    return torch.real(fft_res * (1 + 1j))
