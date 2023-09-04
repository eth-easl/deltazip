import torch

def hard_threshold(x, fraction_of_zero=0.1):
    """
    Set the smallest fraction_of_zero of x to zero.
    If fraction_of_zero is 0, then no thresholding is performed.
    """
    if fraction_of_zero == 0:
        return x
    if fraction_of_zero == 1:
        return torch.zeros_like(x)
    y, _ = torch.sort(x.view(-1).abs().clone())
    num_params = torch.numel(x)
    thresh_index = int(num_params * fraction_of_zero)
    threshold = y[thresh_index]
    mask = x.abs().clone().gt(threshold).type(torch.cuda.HalfTensor)
    return mask * x