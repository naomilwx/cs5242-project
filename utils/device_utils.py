import torch

def get_device():
    return torch.device('dml')
    dev = 'cpu'
    if torch.cuda.is_available():
        dev = 'cuda:0'
    elif torch.backends.mps.is_available():
        dev = 'mps'
    return torch.device(dev)