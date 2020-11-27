import torch
import torch.nn as nn

def get_batch_size():
    return 8

def get_hidden_size():
    return 8

def get_learning_rate():
    return 0.001

def get_num_epochs():
    return 1000

def get_ignore_words():
    return ['?', '!', '.', ',']

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_is_shuffle():
    return True

def get_num_workers():
    return 0

def get_criterion ():
    return nn.CrossEntropyLoss()

def get_optimizer(parameters, learning_rate):
    return torch.optim.Adam(parameters, lr=learning_rate)