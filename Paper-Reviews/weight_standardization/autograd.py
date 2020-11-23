import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np


nrows = 9000
ntrain = int(nrows * .7)

X = torch.randn(nrows, 3)
y = torch.mm(X, torch.Tensor([[.1], [2.], [3.]]))
y = (y >= torch.mean(y)).long().view(nrows)

X_train = X[:ntrain, :]
y_train = y[:ntrain]
X_test  = X[ntrain:, :]
y_test  = y[ntrain:]

grad_dict: dict = {}


def fc_hook(layer_name, grad_input, grad_output)
    global grad_dict
    if layer_name in grad_dict:
        grad_dict[layer_name]["grad_input"].append(grad_input)
        grad_dict[layer_name]["grad_output"].append(grad_output)
    else :
        grad_dict[layer_name] = {}
        grad_dict[layer_name]["grad_input"] = []
        grad_dict[layer_name]["grad_output"] = []


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 20)
        self.fc2 = nn.Linear(20, 30)
        self.fc3 = nn.Linear(30, 2)
        
        self.fc1.register_backward_hook(self.fc1_backward_hook)
        self.fc2.register_backward_hook(self.fc2_backward_hook)
        self.fc3.register_backward_hook(self.fc3_backward_hook)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def backward_hook
    def fc1_backward_hook(, module, grad_input, grad_output):
        fc_hook("fc1", grad_input, grad_output)

    def fc2_backward_hook(self, module, grad_input, grad_output):
        fc_hook("fc2", grad_input, grad_output)

    def fc3_backward_hook(self, module, grad_input, grad_output):
        fc_hook("fc3", grad_input, grad_output)