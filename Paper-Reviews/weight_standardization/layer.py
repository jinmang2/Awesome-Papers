import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(42)

"""
i = input height/width
o = output height/width
p = padding
k = kernel_size
s = stride
d = dilation

o = [i + 2 * p - k - (k - 1) * (d - 1)] / s + 1
"""


def save_grad(names):
    def hook(grad):
        global grads
        for name in names:
            grads[name] = grad
    return hook


def module_hook(module, grad_input, grad_output):
    if grad_input is not None:
        if isinstance(grad_input, tuple):
            for i in range(len(grad_input)):
                print(f'{grad_input[i].shape}')
        else:
            print(grad_input.shape)
    if grad_output is not None:
        if isinstance(grad_output, tuple):
            for i in range(len(grad_output)):
                print(f'{grad_output[i].shape}')
        else:
            print(grad_output.shape)


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight_dot = weight - weight_mean
        std = weight_dot.view(weight_dot.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight_hat = weight_dot / std.expand_as(weight_dot)
        return F.conv2d(x, weight_hat, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


batch_size = 32
in_channels = 3
out_channels = 64
height = width = 30
kernel_size = 2

conv1 = nn.Conv2d(
    in_channels=in_channels, 
    out_channels=out_channels, 
    kernel_size=kernel_size,
    padding=1, dilation=2,
    bias=False)

conv2 = Conv2d(
    in_channels=in_channels, 
    out_channels=out_channels, 
    kernel_size=kernel_size,
    padding=1, dilation=2,
    bias=False)

conv2.weight = nn.Parameter(conv1.weight.clone())

# save origin weight
weight = conv1.weight.clone()

input = torch.randn(
    batch_size,
    in_channels,
    height,
    width
)
input.requires_grad_()

labels = torch.randn(batch_size).abs() * 10 // 5
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(conv1.parameters(), lr=0.01, momentum=0.9)
optimizer2 = optim.SGD(conv2.parameters(), lr=0.01, momentum=0.9)

# conv.register_forward_hook(module_hook)
# conv.register_backward_hook(module_hook)

conv1.zero_grad()
conv2.zero_grad()
optimizer1.zero_grad()
optimizer2.zero_grad()

output1 = conv1(input).view(batch_size, -1)
output2 = conv2(input).view(batch_size, -1)
print(weight - conv1.weight)

loss1 = criterion(output1, labels.long())
loss1.backward()
optimizer1.step()



loss2= criterion(output2, labels.long())
loss2.backward()
optimizer2.step()