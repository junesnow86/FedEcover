import torch
import torch.nn as nn

linear_layer1 = nn.Linear(10, 10)
nn.init.constant_(linear_layer1.weight, 1.0)
nn.init.constant_(linear_layer1.bias, 0.0)
linear_layer2 = nn.Linear(10, 1)
nn.init.constant_(linear_layer2.weight, 1.0)
nn.init.constant_(linear_layer2.bias, 0.0)
sequential1 = nn.Sequential(linear_layer1, linear_layer2)

x = torch.ones(10)
print(linear_layer1(x))
out1 = sequential1(x)
print(out1)

linear_layer1 = nn.Linear(5, 5)
nn.init.constant_(linear_layer1.weight, 1.0)
nn.init.constant_(linear_layer1.bias, 0.0)
linear_layer2 = nn.Linear(5, 1)
nn.init.constant_(linear_layer2.weight, 1.0)
nn.init.constant_(linear_layer2.bias, 0.0)
sequential2 = nn.Sequential(linear_layer1, linear_layer2)

x2 = torch.ones(5)
print(linear_layer1(x2))
out2 = sequential2(x2)
print(out2)
