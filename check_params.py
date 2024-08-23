import torch.nn as nn
from torchvision.models import resnet18

batch_norm = nn.BatchNorm2d(num_features=3)

print(batch_norm.state_dict())

layer_norm = nn.LayerNorm(normalized_shape=[3, 3])
print(layer_norm.state_dict())

model = resnet18()
