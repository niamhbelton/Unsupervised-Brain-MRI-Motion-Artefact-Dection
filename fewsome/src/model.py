import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F



class resnet_batch(nn.Module):
    def __init__(self):
        super(resnet_batch, self).__init__()
        self.pretrained_model = models.resnet18(pretrained=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1):
        x1 = torch.squeeze(x1, dim=0)
        features = self.pretrained_model(x1)
        flattened_features1 = torch.max(features, 0, keepdim=True)[0]
        x1 = self.sigmoid(flattened_features1)
        return x1



class RESNET_pre(nn.Module):
  def __init__(self,vector_size, biases):
      super(RESNET_pre, self).__init__()

      self.features = models.resnet18(pretrained=True)


  def forward(self, x):
      x = torch.unsqueeze(x, dim =0)
      x= self.features(x)
      x=nn.Sigmoid()(x)

      return x #output



class ALEXNET_pre(nn.Module):
    def __init__(self):
        super(ALEXNET_pre, self).__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1):
        x1 = torch.squeeze(x1, dim=0)
        features = self.pretrained_model.features(x1)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features1 = torch.max(pooled_features, 0, keepdim=True)[0]
        x1 = self.sigmoid(flattened_features1)
        return x1
