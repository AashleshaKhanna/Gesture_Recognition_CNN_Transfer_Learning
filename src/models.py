"""Model definitions for ASL gesture recognition."""
from __future__ import annotations
import torch
import torch.nn as nn

class ASLCNN(nn.Module):
    """Custom CNN for 224x224 RGB ASL gesture images."""
    def __init__(self, num_classes:int=9, channels:tuple[int,int,int,int]=(32,64,128,256), dropout:float=0.5):
        super().__init__(); c1,c2,c3,c4=channels
        self.features=nn.Sequential(
            nn.Conv2d(3,c1,3,padding=1), nn.BatchNorm2d(c1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c1,c2,3,padding=1), nn.BatchNorm2d(c2), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c2,c3,3,padding=1), nn.BatchNorm2d(c3), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(c3,c4,3,padding=1), nn.BatchNorm2d(c4), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.classifier=nn.Sequential(nn.Flatten(), nn.Linear(c4*14*14,512), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(512,num_classes))
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.classifier(self.features(x))

class ConfigurableASLCNN(ASLCNN):
    pass

class AlexNetFeatureClassifier(nn.Module):
    """Classifier head trained on cached AlexNet features of shape (N,256,6,6)."""
    def __init__(self, num_classes:int=9, dropout:float=0.5):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(256,128,3,padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.classifier=nn.Sequential(nn.Flatten(), nn.Linear(128*3*3,256), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(256,num_classes))
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.classifier(self.conv(x))

def build_model(model_name:str, num_classes:int=9, channels:tuple[int,int,int,int]=(32,64,128,256), dropout:float=0.5)->nn.Module:
    name=model_name.lower()
    if name in {'custom','cnn','aslcnn'}: return ASLCNN(num_classes, channels, dropout)
    if name in {'alexnet_head','transfer'}: return AlexNetFeatureClassifier(num_classes, dropout)
    raise ValueError(f'Unsupported model name: {model_name}')
