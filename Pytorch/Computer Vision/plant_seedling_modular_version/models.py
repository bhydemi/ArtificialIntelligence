from torch import nn 
from torchvision import models  
import warnings
warnings.filterwarnings("ignore")


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = nn.Sequential(*(list(models.resnet101(pretrained=True).children())[:-2]))
        self.Linear = nn.Linear(in_features=100352, out_features=12)
    
    def forward(self, X):
        X =  self.resnet(X)
        X = X.view(X.shape[0], -1 )
        X = self.Linear(X)
        return X
    
