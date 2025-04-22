import math
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F


class Plain_Vit(nn.Module):
    def __init__(self,
                 backbone_params={},
                 head_params={},
                ):
        super().__init__()
        self.backbone = timm.create_model(**backbone_params)
        self.head = nn.Linear(list(head_params.values())[0],list(head_params.values())[1])

    def forward(self,image):
        x=self.backbone(image)
#         b,c=x.shape
        x=self.head(x)
        
        return x
                        

                 
                 
        