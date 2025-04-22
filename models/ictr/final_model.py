import math
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
# from models.vit_backbone import ViT
from models.cltr.simpleFPN_neck import SimpleFPN
from models.cltr.classification_head import Cls_Head


class F_Model(nn.Module):
    def __init__(self,
                 backbone_params={},
                 neck_params={},
                 head_params={},
                ):
        super().__init__()
#         self.backbone = ViT(**backbone_params)
        self.backbone = timm.create_model(**backbone_params)
        self.neck = SimpleFPN(**neck_params)
        self.head=Cls_Head(**head_params)
        
    def forward(self,image):
        x=self.backbone(image)
        b,c=x.shape
        x=x.view(b,int(c/4),2,2)
        x=self.neck(x)
        x=self.head(x)
        return x
                        

                 
                 
        