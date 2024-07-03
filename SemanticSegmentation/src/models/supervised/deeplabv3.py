import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights

class DeepLabV3(nn.Module):
    def __init__(self, in_channels, out_channels, backbone='resnet50'):
        super(DeepLabV3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        #preprocess convolution to convert our input channels to 3 channels for the resnet backbones
        self.preprocess_conv = nn.Conv2d(in_channels, 3, kernel_size=1)

        
        if backbone == 'resnet50':
            self.deeplabv3 = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        elif backbone == 'resnet101':
            self.deeplabv3 = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        else:
            raise NotImplementedError(f"Backbone {backbone} is not implemented")

        self.deeplabv3.classifier = DeepLabHead(2048, out_channels)
        

    def forward(self, x):
        #replace nans with zeros
        x = torch.nan_to_num(x)
        #preprocess input to match Resnet input channels
        x = self.preprocess_conv(x)
        
        x = self.deeplabv3(x)['out']
        # Ensure the output size is the same size as mask
        # Resize the output to match the mask size (8x8) using interpolation
        x = F.interpolate(x, size=(8,8), mode='bilinear', align_corners=False)
        return x
    
    
# if __name__ == "__main__":
#     model = DeepLabV3(in_channels=22, out_channels=4)
#     x = torch.randn(12, 22, 200,200)
#     print(model(x).shape)
    
