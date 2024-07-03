import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(DilatedConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class PatchPartition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PatchPartition, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(x)

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PatchMerging, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class PatchExpanding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PatchExpanding, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv_transpose(x)))

class LinearEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class DilatedUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedUNet, self).__init__()
        self.patch_partition = PatchPartition(in_channels, 48)
        self.linear_embedding = LinearEmbedding(48, 64)

        self.encoder1 = DilatedConvBlock(64, 64, dilation_rate=1)
        self.encoder2 = DilatedConvBlock(64, 128, dilation_rate=1)
        self.encoder3 = DilatedConvBlock(128, 256, dilation_rate=2)
        self.encoder4 = DilatedConvBlock(256, 512, dilation_rate=3)

        self.patch_merging1 = PatchMerging(64, 64)
        self.patch_merging2 = PatchMerging(128, 128)
        self.patch_merging3 = PatchMerging(256, 256)

        self.decoder1 = DilatedConvBlock(512, 256, dilation_rate=3)  # 512 from expanding + 256 from skip connection
        self.decoder2 = DilatedConvBlock(256, 128, dilation_rate=2)  # 256 from expanding + 128 from skip connection
        self.decoder3 = DilatedConvBlock(128, 64, dilation_rate=1)   # 128 from expanding + 64 from skip connection

        self.patch_expanding1 = PatchExpanding(512, 256)
        self.patch_expanding2 = PatchExpanding(256, 128)
        self.patch_expanding3 = PatchExpanding(128, 64)
        
        self.adaptive_pool = nn.AdaptiveMaxPool2d((8,8))  # can change for different slice size (needs to match the size of ground truth)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.patch_partition(x)
        x = self.linear_embedding(x)

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.patch_merging1(e1))
        e3 = self.encoder3(self.patch_merging2(e2))
        e4 = self.encoder4(self.patch_merging3(e3))

        # Decoder
        d1 = self.patch_expanding1(e4)
        d1 = torch.cat([d1, e3], dim=1)
        d1 = self.decoder1(d1)

        d2 = self.patch_expanding2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)

        d3 = self.patch_expanding3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.decoder3(d3)
        
        out = self.final_conv(d3)
        out = self.adaptive_pool(out)
        # out = F.softmax(out, dim=1)
        # print(out) 
        return out

# # # # # test usage
# if __name__ == "__main__":
#     model = DilatedUNet(in_channels=22, out_channels=4)
#     x = torch.randn(12, 22, 400,400)
#     print(model(x).shape)
    
    
#     # x = torch.randn(12, 12, 200, 200)
#     # print(model(x).shape)


