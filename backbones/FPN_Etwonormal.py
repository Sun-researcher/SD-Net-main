import torch
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.nn.functional as F
#本实验探讨网络是否过度归一化导致网络性能减低

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class CBAMSpatial(nn.Module):
    # 空间注意力机制的添加
    def __init__(self, kernel_size=7, inchannel=64, pool_kernel=2):
        # inchannel为输入通道
        super(CBAMSpatial, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.MaxPool2d(pool_kernel, pool_kernel),
            nn.BatchNorm2d(1),
            nn.SiLU()
        )
        self.gamma = nn.Parameter(torch.zeros(inchannel))
        self.bn = nn.BatchNorm2d(inchannel)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x, y):
        # 计算平均池化和最大池化
        avg_out = torch.mean(y, dim=1, keepdim=True)
        max_out, _ = torch.max(y, dim=1, keepdim=True)
        # 将两个结果在通道维度上拼接
        out = torch.cat([avg_out, max_out], dim=1)
        # 通过卷积层
        out = self.conv(out)
        # 对输入 x 进行加权后相加
        # print((self.gamma.view(1, -1, 1, 1) * x).shape, out.shape)
        out = self.gamma.view(1, -1, 1, 1) * x * out + x
        # 经过批归一化和激活函数
        out = self.bn(out)
        out = self.silu(out)
        return out

class AttFPN_efficientnet_b4(nn.Module):
    def __init__(self):
        super(AttFPN_efficientnet_b4, self).__init__()
        self.parames = {
            'channels': [1792, 448, 160, 56, 32],
            'dim': 3
        }
        # Multi-layer features
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.parames['channels'][0], self.parames['channels'][1], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.parames['channels'][1]),
            nn.SiLU(inplace=True))
        self.bn1 = nn.BatchNorm2d(self.parames['channels'][1])
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.parames['channels'][1], self.parames['channels'][2], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.parames['channels'][2]),
            nn.SiLU(inplace=True))
        self.bn2 = nn.BatchNorm2d(self.parames['channels'][2])
        self.conv3 = nn.Sequential(nn.Conv2d(self.parames['channels'][2], self.parames['channels'][3], 1, 1, 0),
                                   nn.BatchNorm2d(self.parames['channels'][3]),
                                   nn.SiLU(inplace=True))
        self.bn3 = nn.BatchNorm2d(self.parames['channels'][3])
        self.conv4 = nn.Sequential(nn.Conv2d(self.parames['channels'][3], self.parames['channels'][4], 1, 1, 0),
                                   nn.BatchNorm2d(self.parames['channels'][4]),
                                   nn.SiLU(inplace=True))
        self.bn4 = nn.BatchNorm2d(self.parames['channels'][4])
        # Final layers
        self.mscs1 = nn.Sequential(SeparableConv2d(self.parames['channels'][4], self.parames['channels'][4], 3, 1, 1),
                                   SeparableConv2d(self.parames['channels'][4], self.parames['channels'][4], 3, 1, 1),
                                   SeparableConv2d(self.parames['channels'][4], 3, 3, 1, 1))
        self.mscs2 = nn.Sequential(SeparableConv2d(self.parames['channels'][4], self.parames['channels'][4], 3, 1, 1),
                                   SeparableConv2d(self.parames['channels'][4], 3, 3, 1, 1))
        self.mscs3 = nn.Sequential(SeparableConv2d(self.parames['channels'][4], 3, 3, 1, 1))
        self.bn = nn.BatchNorm2d(3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.conv1(x['reduction_6'])
        out= torch.add(out,x['reduction_5'])
        out = self.bn1(out)
        out_upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv2(out_upsample)
        out = torch.add(out,x['reduction_4'])
        out = self.bn2(out)
        out_upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv3(out_upsample)
        out = torch.add(out,x['reduction_3'])
        out = self.bn3(out)
        out_upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv4(out_upsample)
        out = torch.add(out,x['reduction_2'])
        out = self.bn4(out)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        out_1 = self.mscs1(out)
        out_2 = self.mscs2(out)
        out_3 = self.mscs3(out)
        out = out_1+out_2+out_3
        out = self.bn(out)
        out = self.sigmoid(out)
        return out
class AttFPN_efficientnet_b4_CBAM(nn.Module):
    def __init__(self):
        super(AttFPN_efficientnet_b4_CBAM, self).__init__()
        self.parames = {
            'channels': [1792, 448, 160, 56, 32],
            'dim': 3
        }
        # Multi-layer features
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.parames['channels'][0], self.parames['channels'][1], kernel_size=1, stride=1, padding=0))
        self.bn1 = nn.BatchNorm2d(self.parames['channels'][1])
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.parames['channels'][1], self.parames['channels'][2], kernel_size=1, stride=1, padding=0))
        self.bn2 = nn.BatchNorm2d(self.parames['channels'][2])
        self.conv3 = nn.Sequential(nn.Conv2d(self.parames['channels'][2], self.parames['channels'][3], 1, 1, 0),
                                   nn.BatchNorm2d(self.parames['channels'][3]),
                                   nn.SiLU(inplace=True))
        self.bn3 = nn.BatchNorm2d(self.parames['channels'][3])
        self.conv4 = nn.Sequential(nn.Conv2d(self.parames['channels'][3], self.parames['channels'][4], 1, 1, 0),
                                   nn.BatchNorm2d(self.parames['channels'][4]),
                                   nn.SiLU(inplace=True))
        self.bn4 = nn.BatchNorm2d(self.parames['channels'][4])
        self.cbam1 = CBAMSpatial(inchannel=self.parames['channels'][1], pool_kernel=8)  # 加如空间注意力
        self.cbam2 = CBAMSpatial(inchannel=self.parames['channels'][2], pool_kernel=2)
        # Final layers
        self.mscs1 = nn.Sequential(SeparableConv2d(self.parames['channels'][4], self.parames['channels'][4], 3, 1, 1),
                                   SeparableConv2d(self.parames['channels'][4], self.parames['channels'][4], 3, 1, 1),
                                   SeparableConv2d(self.parames['channels'][4], 3, 3, 1, 1))
        self.mscs2 = nn.Sequential(SeparableConv2d(self.parames['channels'][4], self.parames['channels'][4], 3, 1, 1),
                                   SeparableConv2d(self.parames['channels'][4], 3, 3, 1, 1))
        self.mscs3 = nn.Sequential(SeparableConv2d(self.parames['channels'][4], 3, 3, 1, 1))
        self.bn = nn.BatchNorm2d(3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.conv1(x['reduction_6'])
        out = self.cbam1(out, x['reduction_2'])
        out = torch.add(out, x['reduction_5'])
        out = self.bn1(out)
        out_upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv2(out_upsample)
        out = self.cbam2(out, x['reduction_3'])
        out = torch.add(out, x['reduction_4'])
        out = self.bn2(out)
        out_upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv3(out_upsample)
        out = torch.add(out, x['reduction_3'])
        out = self.bn3(out)
        out_upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv4(out_upsample)
        out = torch.add(out, x['reduction_2'])
        out = self.bn4(out)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
        out_1 = self.mscs1(out)
        out_2 = self.mscs2(out)
        out_3 = self.mscs3(out)
        out = out_1 + out_2 + out_3
        out = self.bn(out)
        out = self.sigmoid(out)
        return out

class AttFPN_efficientnet_b4_CBAM_v1(nn.Module):
    def __init__(self):
        super(AttFPN_efficientnet_b4_CBAM_v1, self).__init__()
        self.parames = {
            'channels': [1792, 448, 160, 56, 32, 24],
            'dim': 3
        }
        # Multi-layer features
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.parames['channels'][0], self.parames['channels'][1], kernel_size=1, stride=1, padding=0))
        self.bn1 = nn.BatchNorm2d(self.parames['channels'][1])
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.parames['channels'][1], self.parames['channels'][2], kernel_size=1, stride=1, padding=0))
        self.bn2 = nn.BatchNorm2d(self.parames['channels'][2])
        self.conv3 = nn.Sequential(nn.Conv2d(self.parames['channels'][2], self.parames['channels'][3], 1, 1, 0),
                                   nn.BatchNorm2d(self.parames['channels'][3]),
                                   nn.SiLU(inplace=True))
        self.bn3 = nn.BatchNorm2d(self.parames['channels'][3])
        self.conv4 = nn.Sequential(nn.Conv2d(self.parames['channels'][3], self.parames['channels'][4], 1, 1, 0),
                                   nn.BatchNorm2d(self.parames['channels'][4]),
                                   nn.SiLU(inplace=True))
        self.bn4 = nn.BatchNorm2d(self.parames['channels'][4])
        self.conv5 = nn.Sequential(nn.Conv2d(self.parames['channels'][4], self.parames['channels'][5], 1, 1, 0),
                                   nn.BatchNorm2d(self.parames['channels'][5]),
                                   nn.SiLU(inplace=True))
        self.bn5 = nn.BatchNorm2d(self.parames['channels'][5])
        self.cbam1 = CBAMSpatial(inchannel=self.parames['channels'][1], pool_kernel=16)  # 加如空间注意力
        self.cbam2 = CBAMSpatial(inchannel=self.parames['channels'][2], pool_kernel=4)
        self.cbam3 = CBAMSpatial(inchannel=self.parames['channels'][3], pool_kernel=1)
        # Final layers
        self.mscs1 = nn.Sequential(SeparableConv2d(self.parames['channels'][5], self.parames['channels'][5], 3, 1, 1),
                                   SeparableConv2d(self.parames['channels'][5], self.parames['channels'][5], 3, 1, 1),
                                   SeparableConv2d(self.parames['channels'][5], 3, 3, 1, 1))
        self.mscs2 = nn.Sequential(SeparableConv2d(self.parames['channels'][5], self.parames['channels'][5], 3, 1, 1),
                                   SeparableConv2d(self.parames['channels'][5], 3, 3, 1, 1))
        self.mscs3 = nn.Sequential(SeparableConv2d(self.parames['channels'][5], 3, 3, 1, 1))
        self.bn = nn.BatchNorm2d(3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.conv1(x['reduction_6'])
        out = self.cbam1(out, x['reduction_1'])
        out = torch.add(out, x['reduction_5'])
        out = self.bn1(out)
        out_upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv2(out_upsample)
        out = self.cbam2(out, x['reduction_2'])
        out = torch.add(out, x['reduction_4'])
        out = self.bn2(out)
        out_upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv3(out_upsample)
        out = self.cbam3(out, x['reduction_3'])
        out = torch.add(out, x['reduction_3'])
        out = self.bn3(out)
        out_upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv4(out_upsample) #注意力完毕
        out = torch.add(out, x['reduction_2'])
        out = self.bn4(out)
        out_upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv5(out_upsample)
        out = torch.add(out, x['reduction_1'])
        out = self.bn5(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out_1 = self.mscs1(out)
        out_2 = self.mscs2(out)
        out_3 = self.mscs3(out)
        out = out_1 + out_2 + out_3
        out = self.bn(out)
        out = self.sigmoid(out)
        return out
class AttFPN_efficientnet_b4_CBAM_v2_1(nn.Module):
    def __init__(self):
        super(AttFPN_efficientnet_b4_CBAM_v2_1, self).__init__()
        self.parames = {
            'channels': [1792, 448, 160, 56, 32,24],
            'dim': 3
        }
        # Multi-layer features
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.parames['channels'][0], self.parames['channels'][1], kernel_size=1, stride=1, padding=0))
        self.bn1 = nn.BatchNorm2d(self.parames['channels'][1])
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.parames['channels'][1], self.parames['channels'][2], kernel_size=1, stride=1, padding=0))
        self.bn2 = nn.BatchNorm2d(self.parames['channels'][2])
        self.conv3 = nn.Sequential(nn.Conv2d(self.parames['channels'][2], self.parames['channels'][3], 1, 1, 0),
                                   nn.BatchNorm2d(self.parames['channels'][3]),
                                   nn.SiLU(inplace=True))
        self.bn3 = nn.BatchNorm2d(self.parames['channels'][3])

        self.cbam1 = CBAMSpatial(inchannel=self.parames['channels'][1], pool_kernel=4)  # 加如空间注意力
        self.cbam2 = CBAMSpatial(inchannel=self.parames['channels'][2], pool_kernel=1)
        # Final layers
        self.mscs1 = nn.Sequential(SeparableConv2d(self.parames['channels'][3], self.parames['channels'][3], 3, 1, 1),
                                   SeparableConv2d(self.parames['channels'][3], self.parames['channels'][3], 3, 1, 1),
                                   SeparableConv2d(self.parames['channels'][3], 3, 3, 1, 1))
        self.mscs2 = nn.Sequential(SeparableConv2d(self.parames['channels'][3], self.parames['channels'][3], 3, 1, 1),
                                   SeparableConv2d(self.parames['channels'][3], 3, 3, 1, 1))
        self.mscs3 = nn.Sequential(SeparableConv2d(self.parames['channels'][3], 3, 3, 1, 1))
        self.bn = nn.BatchNorm2d(3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.conv1(x['reduction_6'])
        out = self.cbam1(out, x['reduction_3'])
        out = torch.add(out, x['reduction_5'])
        out = self.bn1(out)
        out_upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv2(out_upsample)
        out = self.cbam2(out, x['reduction_4'])
        out = torch.add(out, x['reduction_4'])
        out = self.bn2(out)
        out_upsample = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.conv3(out_upsample)
        out = torch.add(out, x['reduction_3'])
        out = self.bn3(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out_1 = self.mscs1(out)
        out_2 = self.mscs2(out)
        out_3 = self.mscs3(out)
        out = out_1 + out_2 + out_3
        out = self.bn(out)
        out = self.sigmoid(out)
        return out
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = EfficientNet.from_name("efficientnet-b4", num_classes=2)
    model2 = AttFPN_efficientnet_b4_CBAM()
    inputs = torch.rand(10, 3, 384, 384)
    outputs = model.extract_endpoints(inputs)
    outputs2 = model2(outputs)
    count=count_parameters(model2)
    print(count,outputs2.shape)