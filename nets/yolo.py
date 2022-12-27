import torch
import torch.nn as nn
import math

from .darknet import BaseConv, CSPLayer,DWConv, CBAM
from .edgeVit import edgevit_s

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [256, 512, 1024],
                 act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
        self.stems      = nn.ModuleList()  

        self.cls_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()

        self.reg_convs  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()

        self.stems.append(BaseConv(in_channels = 72, out_channels = 72, ksize = 1, stride = 1, act = act))

        self.cls_convs.append(nn.Sequential(*[
            Conv(in_channels = 72, out_channels = 72, ksize = 3, stride = 1, act = act), 
            Conv(in_channels = 72, out_channels = 72, ksize = 3, stride = 1, act = act), 
        ]))
        self.cls_preds.append(
            nn.Conv2d(in_channels = 72, out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
        )

        self.reg_convs.append(nn.Sequential(*[
            Conv(in_channels = 72, out_channels = 72, ksize = 3, stride = 1, act = act), 
            Conv(in_channels = 72, out_channels = 72, ksize = 3, stride = 1, act = act)
        ]))
        self.reg_preds.append(
            nn.Conv2d(in_channels = 72, out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
        )
        self.obj_preds.append(
            nn.Conv2d(in_channels = 72, out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
        )

        self.stems.append(BaseConv(in_channels = 144, out_channels = 72, ksize = 1, stride = 1, act = act))
        self.cls_convs.append(nn.Sequential(*[
            Conv(in_channels = 72, out_channels = 72, ksize = 3, stride = 1, act = act), 
            Conv(in_channels = 72, out_channels = 72, ksize = 3, stride = 1, act = act), 
        ]))
        self.cls_preds.append(
            nn.Conv2d(in_channels = 72, out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
        )
        self.reg_convs.append(nn.Sequential(*[
            Conv(in_channels = 72, out_channels = 72, ksize = 3, stride = 1, act = act), 
            Conv(in_channels = 72, out_channels = 72, ksize = 3, stride = 1, act = act)
        ]))
        self.reg_preds.append(
            nn.Conv2d(in_channels = 72, out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
        )
        self.obj_preds.append(
            nn.Conv2d(in_channels = 72, out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
        )
        self.stems.append(BaseConv(in_channels = 288, out_channels = 72, ksize = 1, stride = 1, act = act))
        self.cls_convs.append(nn.Sequential(*[
            Conv(in_channels = 72, out_channels = 72, ksize = 3, stride = 1, act = act), 
            Conv(in_channels = 72, out_channels = 72, ksize = 3, stride = 1, act = act), 
        ]))
        self.cls_preds.append(
            nn.Conv2d(in_channels = 72, out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
        )
        self.reg_convs.append(nn.Sequential(*[
            Conv(in_channels = 72, out_channels = 72, ksize = 3, stride = 1, act = act), 
            Conv(in_channels = 72, out_channels = 72, ksize = 3, stride = 1, act = act)
        ]))
        self.reg_preds.append(
            nn.Conv2d(in_channels = 72, out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
        )
        self.obj_preds.append(
            nn.Conv2d(in_channels = 72, out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
        )


    def forward(self, inputs):

        outputs = []
        for k, x in enumerate(inputs):
            x       = self.stems[k](x)
            cls_feat    = self.cls_convs[k](x)
            cls_output  = self.cls_preds[k](cls_feat)
            reg_feat    = self.reg_convs[k](x)
            reg_output  = self.reg_preds[k](reg_feat)
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs

class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0,
                 in_features = ("dark3", "dark4", "dark5"),
                in_channels = [72, 144, 288], depthwise = False, act = "silu"):

        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        
        self.backbone = edgevit_s()

        self.in_features    = in_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.cbam1 = CBAM(c1 = int(in_channels[1] * width))
        self.cbam2 = CBAM(c1 = int(in_channels[0] * width))
        self.cbam3 = CBAM(c1 = int(in_channels[0] * width))
        self.cbam4 = CBAM(c1 = int(in_channels[1] * width))
        
        self.lateral_conv0  = BaseConv(288, 144, 1, 1, act=act)
    
        self.C3_p4 = CSPLayer(
            288,
            144,
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  
        self.reduce_conv1  = BaseConv(144, 72, 1, 1, act=act)
        self.C3_p3 = CSPLayer(
            144,
            72,
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        self.bu_conv2       = Conv(72, 72, 3, 2, act=act)
        self.C3_n3 = CSPLayer(
            144,
            144,
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        self.bu_conv1       = Conv(144, 144, 3, 2, act=act)
        self.C3_n4 = CSPLayer(
            288,
            288,
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

    def forward(self, input):
        feature4x, feature8x, feature16x, feature32x, feature32x2 = self.backbone.forward(input)  # 获得backbone部分输出
       
        P5          = self.lateral_conv0(feature32x2)
        P5_upsample = self.upsample(P5)
        P5_upsample = torch.cat([P5_upsample, feature32x], 1)
        P5_upsample = self.C3_p4(P5_upsample)

        P4          = self.reduce_conv1(P5_upsample) 
        P4_upsample = self.upsample(P4)
        P4_upsample = torch.cat([P4_upsample, feature16x], 1)
       
        P3_out      = self.C3_p3(P4_upsample)  
        P3_downsample   = self.bu_conv2(P3_out)
        P3_downsample   = torch.cat([P3_downsample, P4], 1)

        P4_out          = self.C3_n3(P3_downsample) 
        P4_downsample   = self.bu_conv1(P4_out)
        P4_downsample   = torch.cat([P4_downsample, P5], 1)

        P5_out          = self.C3_n4(P4_downsample)

        return (P3_out, P4_out, P5_out)

class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[phi], width_dict[phi]

        depthwise       = True if phi == 'nano' else False

        self.backbone   = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head       = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs    = self.backbone.forward(x)
        outputs     = self.head.forward(fpn_outs)
        return outputs
