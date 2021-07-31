import numpy as np 
import torch
import torch.nn as nn
from torchvision.models import resnet50

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #torch.nn.init.xavier_normal_(m.weight)
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")


class BottleneckSSMA(nn.Module):
    """PyTorch Module for multi-scale units (modified residual units) for Resnet50 stages"""

    def __init__(self, in_channels, out_channels, r1, r2, d3, stride=1, downsample=None, copy_from=None, drop_out=True):
        """
        :param in_channels: input dimension
        :param out_channels: output dimension
        :param r1: dilation rate and padding 1
        :param r2: dilation rate and padding 2
        :param d3: split factor
        :param stride: stride
        :param downsample: downsample rate
        :param copy_from: copy of residual unit from second/third stage resnet50
        :param drop_out: boolean for inclusion of dropout layer
        """
        super(BottleneckSSMA, self).__init__()
        self.dropout = drop_out

        half_d3 = int(d3 / 2)

        self.conv2a = nn.Conv2d(out_channels, half_d3, kernel_size=3, stride=1, dilation=r1, padding=r1, bias=False)
        self.bn2a = nn.BatchNorm2d(half_d3)
        self.conv2b = nn.Conv2d(out_channels, half_d3, kernel_size=3, stride=1, dilation=r2, padding=r2, bias=False)
        self.bn2b = nn.BatchNorm2d(half_d3)
        self.conv3 = nn.Conv2d(d3, in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)

        if copy_from is None:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.conv1 = copy_from.conv1
            self.bn1 = copy_from.bn1

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward Pass
        :param x: input feature maps
        :return: output feature maps
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out_a = self.conv2a(out)
        out_a = self.bn2a(out_a)
        out_a = self.relu(out_a)

        out_b = self.conv2b(out)
        out_b = self.bn2b(out_b)
        out_b = self.relu(out_b)

        out = torch.cat((out_a, out_b), dim=1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.dropout:
            m = nn.Dropout(p=0.5)
            out = m(out)

        return out


class Encoder(nn.Module):
    """PyTorch Module for encoder"""

    def __init__(self):
        super(Encoder, self).__init__()

        self.enc_skip2_conv = nn.Conv2d(256, 24, kernel_size=1, stride=1)
        self.enc_skip2_conv_bn = nn.BatchNorm2d(24)
        self.enc_skip1_conv = nn.Conv2d(512, 24, kernel_size=1, stride=1)
        self.enc_skip1_conv_bn = nn.BatchNorm2d(24)

        nn.init.kaiming_uniform_(self.enc_skip2_conv.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.enc_skip1_conv.weight, nonlinearity="relu")

        self.res_n50_enc = resnet50(pretrained=True)

        self.res_n50_enc.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.res_n50_enc.bn1 = nn.BatchNorm2d(64)

        self.res_n50_enc.layer2[-1] = BottleneckSSMA(512, 128, 1, 2, 64, copy_from=self.res_n50_enc.layer2[-1])

        u3_sizes_block = [(1024, 256, 1, 2,  256), 
                          (1024, 256, 1, 16, 256), 
                          (1024, 256, 1, 8,  256),
                          (1024, 256, 1, 4,  256)]
        for i, x in enumerate(u3_sizes_block):
            dropout = i == 0
            self.res_n50_enc.layer3[i+2] = BottleneckSSMA(x[0], x[1], x[2], x[3], x[4],
                                                        copy_from=self.res_n50_enc.layer3[i+2],
                                                        drop_out=dropout)

        u4_sizes_block = [(2048, 512, 2, 4, 512), 
                          (2048, 512, 2, 8, 512), 
                          (2048, 512, 2, 16, 512)]

        for i, res in enumerate(u4_sizes_block):
            downsample = None
            if i == 0:
                downsample = self.res_n50_enc.layer4[0].downsample
                downsample[0].stride = (1, 1)

            self.res_n50_enc.layer4[i] = BottleneckSSMA(res[0], res[1], res[2], res[3], res[4], 
                                                        downsample=downsample,
                                                        copy_from=self.res_n50_enc.layer4[i])


    def forward(self, x):       
        x = self.res_n50_enc.conv1(x)
        x = self.res_n50_enc.bn1(x)
        x = self.res_n50_enc.relu(x)
        x = self.res_n50_enc.maxpool(x)

        x = self.res_n50_enc.layer1(x)
        s2 = self.enc_skip2_conv_bn(self.enc_skip2_conv(x))

        x = self.res_n50_enc.layer2(x)
        s1 = self.enc_skip1_conv_bn(self.enc_skip1_conv(x))

        x = self.res_n50_enc.layer3(x)

        x = self.res_n50_enc.layer4(x)

        return x, s2, s1


class eASPP(nn.Module):
    """PyTorch Module for eASPP"""

    def __init__(self, in_chs, mid_chs, out_chs):

        super(eASPP, self).__init__()

        # branch 1
        self.branch1_conv = nn.Conv2d(in_chs, out_chs, kernel_size=1)
        self.branch1_bn = nn.BatchNorm2d(out_chs)

        self.branch234 = nn.ModuleList([])
        branch_rates = [3, 6, 12]
        for rate in branch_rates:
            # branch 2-4
            branch = nn.Sequential(
                nn.Conv2d(in_chs, mid_chs, kernel_size=1),
                nn.BatchNorm2d(mid_chs),
                nn.ReLU(),
                nn.Conv2d(mid_chs, mid_chs, kernel_size=3, dilation=rate, padding=rate),
                nn.BatchNorm2d(mid_chs),
                nn.ReLU(),
                nn.Conv2d(mid_chs, mid_chs, kernel_size=3, dilation=rate, padding=rate),
                nn.BatchNorm2d(mid_chs),
                nn.ReLU(),
                nn.Conv2d(mid_chs, out_chs, kernel_size=1),
                nn.BatchNorm2d(out_chs),
                nn.ReLU(),
            )
            self.branch234.append(branch)

        # branch 5
        self.branch5_conv = nn.Conv2d(in_chs, out_chs, kernel_size=1)
        self.branch5_bn = nn.BatchNorm2d(out_chs)

        # final layer
        self.eASPP_fin_conv = nn.Conv2d(out_chs * 5, out_chs, kernel_size=1)
        self.eASPP_fin_bn = nn.BatchNorm2d(out_chs)

    def forward(self, x):
        """Forward pass
        :param x: input from encoder (in stage 1) or from fused encoders (in stage 2 and 3)
        :return: feature maps to be forwarded to decoder
        """
        # branch 1: 1x1 convolution
        out = torch.relu(self.branch1_bn(self.branch1_conv(x)))

        # branch 2-4: atrous pooling
        y = self.branch234[0](x)
        out = torch.cat((out, y), 1)
        y = self.branch234[1](x)
        out = torch.cat((out, y), 1)
        y = self.branch234[2](x)
        out = torch.cat((out, y), 1)

        # branch 5: image pooling
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        #x = torch.relu(self.branch5_bn(self.branch5_conv(x)))
        x = torch.relu(self.branch5_conv(x))
        x = nn.Upsample((out.shape[2], out.shape[3]), mode="bilinear", align_corners=True)(x)
        out = torch.cat((out, x), 1)

        out = torch.relu(self.eASPP_fin_bn(self.eASPP_fin_conv(out)))

        return out


class Decoder(nn.Module):
    """PyTorch Module for decoder"""

    def __init__(self, C, fusion=False):
        """Constructor
        :param C: Number of categories
        :param fusion: boolean for fused skip connections (False for stage 1, True for stages 2 and 3)
        """
        super(Decoder, self).__init__()

        # variables
        self.n_classes = C
        self.fusion = fusion

        # layers stage 1
        self.deconv1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.deconv1_bn = nn.BatchNorm2d(256)

        # layers stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(280, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256)
        )

        # layers stage 3
        self.stage3 = nn.Sequential(
            nn.Conv2d(280, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, self.n_classes, 1),
            nn.BatchNorm2d(self.n_classes),
            nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(self.n_classes)
        )

        # decoder auxiliary layers
        self.aux_conv1 = nn.Conv2d(256, self.n_classes, 1)
        self.aux_conv1_bn = nn.BatchNorm2d(self.n_classes)
        self.aux_conv2 = nn.Conv2d(256, self.n_classes, 1)
        self.aux_conv2_bn = nn.BatchNorm2d(self.n_classes)

        # decoder fuse skip layers
        self.fuse_conv1 = nn.Conv2d(256, 24, 1)
        self.fuse_conv1_bn = nn.BatchNorm2d(24)
        self.fuse_conv2 = nn.Conv2d(256, 24, 1)
        self.fuse_conv2_bn = nn.BatchNorm2d(24)

    def forward(self, x, skip1, skip2):
        """Forward pass
        :param x: input feature maps from eASPP
        :param skip1: skip connection 1
        :param skip2: skip connection 2
        :return: final output and auxiliary output 1 and 2
        """
        # stage 1
        x = torch.relu(self.deconv1_bn(self.deconv1(x)))
        y1 = self.aux(x, self.aux_conv1, self.aux_conv1_bn, 8)
        if self.fusion:
            # integrate fusion skip
            int_fuse_skip = self.integrate_fuse_skip(x, skip1, self.fuse_conv1, self.fuse_conv1_bn)
            x = torch.cat((x, int_fuse_skip), 1)
        else:
            x = torch.cat((x, skip1), 1)

        # stage 2
        x = self.stage2(x)
        y2 = self.aux(x, self.aux_conv2, self.aux_conv2_bn, 4)
        if self.fusion:
            # integrate fusion skip
            int_fuse_skip = self.integrate_fuse_skip(x, skip2, self.fuse_conv2, self.fuse_conv2_bn)
            x = torch.cat((x, int_fuse_skip), 1)
        else:
            x = torch.cat((x, skip2), 1)

        # stage 3
        y3 = self.stage3(x)

        return y1, y2, y3

    def aux(self, x, conv, bn, scale):
        """Compute auxiliary output"""
        x = bn(conv(x))
        return nn.UpsamplingBilinear2d(scale_factor=scale)(x)

    def integrate_fuse_skip(self, x, fuse_skip, conv, bn):
        """Integrate fuse skip connection with decoder"""
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        #x = torch.relu(bn(conv(x)))
        x = torch.relu(conv(x))
        return torch.mul(x, fuse_skip)


class SSMA(nn.Module):
    """PyTorch Module for SSMA"""

    def __init__(self, features, bottleneck):
        """Constructor
        :param features: number of feature maps
        :param bottleneck: bottleneck compression rate
        """
        super(SSMA, self).__init__()
        reduce_size = int(features / bottleneck)
        double_features = int(2 * features)
        self.link = nn.Sequential(
            nn.Conv2d(double_features, reduce_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(reduce_size, double_features, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(double_features, features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features)
        )

    def forward(self, x1, x2):
        """Forward pass
        :param x1: input data from encoder 1
        :param x2: input data from encoder 2
        :return: Fused feature maps
        """
        x_12 = torch.cat((x1, x2), dim=1)

        x_12_est = self.link(x_12)
        x_12 = x_12 * x_12_est
        x_12 = self.final_conv(x_12)

        return x_12

class AdapNet(nn.Module):
    """PyTorch module for 'AdapNet++' and 'AdapNet++ with fusion architecture' """

    def __init__(self, config):

        super(AdapNet, self).__init__()
        
        self.stage = config.stage
        self.n_classes = config.n_classes
        self.fusion = False

        if self.stage == 1:
            self.encoder_mod1 = Encoder()
            self.eASPP = eASPP(2048, 64, 256)

        else:
            self.fusion = True

            self.encoder_mod1 = Encoder()
            self.encoder_mod2 = Encoder()

            self.eASPP_mod1 = eASPP(2048, 64, 256)
            self.eASPP_mod2 = eASPP(2048, 64, 256)
            self.ssma_res = SSMA(256, 16)

            self.ssma_s1 = SSMA(24, 6)
            self.ssma_s2 = SSMA(24, 6)

        self.decoder = Decoder(self.n_classes, self.fusion)

    def no_resn50_dropout(self):
        self.encoder_mod1.res_n50_enc.layer3[2].dropout = False
        self.encoder_mod2.res_n50_enc.layer3[2].dropout = False

    def forward(self, mod1, mod2=None):
        """Forward pass
        In the case of AdapNet++, only 1 modality is used (either the RGB-image, or the Depth-image). 
        With 'AdapNet++ with fusion architecture' two modalities are used (both the RGB and Depth).
        :param mod1: modality 1
        :param mod2: modality 2
        :return: final output and auxiliary output 1 and 2
        """
        if self.stage == 1:
            m1_x, skip2, skip1 = self.encoder_mod1(mod1)
            m1_x = self.eASPP(m1_x)

        else:
            m1_x, skip2, skip1 = self.encoder_mod1(mod1)
            m2_x, m2_s2, m2_s1 = self.encoder_mod2(mod2)

            m1_x = self.eASPP_mod1(m1_x)
            m2_x = self.eASPP_mod2(m2_x)

            skip2 = self.ssma_s2(skip2, m2_s2)
            skip1 = self.ssma_s1(skip1, m2_s1)
            m1_x = self.ssma_res(m1_x, m2_x)

        aux1, aux2, res = self.decoder(m1_x, skip1, skip2)

        return [res, aux1, aux2]