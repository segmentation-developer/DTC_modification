import torch
from torch import nn
import torch.nn.functional as F

"""
Differences with V-Net
Adding nn.Tanh in the end of the conv. to make the outputs in [-1, 1].
"""

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class DilationConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):      #256,256
        super(DilationConvBlock, self).__init__()

        ops1 = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops1.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1, dilation=1))
            if normalization == 'batchnorm':
                ops1.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops1.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops1.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops1.append(nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(*ops1)

        ops2 = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops2.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=2, dilation=2))
            if normalization == 'batchnorm':
                ops2.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops2.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops2.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops2.append(nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(*ops2)

        ops3 = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops3.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=3, dilation=3))
            if normalization == 'batchnorm':
                ops3.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops3.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops3.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops3.append(nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(*ops3)

        '''
        self.conv1 = nn.Sequential(nn.Conv3d(n_filters_in, n_filters_out, 3, padding=1, dilation=1),
                                   nn.BatchNorm3d(n_filters_out),
                                   nn.ReLU(inplace=True)) # kernel :3,3,3

        self.conv2 = nn.Sequential(nn.Conv3d(n_filters_in, n_filters_out, 3, padding=2, dilation=2),
                                   nn.BatchNorm3d(n_filters_out),
                                   nn.ReLU(inplace=True) ) # kernel :5,5,5


        self.conv3 = nn.Sequential(nn.Conv3d(n_filters_in, n_filters_out, 3, padding=3, dilation=3),
                                   nn.BatchNorm3d(n_filters_out),
                                   nn.ReLU(inplace=True) ) # kernel :7,7,7
                                   '''



        self.conv_final = nn.Conv3d(n_filters_in * 4, n_filters_out, 1, padding=0)

    def forward(self, x):
        out = torch.cat([x, self.conv1(x), self.conv2(x), self.conv3(x)], 1)
        out = self.conv_final(out)
        '''
        out = torch.cat([x, self.conv1(x), self.conv2(x), self.conv3(x)], 1)
        out = self.conv_final(out) '''
        return out


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='none', has_dropout=True, has_residual=False):
        super(VNet, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.Dil_five = DilationConvBlock(3,n_filters * 16, n_filters * 16, normalization=normalization)
        #self.block_five = convBlock(2, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(3, n_filters * 16, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 8, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 4, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(1, n_filters * 2 , n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv2 = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.tanh = nn.Tanh()

        '''
        self.Dil_one = DilationConvBlock( n_channels, n_filters, normalization=normalization)
        self.block_one = convBlock(1, n_filters * 2, n_filters , normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.Dil_two = DilationConvBlock(n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two = convBlock(1, n_filters * 4, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.Dil_three = DilationConvBlock(n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three = convBlock(2, n_filters * 8, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.Dil_four = DilationConvBlock(n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four = convBlock(2, n_filters * 16, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.Dil_five = DilationConvBlock(n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five = convBlock(2, n_filters * 32, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.Dil_six = DilationConvBlock(n_filters * 16, n_filters * 8, normalization=normalization)
        self.block_six = convBlock(2, n_filters * 16, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.Dil_seven = DilationConvBlock(n_filters * 8, n_filters * 4, normalization=normalization)
        self.block_seven = convBlock(2, n_filters * 8, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.Dil_eight = DilationConvBlock(n_filters * 4, n_filters * 2, normalization=normalization)
        self.block_eight = convBlock(1, n_filters * 4, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.Dil_nine = DilationConvBlock(n_filters * 2, n_filters, normalization=normalization)
        self.block_nine = convBlock(1, n_filters * 2, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv2 = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.tanh = nn.Tanh()
        '''

        #self.one = convBlock(3, n_filters, n_filters, normalization=normalization)
        #self.two = convBlock(3, n_filters * 2, n_filters * 2, normalization=normalization)
        #self.three = convBlock(2, n_filters * 4, n_filters * 4, normalization=normalization)
        #self.four = convBlock(1, n_filters * 8, n_filters * 8, normalization=normalization)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.Dil_five(x4_dw)
        #x5 = self.block_five(x5)

        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res

    def decoder(self, features):
        x1 = features[0]  # 4,16,112,,112,80
        x2 = features[1]  # 4,32,56,56,40
        x3 = features[2]  # 4,64,28,28,20
        x4 = features[3]  # 4,128,14,14,10
        x5 = features[4]  # 4,256,7,7,5

        #x4_conv = self.four(x4)
        #x3_conv = self.three(x3)
        #x2_conv = self.two(x2)
        #x1_conv = self.one(x1)

        x5_up = self.block_five_up(x5)  # 4,128,14,14,10    #upsample -> BN -> Relu
        # x5_up = x5_up + x4
        x5_up = torch.cat([x5_up, x4], 1)  ##4,256,14,14,10

        x6 = self.block_six(x5_up)  # 4,128,14,14,10   #conv -> BN -> Relu (ch 1/2)
        # x6 = x6 + x4_conv  # enc가 아닌 residual = VNet2020
        x6_up = self.block_six_up(x6)  # 4,64,28,28,20    #upsample -> BN -> Relu  (ch 1/2)
        # x6_up = x6_up + x3
        x6_up = torch.cat([x6_up, x3], 1)  # 4,128,28,28,20

        x7 = self.block_seven(x6_up)  # 4,64,28,28,20
        # x7 = x7 + x3
        x7_up = self.block_seven_up(x7)  # 4,32,56,56,40
        # x7_up = x7_up + x2
        x7_up = torch.cat([x7_up, x2], 1)  # 4,64,56,56,40   #64

        x8 = self.block_eight(x7_up)  # 4,32,56,56,40
        # x8 = x8 + x2
        x8_up = self.block_eight_up(x8)  # 4,16,112,,112,80
        # x8_up = x8_up + x1
        x8_up = torch.cat([x8_up, x1], 1)  # 4,32,112,,112,80        #32

        x9 = self.block_nine(x8_up)  # 4,16,112,,112,80
        # x9 = F.dropout3d(x9, p=0.5, training=True)

        if self.has_dropout:
            x9 = self.dropout(x9)

        out = self.out_conv(x9)
        out_tanh = self.tanh(out)

        out_seg = self.out_conv2(x9)
        return out_tanh, out_seg


    def forward(self, input, turnoff_drop=True):

        features = self.encoder(input)
        out_tanh, out_seg = self.decoder(features)

        return out_tanh, out_seg

    # def __init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             m.weight.data.fill_(1)

if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format
    model = VNet(n_channels=1, n_classes=2)
    input = torch.randn(4, 1, 112, 112, 80)
    flops, params = profile(model, inputs=(input,))
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)
    print("VNet have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    # import ipdb; ipdb.set_trace()