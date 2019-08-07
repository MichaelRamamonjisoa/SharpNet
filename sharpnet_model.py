import torch.nn as nn
import torch
from resnet import resnet50, conv3x3, conv1x1
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, inplanes, in_channels=[1024, 512, 256, 64, 16],
                 out_channels=1, layers_nums=[2, 2, 2, 2, 2],
                 kernel_size=3,
                 bias=False, normalize_output=False,
                 interpolation='bilinear',
                 out_activation='ReLU'):

        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers_nums = layers_nums
        self.kernel_size = kernel_size
        self.bias = bias
        self.out_activation = out_activation
        self.inplanes = inplanes
        self.normalize_output = normalize_output
        self.interpolation = interpolation

        decoder_layers = self._make_decoder(in_channels, out_channels, layers_nums, kernel_size, bias, out_activation)

        self.upconv4, self.upconv3, self.upconv2, self.upconv1, self.upconv0, self.conv_out = tuple(decoder_layers)

    def _make_decoder(self, in_channels, out_channels, in_layers_nums, kernel_size=3, bias=True,
                      out_activation='ReLU'):

        layers_list = []
        for i, convs in enumerate(in_layers_nums):
            layers = []
            for _ in range(convs):
                layers.extend([
                    nn.Conv2d(self.inplanes, in_channels[i], kernel_size=kernel_size,
                              stride=1, padding=1, bias=bias, dilation=1),
                    nn.BatchNorm2d(in_channels[i]),
                    nn.ReLU(inplace=True)])
                self.inplanes = in_channels[i]

            if i != len(in_layers_nums) - 1:
                self.inplanes *= 2

            layers_list.append(nn.Sequential(*layers))

        # Output layer
        layers = []
        layers.extend([nn.Conv2d(self.inplanes, out_channels, kernel_size=kernel_size,
                                 stride=1,
                                 padding=1, bias=bias, dilation=1),
                       nn.BatchNorm2d(out_channels)
                       ])

        if out_activation == 'ReLU':
            layers.extend([nn.ReLU(inplace=True)])

        elif out_activation == 'Sigmoid':
            layers.extend([nn.Sigmoid()])

        elif out_activation is 'Tanh':
            layers.extend([nn.Tanh()])

        layers_list.append(nn.Sequential(*layers))

        return layers_list

    def freeze(self):
        for m in self.modules():
            m.eval()

    def forward(self, resized_resnet_outputs, input_image):

        # upconv4
        if self.interpolation == 'bilinear':
            resized_resnet_outputs[4] = F.interpolate(resized_resnet_outputs[4],
                                                      size=tuple(resized_resnet_outputs[3].shape[2:]),
                                                      mode='bilinear',
                                                      align_corners=True)
        else:
            resized_resnet_outputs[4] = F.interpolate(resized_resnet_outputs[4],
                                                      size=tuple(resized_resnet_outputs[3].shape[2:]),
                                                      mode=self.interpolation)
        x_out = self.upconv4(resized_resnet_outputs[4])
        x_out = torch.cat((x_out, resized_resnet_outputs[3]), 1)

        # upconv3
        x_out = self.upconv3(x_out)
        if self.interpolation == 'bilinear':
            x_out = F.interpolate(x_out, size=tuple(resized_resnet_outputs[2].shape[2:]),
                                  mode='bilinear',
                                  align_corners=True)
        else:
            x_out = F.interpolate(x_out, size=tuple(resized_resnet_outputs[2].shape[2:]),
                                  mode=self.interpolation)
        x_out = torch.cat((x_out, resized_resnet_outputs[2]), 1)

        # upconv2
        x_out = self.upconv2(x_out)
        if self.interpolation == 'bilinear':
            x_out = F.interpolate(x_out, size=tuple(resized_resnet_outputs[1].shape[2:]),
                                  mode='bilinear',
                                  align_corners=True)
        else:
            x_out = F.interpolate(x_out, size=tuple(resized_resnet_outputs[1].shape[2:]),
                                  mode=self.interpolation)
        x_out = torch.cat((x_out, resized_resnet_outputs[1]), 1)

        # upconv1
        x_out = self.upconv1(x_out)
        if self.interpolation == 'bilinear':
            x_out = F.interpolate(x_out, size=tuple(resized_resnet_outputs[0].shape[2:]),
                                  mode='bilinear',
                                  align_corners=True)
        else:
            x_out = F.interpolate(x_out, size=tuple(resized_resnet_outputs[0].shape[2:]),
                                  mode=self.interpolation)
        x_out = torch.cat((x_out, resized_resnet_outputs[0]), 1)

        # upconv0
        x_out = self.upconv0(x_out)
        if self.interpolation == 'bilinear':
            x_out = F.interpolate(x_out, size=tuple(input_image.shape[2:]),
                                  mode='bilinear',
                                  align_corners=True)
        else:
            x_out = F.interpolate(x_out, size=tuple(input_image.shape[2:]),
                                  mode=self.interpolation)
        x_out = self.conv_out(x_out)

        if self.normalize_output:
            x_out = F.normalize(x_out, p=2, dim=1)

        return x_out


class SharpNet(nn.Module):
    def __init__(self, block, layers_encoder, layers_decoders,
                 use_normals=False,
                 use_depth=False,
                 use_boundary=False, bias_decoder=True):
        super(SharpNet, self).__init__()

        if use_normals:
            print('Deploying model with normals estimation')
        if use_depth:
            print('Deploying model with depth estimation')
        if use_boundary:
            print('Deploying model with boundary estimation')

        self.use_depth = use_depth
        self.use_normals = use_normals
        self.use_boundary = use_boundary

        # ResNet encoder
        self.conv1_img = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 3 (RGB) * 7x7 * 64
        self.bn1_img = nn.BatchNorm2d(64)
        self.relu_img = nn.ReLU(inplace=True)
        self.maxpool_img = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inplanes = 64  # now set to 64
        self.layer1_img = self._make_res_layer(block, 64, layers_encoder[0])
        self.layer2_img = self._make_res_layer(block, 128, layers_encoder[1], stride=2)
        self.layer3_img = self._make_res_layer(block, 256, layers_encoder[2], stride=2)
        self.layer4_img = self._make_res_layer(block, 512, layers_encoder[3], stride=1, dilation=2)

        if self.use_depth:
            layers_decoders[0] = int(layers_decoders[0] * 3)
            layers_decoders[1] = int(layers_decoders[1] * 3)
            self.depth_decoder = Decoder(self.inplanes,
                                         in_channels=[1024, 512, 256, 64, 16],
                                         out_channels=1,
                                         layers_nums=layers_decoders, kernel_size=3,
                                         bias=bias_decoder,
                                         interpolation='bilinear',
                                         out_activation='ReLU')

            layers_decoders[0] = int(layers_decoders[0] / 3)
            layers_decoders[1] = int(layers_decoders[1] / 3)

        if self.use_normals:
            layers_decoders[0] = int(layers_decoders[0] * 2)
            layers_decoders[1] = int(layers_decoders[1] * 2)
            self.normals_decoder = Decoder(self.inplanes,
                                           in_channels=[1024, 512, 256, 64, 16],
                                           out_channels=3,
                                           layers_nums=layers_decoders, kernel_size=3,
                                           bias=bias_decoder, normalize_output=True,
                                           interpolation='bilinear',
                                           out_activation='Tanh')

            layers_decoders[0] = int(layers_decoders[0] / 2)
            layers_decoders[1] = int(layers_decoders[1] / 2)

        if self.use_boundary:
            self.boundary_decoder = Decoder(self.inplanes,
                                            in_channels=[1024, 512, 256, 64, 16],
                                            out_channels=1,
                                            layers_nums=layers_decoders, kernel_size=3,
                                            bias=bias_decoder,
                                            interpolation='nearest',
                                            out_activation='Sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                m.train()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                m.eval()

    def _make_res_layer(self, block, planes, blocks, stride=1, expansion=4, dilation=1):
        downsample = None
        # either only first block of all ResBlock is downsampled to match x with the output dimension
        # or also if the block has in_planes = out_planes
        assert dilation == 1 or dilation % 2 == 0

        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * expansion, stride),
                nn.BatchNorm2d(planes * expansion),
            )

        layers = list([])
        # add first convblock: this is the only one with potential downsampling (stride != 1 or dilation != 1)
        layers.append(block(self.inplanes, planes, stride, downsample, expansion, dilation=(dilation, dilation)))
        # then add the others
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x_img):
        x_img_out = self.conv1_img(x_img)
        x_img_out = self.bn1_img(x_img_out)
        x_img_out = self.relu_img(x_img_out)
        x_img_out = self.maxpool_img(x_img_out)

        x1 = self.layer1_img(x_img_out)  # save res2 output for residual concatenation
        x2 = self.layer2_img(x1)
        x3 = self.layer3_img(x2)
        x4 = self.layer4_img(x3)

        if self.use_normals:
            x_normals = self.normals_decoder([x_img_out, x1, x2, x3, x4], x_img)
        else:
            x_normals = None

        x_lf = None
        x_depth = None
        x_mask = None
        x_boundary = None

        if self.use_depth:
            x_depth = self.depth_decoder([x_img_out, x1, x2, x3, x4], x_img)

        if self.use_boundary:
            x_boundary = self.boundary_decoder([x_img_out, x1, x2, x3, x4], x_img)

        return_list = [x_out for x_out in [x_mask, x_depth, x_lf, x_normals, x_boundary] if x_out is not None]
        if len(return_list) == 1:
            return return_list[0]
        else:
            return tuple(return_list)
