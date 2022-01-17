import torch.nn as nn
import functools
import torch
import functools


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adin(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class PATBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False):
        super(PATBlock, self).__init__()
        self.conv_block_stream1 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias,
                                                        cal_att=False)
        self.conv_block_stream2 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias,
                                                        cal_att=True, cated_stream2=cated_stream2)

        self.query_conv = nn.Conv2d(in_channels=256, out_channels=256//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=256, out_channels=256//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

        self.stream1_down = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            stride=2)
        self.stream1_up = nn.ConvTranspose2d(
            dim,
            dim,
            kernel_size=2,
            stride=2)

        self.stream2_down = nn.Conv2d(
            dim * 2,
            dim,
            kernel_size=1,
            stride=2)

        self.stream2_up = nn.ConvTranspose2d(
            dim * 2,
            dim,
            kernel_size=2,
            stride=2)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False,
                         cal_att=False):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if cated_stream2:
            conv_block += [nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim * 2),
                           nn.ReLU(True)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim),
                           nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        if cal_att:
            if cated_stream2:
                conv_block += [nn.Conv2d(dim * 2, dim, kernel_size=3, padding=p, bias=use_bias)]
            else:
                conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x1, x2):
        x1_out = self.conv_block_stream1(x1)
        x2_out = self.conv_block_stream2(x2)
        # Update Image Branch
        m_batchsize, C, height, width = x1_out.size()
        proj_query = self.query_conv(x1_out).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x2_out).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x1_out).view(m_batchsize, -1, width * height)
        x1_out_1 = torch.bmm(proj_value, attention.permute(0, 2, 1))
        x1_out_1 = x1_out_1.view(m_batchsize, C, height, width)
        #AdaIN
        x1_out_1 = adin(x1_out_1,x1_out)
        x1_out = x1_out + x1_out_1  # connection
        x2_out = torch.cat((x2_out, x1_out), 1)
        # up and down
        x1_out_down = self.stream1_down(x1_out)
        x1_out_up = self.stream1_up(x1_out)
        x2_out_down = self.stream2_down(x2_out)
        x2_out_up = self.stream2_up(x2_out)
        return x1_out, x2_out, x1_out_down, x1_out_up, x2_out_down, x2_out_up,


class PATNModel(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_downsampling=2):
        assert (n_blocks >= 0 and type(input_nc) == list)
        super(PATNModel, self).__init__()
        self.input_nc_s1 = input_nc[0]
        self.input_nc_s2 = input_nc[1]
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # down_sample
        model_stream1_down = [nn.ReflectionPad2d(3),
                              nn.Conv2d(self.input_nc_s1, ngf, kernel_size=7, padding=0,
                                        bias=use_bias),
                              norm_layer(ngf),
                              nn.ReLU(True)]

        model_stream2_down = [nn.ReflectionPad2d(3),
                              nn.Conv2d(self.input_nc_s2, ngf, kernel_size=7, padding=0,
                                        bias=use_bias),
                              norm_layer(ngf),
                              nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            model_stream1_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                             stride=2, padding=1, bias=use_bias),
                                   norm_layer(ngf * mult * 2),
                                   nn.ReLU(True)]
            model_stream2_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                             stride=2, padding=1, bias=use_bias),
                                   norm_layer(ngf * mult * 2),
                                   nn.ReLU(True)]

        # att_block in place of res_block
        n_blocks = 6
        n_blocks_level1 = n_blocks
        self.level1_nb = n_blocks_level1
        mult = 2 ** n_downsampling
        cated_stream2_level1 = [True for i in range(n_blocks_level1)]
        cated_stream2_level1[0] = False

        attBlock_level1 = nn.ModuleList()
        for i in range(n_blocks_level1):
            attBlock_level1.append(
                PATBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                         use_bias=use_bias, cated_stream2=cated_stream2_level1[i]))

        n_blocks_level2 = n_blocks_level1 - 2
        self.level2_nb = n_blocks_level2
        cated_stream2_level2 = [True for i in range(n_blocks_level2)]
        cated_stream2_level2[0] = False
        attBlock_level2 = nn.ModuleList()
        for i in range(n_blocks_level2):
            attBlock_level2.append(
                PATBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                         use_bias=use_bias, cated_stream2=cated_stream2_level2[i]))

        n_blocks_level3 = n_blocks_level2 - 2
        self.level3_nb = n_blocks_level3
        cated_stream2_level3 = [True for i in range(n_blocks_level3)]
        cated_stream2_level3[0] = False
        attBlock_level3 = nn.ModuleList()
        for i in range(n_blocks_level3):
            attBlock_level3.append(
                PATBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                         use_bias=use_bias, cated_stream2=cated_stream2_level3[i]))

        # 1x1 卷积
        self.conv1_12_1 = nn.Conv2d(512, 256, 1, 1)
        self.conv1_12_2 = nn.Conv2d(768, 512, 1, 1)
        self.conv1_21_1 = nn.Conv2d(512, 256, 1, 1)
        self.conv1_21_2 = nn.Conv2d(768, 512, 1, 1)

        self.conv1_13_1 = nn.Conv2d(512, 256, 1, 1)
        self.conv1_13_2 = nn.Conv2d(768, 512, 1, 1)
        self.conv1_22_1 = nn.Conv2d(768, 256, 1, 1)
        self.conv1_22_2 = nn.Conv2d(1024, 512, 1, 1)
        self.conv1_31_1 = nn.Conv2d(512, 256, 1, 1)
        self.conv1_31_2 = nn.Conv2d(768, 512, 1, 1)

        self.conv1_14_1 = nn.Conv2d(512, 256, 1, 1)
        self.conv1_14_2 = nn.Conv2d(768, 512, 1, 1)
        self.conv1_23_1 = nn.Conv2d(768, 256, 1, 1)
        self.conv1_23_2 = nn.Conv2d(1024, 512, 1, 1)

        self.conv1_15_1 = nn.Conv2d(512, 256, 1, 1)
        self.conv1_15_2 = nn.Conv2d(768, 512, 1, 1)

        # up_sample
        model_stream1_up = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model_stream1_up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1,
                                                    bias=use_bias),
                                 norm_layer(int(ngf * mult / 2)),
                                 nn.ReLU(True)]

        model_stream1_up += [nn.ReflectionPad2d(3)]
        model_stream1_up += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_stream1_up += [nn.Tanh()]

        self.stream1_down = nn.Sequential(*model_stream1_down)
        self.stream2_down = nn.Sequential(*model_stream2_down)
        self.att_level1 = attBlock_level1
        self.att_level2 = attBlock_level2
        self.att_level3 = attBlock_level3

        self.stream1_up = nn.Sequential(*model_stream1_up)

    def forward(self, input):
        x1, x2 = input
        x1 = self.stream1_down(x1)
        x2 = self.stream2_down(x2)

        # att_block
        x11_1, x11_2, x20_1, _, x20_2, _ = self.att_level1[0](x1, x2)
        x12_1, x12_2, x12_1_down, _, x12_2_down, _ = self.att_level1[1](x11_1, x11_2)
        x21_1, x21_2, x21_1_down, x21_1_up, x21_2_down, x21_2_up = self.att_level2[0](x20_1, x20_2)
        x12_1 = torch.cat((x12_1, x21_1_up), 1)
        x12_1 = self.conv1_12_1(x12_1)
        x12_2 = torch.cat((x12_2, x21_2_up), 1)
        x12_2 = self.conv1_12_2(x12_2)
        x21_1 = torch.cat((x21_1, x12_1_down), 1)
        x21_1 = self.conv1_21_1(x21_1)
        x21_2 = torch.cat((x21_2, x12_2_down), 1)
        x21_2 = self.conv1_21_2(x21_2)
        x13_1, x13_2, x13_1_down, _, x13_2_down, _ = self.att_level1[2](x12_1, x12_2)
        x22_1, x22_2, x22_1_down, x22_1_up, x22_2_down, x22_2_up = self.att_level2[1](x21_1, x21_2)
        x31_1, x31_2, _, x31_1_up, _, x31_2_up = self.att_level3[0](x21_1_down, x21_2_down)
        x13_1 = torch.cat((x13_1, x22_1_up), 1)
        x13_1 = self.conv1_13_1(x13_1)
        x13_2 = torch.cat((x13_2, x22_2_up), 1)
        x13_2 = self.conv1_13_2(x13_2)
        x22_1 = torch.cat((x22_1, x13_1_down, x31_1_up), 1)
        x22_1 = self.conv1_22_1(x22_1)
        x22_2 = torch.cat((x22_2, x13_2_down, x31_2_up), 1)
        x22_2 = self.conv1_22_2(x22_2)
        x31_1 = torch.cat((x31_1, x22_1_down), 1)
        x31_1 = self.conv1_31_1(x31_1)
        x31_2 = torch.cat((x31_2, x22_2_down), 1)
        x31_2 = self.conv1_31_2(x31_2)
        x14_1, x14_2, x14_1_down, _, x14_2_down, _ = self.att_level1[3](x13_1, x13_2)
        x23_1, x23_2, _, x23_1_up, _, x23_2_up = self.att_level2[2](x22_1, x22_2)
        _, _, _, x32_1_up, _, x32_2_up = self.att_level3[1](x31_1, x31_2)
        x14_1 = torch.cat((x14_1, x23_1_up), 1)
        x14_1 = self.conv1_14_1(x14_1)
        x14_2 = torch.cat((x14_2, x23_2_up), 1)
        x14_2 = self.conv1_14_2(x14_2)
        x23_1 = torch.cat((x23_1, x14_1_down, x32_1_up), 1)
        x23_1 = self.conv1_23_1(x23_1)
        x23_2 = torch.cat((x23_2, x14_2_down, x31_2_up), 1)
        x23_2 = self.conv1_23_2(x23_2)
        x15_1, x15_2, _, _, _, _ = self.att_level1[4](x14_1, x14_2)
        _, _, _, x24_1_up, _, x24_2_up = self.att_level2[3](x23_1, x23_2)
        x15_1 = torch.cat((x15_1, x24_1_up), 1)
        x15_1 = self.conv1_15_1(x15_1)
        x15_2 = torch.cat((x15_2, x24_2_up), 1)
        x15_2 = self.conv1_15_2(x15_2)
        x16_1, x16_2, _, _, _, _ = self.att_level1[4](x15_1, x15_2)
        x_out = self.stream1_up(x16_1)
        return x_out


class PATNetwork(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', n_downsampling=2):
        super(PATNetwork, self).__init__()
        assert type(input_nc) == list and len(input_nc) == 2, 'The AttModule take input_nc in format of list only!!'
        self.gpu_ids = gpu_ids
        self.model = PATNModel(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, gpu_ids, padding_type,
                               n_downsampling=n_downsampling)

    def forward(self, input):
        if self.gpu_ids and isinstance(input[0].data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)