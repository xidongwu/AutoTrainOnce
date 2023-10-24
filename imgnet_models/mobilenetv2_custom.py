"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math
from .gate_function import virtual_gate

__all__ = ['mobilenet_v2', 'my_mobilenet_v2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, cfg=None, gate_flag=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        #hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if cfg is None:
            hidden_dim = int(round(inp * expand_ratio))
        else:
            hidden_dim = cfg

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            if gate_flag:
                #self.gate = virtual_gate(hidden_dim)
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # dw
                    virtual_gate(hidden_dim),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:

                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # dw

                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
        self.gate_flag = gate_flag

    def forward(self, x):
        if not self.gate_flag:
            if self.identity:
                return x + self.conv(x)
            else:
                return self.conv(x)
        else:
            if self.identity:
                out = self.conv.__getitem__(0)(x)
                out = self.conv.__getitem__(1)(out)
                out = self.conv.__getitem__(2)(out)

                out = self.conv.__getitem__(3)(out)

                out = self.conv.__getitem__(4)(out)
                out = self.conv.__getitem__(5)(out)
                out = self.conv.__getitem__(6)(out)

                out = self.conv.__getitem__(3)(out)

                out = self.conv.__getitem__(7)(out)
                out = self.conv.__getitem__(8)(out)
                return out+x
            else:
                out = self.conv.__getitem__(0)(x)
                out = self.conv.__getitem__(1)(out)
                out = self.conv.__getitem__(2)(out)

                out = self.conv.__getitem__(3)(out)

                out = self.conv.__getitem__(4)(out)
                out = self.conv.__getitem__(5)(out)
                out = self.conv.__getitem__(6)(out)

                out = self.conv.__getitem__(3)(out)

                out = self.conv.__getitem__(7)(out)
                out = self.conv.__getitem__(8)(out)
                return out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0 ,custom_cfg=False, gate_flag=False,cfgs=None):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks

        if custom_cfg:
            self.cfgs = cfgs
        else:
            self.cfgs = [
                # t, c, n, s
                [1,  16, 1, 1],
                [6,  24, 2, 2],
                [6,  32, 3, 2],
                [6,  64, 4, 2],
                [6,  96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # building first layer
        #input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)

        if custom_cfg:
            input_channel = cfgs[0][-1][0]

        else:
            input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)

        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual

        if custom_cfg is False:
            for t, c, n, s in self.cfgs:
                output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
                for i in range(n):
                    if t == 1:
                        layers.append(
                            block(input_channel, output_channel, s if i == 0 else 1, expand_ratio=t, gate_flag=False))
                    else:
                        layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, gate_flag=gate_flag))
                    input_channel = output_channel
        else:
            for t, c, n, s, p_list in self.cfgs:
                #strides = [stride] + [1]*(num_blocks-1)
                output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
                for i in range(n):
                    stride = s if i == 0 else 1
                    # features.append(block(input_channel, output_channel, stride, expand_ratio=t, cfg = p_list[i], gate_flag=False))
                    # input_channel = output_channel
                    if t == 1:
                        layers.append(
                            block(input_channel, output_channel, stride, expand_ratio=t, gate_flag=False))
                    else:
                        layers.append(block(input_channel, output_channel, stride, expand_ratio=t, cfg = p_list[i], gate_flag=gate_flag))
                    input_channel = output_channel

        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    def count_structure(self):
        structure = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                structure.append(m.width)
        self.structure = structure
        return sum(structure), structure

    def set_vritual_gate(self, arch_vector):
        i = 0
        start = 0
        for m in self.modules():
            if isinstance(m, virtual_gate):
                end = start + self.structure[i]
                m.set_structure_value(arch_vector.squeeze()[start:end])
                start = end

                i+=1

    def get_weights(self):
        modules = list(self.modules())
        weights_list = []
        truncate_module = []
        for layer_id in range(len(modules)):
            m0 = modules[layer_id]
            if isinstance(m0, nn.BatchNorm2d) or isinstance(m0, nn.Conv2d) or isinstance(m0, nn.Linear) or isinstance(
                    m0, nn.ReLU6) or isinstance(m0, virtual_gate):
                truncate_module.append(m0)

        for layer_id in range(len(truncate_module)):
            m = truncate_module[layer_id]
            current_list = []
            # print(m)
            # if layer_id + 3 <= len(modules):
            if isinstance(m, virtual_gate):
                # print(m)

                up_weight = truncate_module[layer_id - 3].weight
                middle_weight = truncate_module[layer_id + 1].weight
                low_weight = truncate_module[layer_id + 4].weight

                current_list.append(up_weight), current_list.append(middle_weight), current_list.append(low_weight)
                weights_list.append(current_list)

        return weights_list

    def reset_gates(self):
        for m in self.modules():
            if isinstance(m, virtual_gate):
                m.reset_value()

    def project_wegit(self, masks, lmd, lr):
        self.lmd, self.lr = lmd, lr
        # print("self.lam * ratio * self.lr", self.lmd, self.lr)

        N_t = 0
        for itm in masks:
            N_t += (1 - itm).sum()
        # gap = 3 #2 if self.block_string == 'Bottleneck' else 
        modules = list(self.modules())
        weights_list = []
        vg_idx = 0


        for layer_id in range(len(modules)):
            m = modules[layer_id]
            if isinstance(m, virtual_gate):
                ratio = (1 - masks[vg_idx]).sum() / N_t
                if ratio == 0:
                    vg_idx += 1
                    continue

                m_out = (masks[vg_idx] == 0)
                vg_idx += 1
                ## calculate group norm
                w_norm = (modules[layer_id - 3].weight.data[m_out]).pow(2).sum((1,2,3))
                w_norm += (modules[layer_id - 2].weight.data[m_out]).pow(2)
                w_norm += (modules[layer_id - 2].bias.data[m_out]).pow(2)

                w_norm += (modules[layer_id + 1].weight.data[m_out]).pow(2).sum((1,2,3))
                w_norm += (modules[layer_id + 2].weight.data[m_out]).pow(2)
                w_norm += (modules[layer_id + 2].bias.data[m_out]).pow(2)

                w_norm = w_norm.add(1e-8).pow(1/2.)
                # print("w_norm shape", w_norm.size())

                modules[layer_id - 3].weight.copy_(self.groupproximal(modules[layer_id - 3].weight.data, m_out, ratio, w_norm))
                modules[layer_id - 2].weight.copy_(self.groupproximal(modules[layer_id - 2].weight.data, m_out, ratio, w_norm))
                modules[layer_id - 2].bias.copy_(  self.groupproximal(modules[layer_id - 2].bias.data, m_out, ratio, w_norm))

                modules[layer_id + 1].weight.copy_(self.groupproximal(modules[layer_id + 1].weight.data, m_out, ratio, w_norm))
                modules[layer_id + 2].weight.copy_(self.groupproximal(modules[layer_id + 2].weight.data, m_out, ratio, w_norm))
                modules[layer_id + 2].bias.copy_(  self.groupproximal(modules[layer_id + 2].bias.data, m_out, ratio, w_norm))


    def groupproximal(self, weight, m_out, ratio, w_norm):
        # #######  Test ######
        # weight[m_out] = 0
        # return weight
        ####################

        # print(weight.size(), weight[m_out].size())
        dimlen = len(weight.size())
        while dimlen > 1:
            w_norm = w_norm.unsqueeze(1)
            dimlen -= 1

        weight[m_out] = weight[m_out] / w_norm 
        tmp = - self.lmd * ratio * self.lr + w_norm
        tmp[tmp < 0] = 0 # tmp = max(0, - self.lmd * ratio * self.lr + w_norm)

        weight[m_out] = weight[m_out] * tmp
        return weight

# def mobilenet_v2(**kwargs):
#     """
#     Constructs a MobileNet V2 model
#     """
#     return MobileNetV2(**kwargs)

def my_mobilenet_v2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs, gate_flag=True)