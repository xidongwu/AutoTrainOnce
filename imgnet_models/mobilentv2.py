from torch import nn
from .imgnet_utils import load_state_dict_from_url
#from .gate_function import soft_gate, custom_STE
from .gate_function import virtual_gate
import torch.nn.functional as F

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


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


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__()
        #if use_gate is False:
        super(ConvBNReLU, self).__init__(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU6(inplace=True)
        )
    # else:
    #     super(ConvBNReLU, self).__init__(
    #         soft_gate(in_planes),
    #         nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
    #         nn.BatchNorm2d(out_planes),
    #         nn.ReLU6(inplace=True),
    #         soft_gate(out_planes)
    #     )
    #     self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False)
    #     self.bn = nn.BatchNorm2d(out_planes)
    #     self.relu6 = nn.ReLU6()
    #     self.use_gate = use_gate
    #     if use_gate:
    #         self.gate = soft_gate(in_planes)
    # def forward(self, input):
    #     if self.use_gate:
    #         input = self.gate(input)
    #     out = self.conv(input)
    #     out = self.bn(out)
    #     out = self.relu6(out)
    #     return out




class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, cfg=None, gate_flag=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        #hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        if cfg is None:
            hidden_dim = int(round(inp * expand_ratio))
        else:
            hidden_dim = cfg

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))


        if gate_flag is False:
            layers.extend([
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ])
        else:
            layers.extend([
                virtual_gate(hidden_dim),
                # dw
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            ])
        #print(len(layers))
        self.conv = nn.Sequential(*layers)
        self.gate_flag = gate_flag
    def forward(self, x):
        if self.gate_flag is False:
            if self.use_res_connect:
                # print(self.conv)
                # print(x.size())
                return x + self.conv(x)
            else:
                return self.conv(x)
        else:
            if self.use_res_connect:
                out = self.conv.__getitem__(0)(x)
                out = self.conv.__getitem__(1)(out)
                out = self.conv.__getitem__(2)(out)
                out = self.conv.__getitem__(1)(out)
                out = self.conv.__getitem__(3)(out)
                out = self.conv.__getitem__(4)(out)
                return x + out
            else:
                out = self.conv.__getitem__(0)(x)
                out = self.conv.__getitem__(1)(out)
                out = self.conv.__getitem__(2)(out)
                out = self.conv.__getitem__(1)(out)
                out = self.conv.__getitem__(3)(out)
                out = self.conv.__getitem__(4)(out)
                return out


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=1, custom_cfg=False, gate_flag=True, last_flag=False):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual

        if custom_cfg:
            input_channel = inverted_residual_setting[0][-1][0]
            if last_flag:
                print(inverted_residual_setting[-1])
                custom_last_channel = inverted_residual_setting[-1]
                #del inverted_residual_setting[-1]
                inverted_residual_setting = inverted_residual_setting[:-1]
                #inverted_residual_setting.pop()
        else:
            input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) > 5:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.gate_flag = gate_flag
        self.last_flag = last_flag
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        if custom_cfg is False:
            for t, c, n, s in inverted_residual_setting:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    if t == 1:
                        features.append(
                            block(input_channel, output_channel, stride, expand_ratio=t, gate_flag=False))
                    else:
                        features.append(block(input_channel, output_channel, stride, expand_ratio=t, gate_flag=gate_flag))
                    input_channel = output_channel
        else:
            for t, c, n, s, p_list in inverted_residual_setting:
                #strides = [stride] + [1]*(num_blocks-1)
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    # features.append(block(input_channel, output_channel, stride, expand_ratio=t, cfg = p_list[i], gate_flag=False))
                    # input_channel = output_channel
                    if t == 1:
                        features.append(
                            block(input_channel, output_channel, stride, expand_ratio=t, gate_flag=False))
                    else:
                        features.append(block(input_channel, output_channel, stride, expand_ratio=t, cfg = p_list[i], gate_flag=gate_flag))
                    input_channel = output_channel
        # building last several layers
        if custom_cfg and self.last_flag:
            print(custom_last_channel)
            features.append(ConvBNReLU(input_channel, custom_last_channel, kernel_size=1))
            # make it nn.Sequential
            self.features = nn.Sequential(*features)

            # building classifier
            if gate_flag:
                self.final_gate = virtual_gate(custom_last_channel)
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(custom_last_channel, num_classes),
            )


        else:

            features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
            # make it nn.Sequential
            self.features = nn.Sequential(*features)
            if gate_flag and self.last_flag:
                self.final_gate = virtual_gate(self.last_channel)
            # building classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes),
            )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        #x = x.mean([2, 3])
        x = x.view(x.size(0),x.size(1),-1)
        x = x.mean(dim=-1)
        if self.gate_flag and self.last_flag:
            x = self.final_gate(x)
        x = self.classifier(x)
        return x

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




def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(gate_flag=False, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def my_mobilenet_v2(pretrained=False, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(gate_flag=True, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model