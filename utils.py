import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from imgnet_models.gate_function import custom_STE
from imgnet_models.gate_function import virtual_gate
from imgnet_models.mobilenetv3 import Hswish, Hsigmoid

import torchvision
from math import floor

def print_flops(hyper_net, args):
    hyper_net.eval()
    total_flops =  args.resource_constraint.get_flops(hyper_net.resource_output())
    print('+ Number of FLOPs: %.5fG'%(total_flops/1e9))

    # with torch.no_grad():
    #     hyper_net.eval()
    #     vector = hyper_net()
    #     masks = hyper_net.vector2mask(vector)

    # vector = hyper_net() # 有分数
    # if hasattr(net,'module'):
    #     net.module.set_vritual_gate(vector)
    # else:
    #     net.set_vritual_gate(vector)



DEFAULT_OPT_PARAMS = {
    "sgd": {
        "first_momentum": 0.0,
        "second_momentum": 0.0,
        "dampening": 0.0,
        "weight_decay": 0.0,
        "lmbda": 1e-3,
        "lmbda_amplify": 2,
        "hat_lmbda_coeff": 10
    }
    ,
    "adam": {
        "lr": 1e-3,
        "first_momentum": 0.9,
        "second_momentum": 0.999,
        "dampening": 0.0,
        "weight_decay": 0.0,
        "lmbda": 1e-2,
        "lmbda_amplify": 20,
        "hat_lmbda_coeff": 1e3
    }
    ,
    "adamw": {
        "lr": 1e-3,
        "first_momentum": 0.9,
        "second_momentum": 0.999,
        "dampening": 0.0,
        "weight_decay": 1e-2,
        "lmbda": 1e-2,
        "lmbda_amplify": 20,
        "hat_lmbda_coeff": 1e3
    }
}


class resource_constraint(nn.Module):
    def __init__(self, num_epoch, cut_off_epoch, p):
        super(resource_constraint, self).__init__()
        self.num_epoch = num_epoch
        self.cut_off_epoch = cut_off_epoch
        self.p = p

    def forward(self, input, epoch):
        overall_length = 0
        for i in range(len(input)):
            overall_length+= input[i].size(0)

        for i in range(len(input)):
            if i == 0:
                #cat_tensor = input[i]
                #cat_tensor = F.tanh(input[i].abs().pow(1 / 2))
                cat_tensor = custom_STE.apply(input[i], False)
            else:
                #current_value = F.tanh(input[i].abs().pow(1 / 2))
                current_value = custom_STE.apply(input[i], False)
                cat_tensor = torch.cat([cat_tensor, current_value])

        # if epoch<= self.cut_off_epoch:
        #     w = (epoch/self.cut_off_epoch)
        # else:
        #     w = 1
        #loss = w*torch.log(F.relu(cat_tensor.mean() - self.p) + 1)
        #loss = torch.abs(cat_tensor.sum()-int(self.p*cat_tensor.size(0)))
        loss = torch.abs(cat_tensor.mean() - (self.p))
        #return (1/cat_tensor.size(0))*loss
        return loss

class Flops_constraint(nn.Module):
    def __init__(self, p, kernel_size, out_size, group_size, size_inchannel, size_outchannel, in_channel=3):
        super(Flops_constraint, self).__init__()

        self.p = p
        self.k_size = kernel_size
        self.out_size = out_size
        self.g_size = group_size
        self.in_csize = size_inchannel
        self.out_csize = size_outchannel
        self.t_flops = self.init_total_flops()
        self.inc_1st = in_channel
    def init_total_flops(self):
        total_flops = 0
        for i in range(len(self.k_size)):
            total_flops+= self.k_size[i]*(self.in_csize[i]/self.g_size[i])*self.out_csize[i]*self.out_size[i]+3*self.out_csize[i]*self.out_size[i]

        print('+ Number of FLOPs: %.5fG'%(total_flops/1e9))
        return total_flops


    def forward(self, input):
        c_in = self.inc_1st
        sum_flops = 0
        #print(len(self.k_size))
        for i in range(len(input)):
            #print(i)

            current_tensor = custom_STE.apply(input[i], False)
            if i >0:
                c_in = custom_STE.apply(input[i-1], False).sum()
            c_out = current_tensor.sum()
            #sum_flops+=current_tensor.sum()
            sum_flops+= self.k_size[i]*(c_in/self.g_size[i])*c_out*self.out_size[i]+3*c_out*self.out_size[i]
        loss = torch.log(torch.abs(sum_flops/self.t_flops - (self.p))+ 1)
        #loss = torch.abs(sum_flops/self.t_flops - (self.p))**2
        return 2*loss

class Flops_constraint_resnet(nn.Module):
    def __init__(self, p, kernel_size, out_size, group_size, size_inchannel, size_outchannel, in_channel=3, w=8,HN=False,structure=None):
        super(Flops_constraint_resnet, self).__init__()

        self.p = p
        self.k_size = kernel_size
        self.out_size = out_size
        self.g_size = group_size
        self.in_csize = size_inchannel
        self.out_csize = size_outchannel
        self.t_flops = self.init_total_flops()
        self.inc_1st = in_channel
        self.weight = w
        self.HN = HN
        self.structure = structure
    def init_total_flops(self):
        total_flops = 0
        for i in range(len(self.k_size)):
            total_flops+= self.k_size[i]*(self.in_csize[i]/self.g_size[i])*self.out_csize[i]*self.out_size[i]+3*self.out_csize[i]*self.out_size[i]

        print('+ Number of FLOPs: %.5fG'%(total_flops/1e9))
        return total_flops

    def forward(self, input):
        c_in = self.inc_1st
        sum_flops = 0
        if self.HN:
            #self.h = self.input.register_hook(lambda grad: grad)
            arch_vector = []
            start = 0
            for i in range(len(self.structure)):
                end = start + self.structure[i]
                input = input.squeeze()
                # print(input[start:end].size())
                arch_vector.append(input[start:end])
                start = end

            length = len(arch_vector)
            #print(length)
        else:
            length = len(input)
        for i in range(length):
            if self.HN is False:
                current_tensor = custom_STE.apply(input[i], False)
            # if i >0:
            #     c_in = custom_STE.apply(input[i-1], False).sum()
            else:
                current_tensor = arch_vector[i]


            #current_tensor = custom_STE.apply(input[i], False)
            # if i >0:
            #     c_in = custom_STE.apply(input[i-1], False).sum()
            c_out = current_tensor.sum()
            #two layer as a group
            sum_flops+= self.k_size[2*i]*(self.in_csize[2*i]/self.g_size[2*i])*c_out*self.out_size[2*i]+3*c_out*self.out_size[2*i]
            sum_flops+= self.k_size[2*i+1]*(c_out/self.g_size[2*i+1])*self.out_csize[2*i+1]*self.out_size[2*i+1]+3*self.out_csize[2*i+1]*self.out_size[2*i+1]
        # loss = torch.log(torch.abs(sum_flops/self.t_flops - (self.p))+ 1)

        resource_ratio = (sum_flops / self.t_flops)
        if resource_ratio > self.p:
            # resource_ratio = (sum_flops / self.t_flops)
            abs_rv = torch.clamp(resource_ratio, min=self.p + 0.005)
            loss = torch.log((abs_rv / (self.p)))
        else:
            # resource_ratio = (sum_flops / self.t_flops)
            abs_rv = torch.clamp(resource_ratio, max=self.p - 0.005)
            loss = torch.log(((self.p) / abs_rv))

        #loss = torch.abs(sum_flops/self.t_flops - (self.p))**2
        return self.weight*loss


# def conv_hook(self, input, output):
#     batch_size, input_channels, input_height, input_width = input[0].size()
#     output_channels, output_height, output_width = output[0].size()
#
#     kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
#     bias_ops = 1 if self.bias is not None else 0
#
#     params = output_channels * (kernel_ops + bias_ops)
#     flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size
#
#     list_conv.append(flops)

class Flops_constraint_resnet_bb(nn.Module):
    def __init__(self, p, kernel_size, out_size, group_size, size_inchannel, size_outchannel, in_channel=3, w=10, HN=False, structure=None):
        super(Flops_constraint_resnet_bb, self).__init__()

        self.p = p

        self.k_size = kernel_size
        self.out_size = out_size
        self.g_size = group_size
        self.in_csize = size_inchannel
        self.out_csize = size_outchannel
        self.t_flops = self.init_total_flops() # total FLOPs
        self.inc_1st = in_channel
        self.num_gates = 2
        self.weight = w
        self.HN = HN   # True
        self.structure = structure
        print(self.p)

    def init_total_flops(self):
        total_flops = 0
        for i in range(len(self.k_size)):
            kernel_ops = self.k_size[i]*(self.in_csize[i]/self.g_size[i])
            total_flops+= kernel_ops*self.out_csize[i]*self.out_size[i] + 3*self.out_csize[i]*self.out_size[i]
            # if i == 0:
            #     print(total_flops)
        #print(i)
        print('+ Number of FLOPs: %.5fG'%(total_flops/1e9))
        return total_flops

    def get_flops(self, input):
        c_in = self.inc_1st
        sum_flops = 0
        #print(len(self.k_size))

        #print(len(input))
        if self.num_gates==2:
            i = 0
            if self.HN:
                # self.h = self.input.register_hook(lambda grad: grad)
                arch_vector = []
                start = 0
                for i in range(len(self.structure)):
                    end = start + self.structure[i]
                    input = input.squeeze()
                    #print(input[start:end].size())
                    arch_vector.append(input[start:end])
                    start = end
                    #print(len(input[start:end]))
                length = len(arch_vector)
                # print(length)
            else:
                length = len(input)
            i = 0
            for ind in range(0,length,2):

                if self.HN:
                    c_in = arch_vector[ind].sum() # calculate how many channels remaining
                    c_out = arch_vector[ind+1].sum()

                else:
                    current_tensor = custom_STE.apply(input[ind], False)

                    next_tensor = custom_STE.apply(input[ind+1], False)

                    c_in = current_tensor.sum()
                    c_out = next_tensor.sum()

                sum_flops+= self.k_size[3*i]*(self.in_csize[3*i]/self.g_size[3*i]) * c_in * self.out_size[3*i] + 3* c_in *self.out_size[3*i]
                sum_flops+= self.k_size[3*i+1]*(c_in/self.g_size[3*i+1])*c_out*self.out_size[3*i+1] + 3*c_out*self.out_size[3*i+1]
                sum_flops+= self.k_size[3*i+2]*(c_out/self.g_size[3*i+2])*self.out_csize[3*i+2]*self.out_size[3*i+2] + 3*self.out_csize[3*i+2]*self.out_size[3*i+2]
                i+=1

            #print(i)
            #print(3*i+2)
        # else:
        #     for i in range(0, len(input)):
        #         #print(i)

        #         current_tensor = custom_STE.apply(input[i], False)

        #         #next_tensor = custom_STE.apply(input[ind+1], False)
        #         # if i >0:
        #         #     c_in = custom_STE.apply(input[i-1], False).sum()

        #         channels = current_tensor.sum()

        #         #c_in = current_tensor.sum()
        #         #c_out = next_tensor.sum()

        #         #two layer as a group
        #         # sum_flops+= self.k_size[3*i]*(self.in_csize[3*i]/self.g_size[3*i])*c_in*self.out_size[3*i]+3*c_in*self.out_size[3*i]
        #         # sum_flops+= self.k_size[3*i+1]*(c_in/self.g_size[3*i+1])*c_out*self.out_size[3*i+1]+3*c_out*self.out_size[3*i+1]
        #         # sum_flops+= self.k_size[3*i+2]*(c_out/self.g_size[3*i+2])*self.out_csize[3*i+2]*self.out_size[3*i+2]+3*self.out_csize[3*i+2]*self.out_size[3*i+2]
        #         # i+=1

        #         sum_flops += self.k_size[3 * i] * (self.in_csize[3 * i] / self.g_size[3 * i]) * channels * self.out_size[
        #             3 * i] + 3 * channels * self.out_size[3 * i]
        #         sum_flops += self.k_size[3 * i + 1] * (channels / self.g_size[3 * i + 1]) * channels * self.out_size[
        #             3 * i + 1] + 3 * channels * self.out_size[3 * i + 1]
        #         sum_flops += self.k_size[3 * i + 2] * (channels / self.g_size[3 * i + 2]) * self.out_csize[3 * i + 2] * \
        #                      self.out_size[3 * i + 2] + 3 * self.out_csize[3 * i + 2] * self.out_size[3 * i + 2]
        # #print(sum_flops/self.t_flops)

        return sum_flops


    def forward(self, input):

        sum_flops = self.get_flops(input)

        resource_ratio = (sum_flops / self.t_flops)
        # abs_rv = torch.clamp(resource_ratio, min=self.p) ### 现在有大的变小，之后补充小的
        # loss = torch.log((abs_rv / (self.p))+1e-8)
        if resource_ratio > self.p:
            abs_rv = torch.clamp(resource_ratio, min=self.p + 0.005)
            loss = torch.log((abs_rv / (self.p)))
        else:
            abs_rv = torch.clamp(resource_ratio, max=self.p - 0.005)
            loss = torch.log(((self.p) / abs_rv))

        # loss = torch.log(torch.abs(sum_flops/self.t_flops - (self.p))+ 1)

        # print(sum_flops)
        #print(sum_flops/self.t_flops)
        #loss = -torch.log(1-torch.abs(sum_flops/self.t_flops - (self.p)) + 1e-8)
        #loss = torch.abs(sum_flops/self.t_flops - (self.p))**2
        return self.weight*loss

class Channel_constraint(nn.Module):
    def __init__(self, remaining_rate,  w=10):
        super(Channel_constraint, self).__init__()

        self.p = remaining_rate
        self.w = w

    def forward(self, input):
        overall_length = 0
        for i in range(len(input)):
            overall_length += input[i].size(0)

        for i in range(len(input)):
            if i == 0:
                # cat_tensor = input[i]
                # cat_tensor = F.tanh(input[i].abs().pow(1 / 2))
                cat_tensor = custom_STE.apply(input[i], False)
            else:
                # current_value = F.tanh(input[i].abs().pow(1 / 2))
                current_value = custom_STE.apply(input[i], False)
                cat_tensor = torch.cat([cat_tensor, current_value])

        loss = torch.log(torch.abs(cat_tensor.sum() / overall_length - (self.p)) + 1)
        return self.w*loss

class Flops_constraint_densenet(nn.Module):
    def __init__(self, p, kernel_size, out_size, group_size, size_inchannel, size_outchannel, in_channel=3, w=10, HN=False, structure=None):
        super(Flops_constraint_densenet, self).__init__()

        self.p = p
        self.k_size = kernel_size
        self.out_size = out_size
        self.g_size = group_size
        self.in_csize = size_inchannel
        self.out_csize = size_outchannel
        self.t_flops = self.init_total_flops()
        self.inc_1st = in_channel
        self.num_gates = 2
        self.weight = w
        self.HN = HN
        self.structure = structure
        print(self.p)
    def init_total_flops(self):
        total_flops = 0
        for i in range(len(self.k_size)):
            kernel_ops = self.k_size[i]*(self.in_csize[i]/self.g_size[i])

            total_flops+= kernel_ops*self.out_csize[i]*self.out_size[i]+3*self.out_csize[i]*self.out_size[i]
        print('+ Number of FLOPs: %.5fG'%(total_flops/1e9))
        return total_flops
    def forward(self, input):
        #c_in = self.inc_1st
        sum_flops = 0
        #print(len(self.k_size))

        #print(len(input))
        if self.num_gates==2:
            i = 0
            if self.HN:
                # self.h = self.input.register_hook(lambda grad: grad)
                arch_vector = []
                start = 0
                for i in range(len(self.structure)):
                    end = start + self.structure[i]
                    input = input.squeeze()
                    #print(input[start:end].size())
                    arch_vector.append(input[start:end])
                    start = end

                length = len(arch_vector)
                # print(length)
            else:
                length = len(input)
            i = 0
            for ind in range(0,length):

                if self.HN:
                    channels = arch_vector[ind].sum()
                    #c_out = arch_vector[ind+1].sum()

                else:
                    current_tensor = custom_STE.apply(input[ind], False)

                    #next_tensor = custom_STE.apply(input[ind+1], False)

                    channels = current_tensor.sum()
                    #c_out = next_tensor.sum()

                sum_flops += self.k_size[2 * i] * (self.in_csize[2 * i] / self.g_size[2 * i]) * channels * self.out_size[
                    2 * i] + 3 * channels * self.out_size[2 * i]
                sum_flops += self.k_size[2 * i + 1] * (channels / self.g_size[2 * i + 1]) * self.out_csize[2 * i + 1] * \
                             self.out_size[2 * i + 1] + 3 * self.out_csize[2 * i + 1] * self.out_size[2 * i + 1]
                i+=1

            #print(i)
            #print(3*i+2)
        else:
            for i in range(0, len(input)):
                #print(i)

                current_tensor = custom_STE.apply(input[i], False)


                channels = current_tensor.sum()

                sum_flops += self.k_size[2 * i] * (self.in_csize[2 * i] / self.g_size[2 * i]) * channels * self.out_size[
                    2 * i] + 3 * channels * self.out_size[2 * i]
                sum_flops += self.k_size[2 * i + 1] * (channels / self.g_size[2 * i + 1]) * self.out_csize[2 * i + 1] * \
                             self.out_size[2 * i + 1] + 3 * self.out_csize[2 * i + 1] * self.out_size[2 * i + 1]
        #print(sum_flops/self.t_flops)

        resource_value = sum_flops / self.t_flops - (self.p)
        abs_rv = torch.clamp(torch.abs(resource_value), min=0.002)
        loss = torch.log(abs_rv+ 1)

        return self.weight*loss


class flops_tracker(object):
    def __init__(self, kernel_size, out_size, group_size, size_inchannel, size_outchannel):
        self.k_size = kernel_size
        self.out_size = out_size
        self.g_size = group_size
        self.in_csize = size_inchannel
        self.out_csize = size_outchannel
    def current_flops(self, input):
        sum_flops = 0
        i = 0
        for ind in range(0, len(input), 2):
            current_tensor = custom_STE.apply(input[ind].detach(), False)

            next_tensor = custom_STE.apply(input[ind + 1].detach(), False)

            c_in = current_tensor.sum()
            c_out = next_tensor.sum()

            sum_flops += self.k_size[3 * i] * (self.in_csize[3 * i] / self.g_size[3 * i]) * c_in * self.out_size[
                3 * i] + 3 * c_in * self.out_size[3 * i]
            sum_flops += self.k_size[3 * i + 1] * (c_in / self.g_size[3 * i + 1]) * c_out * self.out_size[
                3 * i + 1] + 3 * c_out * self.out_size[3 * i + 1]
            sum_flops += self.k_size[3 * i + 2] * (c_out / self.g_size[3 * i + 2]) * self.out_csize[3 * i + 2] * \
                         self.out_size[3 * i + 2] + 3 * self.out_csize[3 * i + 2] * self.out_size[3 * i + 2]
            i += 1
        print('+ Number of FLOPs: %.5fG' % (sum_flops / 1e9))

class Flops_constraint_mobnet(nn.Module):
    def __init__(self, p, kernel_size, out_size, group_size, size_inchannel, size_outchannel, in_channel=3, w=2, HN=False, structure=None, last_flag=False):
        super(Flops_constraint_mobnet, self).__init__()

        self.p = p
        self.k_size = kernel_size
        self.out_size = out_size
        self.g_size = group_size
        self.in_csize = size_inchannel
        self.out_csize = size_outchannel

        self.weight = w
        self.inc_1st = in_channel
        self.HN = HN
        self.structure = structure
        self.last_flag = last_flag
        self.t_flops = self.init_total_flops()
    def init_total_flops(self):
        total_flops = 0
        detail_flops = []


        for i in range(len(self.k_size)):
            if self.last_flag and i == len(self.k_size)-1:
                current_flops = self.in_csize[i]*self.out_csize[i] + self.out_csize[i]
                total_flops+= current_flops
            else:
                current_flops = self.k_size[i] * (self.in_csize[i] / self.g_size[i]) * self.out_csize[i] * \
                                self.out_size[i] + 3 * self.out_csize[i] * self.out_size[i]
                total_flops += current_flops

            detail_flops.append(current_flops)
        detail_flops = [f/1e9 for f in detail_flops]
        #print(detail_flops)
        print('+ Number of FLOPs: %.5fG'%(total_flops/1e9))
        self.last_flops = total_flops
        return total_flops

    def print_current_FLOPs(self, input):
        sum_flops = 0
        #print(len(self.k_size))
        if self.HN:
            #self.h = self.input.register_hook(lambda grad: grad)
            arch_vector = []
            start = 0
            for i in range(len(self.structure)):
                end = start + self.structure[i]
                input = input.squeeze()
                # print(input[start:end].size())
                arch_vector.append(input[start:end])
                start = end

            length = len(arch_vector)
            print(length)
        else:
            length = len(input)
        #print(length)
        for i in range(length):
            #print(i)
            if self.HN is False:
                current_tensor = custom_STE.apply(input[i].detach(), False)
            # if i >0:
            #     c_in = custom_STE.apply(input[i-1], False).sum()
            else:
                current_tensor = arch_vector[i].detach()
            #current_tensor = custom_STE.apply(input[i].detach().cpu(), False)
            # if i >0:
            #     c_in = custom_STE.apply(input[i-1], False).sum()
            channels = current_tensor.sum()
            #two layer as a group
            if i == length-1 and self.last_flag:
                #if self.last_flag:
                sum_flops+=self.k_size[3*i]*(self.in_csize[3*i]/self.g_size[3*i])*channels*self.out_size[3*i]+3*channels*self.out_size[3*i]

                sum_flops+=channels*self.out_csize[3*i+1] + self.out_csize[3*i+1]
            else:
                sum_flops+= self.k_size[3*i]*(self.in_csize[3*i]/self.g_size[3*i])*channels*self.out_size[3*i]+3*channels*self.out_size[3*i]

                sum_flops+= self.k_size[3*i+1]*(channels/self.g_size[3*i+1])*channels*self.out_size[3*i+1]+3*channels*self.out_size[3*i+1]

                sum_flops += self.k_size[3 * i + 2] * (channels / self.g_size[3 * i + 2]) * self.out_csize[3 * i + 2] * \
                             self.out_size[3 * i + 2] + 3 * self.out_csize[3 * i + 2] * self.out_size[3 * i + 2]

        print('+ Current FLOPs: %.5fG'%(sum_flops/1e9))

    def forward(self, input):
        #c_in = self.inc_1st
        sum_flops = 0
        if self.HN:
            #self.h = self.input.register_hook(lambda grad: grad)
            arch_vector = []
            start = 0
            for i in range(len(self.structure)):
                end = start + self.structure[i]
                input = input.squeeze()
                # print(input[start:end].size())
                arch_vector.append(input[start:end])
                start = end

            length = len(arch_vector)
            #print(length)
        else:
            length = len(input)

        #print(len(self.k_size))
        for i in range(length):
            #print(i)

            if self.HN is False:
                current_tensor = custom_STE.apply(input[i], False)
            # if i >0:
            #     c_in = custom_STE.apply(input[i-1], False).sum()
            else:
                current_tensor = arch_vector[i]
            # if i >0:
            #     c_in = custom_STE.apply(input[i-1], False).sum()
            channels = current_tensor.sum()
            #two layer as a group
            if i == length - 1 and self.last_flag:
                # if self.last_flag:
                sum_flops += self.k_size[3 * i] * (self.in_csize[3 * i] / self.g_size[3 * i]) * channels * \
                             self.out_size[3 * i] + 3 * channels * self.out_size[3 * i]

                sum_flops += channels * self.out_csize[3 * i + 1] + self.out_csize[3 * i + 1]

            else:
                sum_flops+= self.k_size[3*i]*(self.in_csize[3*i]/self.g_size[3*i])*channels*self.out_size[3*i]+3*channels*self.out_size[3*i]

                sum_flops+= self.k_size[3*i+1]*(channels/self.g_size[3*i+1])*channels*self.out_size[3*i+1]+3*channels*self.out_size[3*i+1]

                sum_flops += self.k_size[3 * i + 2] * (channels / self.g_size[3 * i + 2]) * self.out_csize[3 * i + 2] * \
                             self.out_size[3 * i + 2] + 3 * self.out_csize[3 * i + 2] * self.out_size[3 * i + 2]
        #if self.t_flops - (self.p)>0:
        #resource_value = torch.clamp(sum_flops / self.t_flops - (self.p), min=0)


        # resource_ratio = (sum_flops / self.t_flops)
        # abs_rv = (resource_ratio/self.p - 1).abs()

        # resource_value = sum_flops / self.t_flops - (self.p)
        # abs_rv = torch.abs(resource_value)
        #
        #
        # print(abs_rv)

        # resource_value = (sum_flops / self.t_flops) / (self.p)
        # abs_rv = (resource_value -1).abs()
        # # loss = 1 - torch.exp(-abs_rv / 0.25)
        # loss = torch.exp(abs_rv / 0.25) - 1
        # loss = - torch.log(1 - abs_rv + 1e-8)

        #ratio regularization
        resource_ratio = (sum_flops / self.t_flops)
        abs_rv = torch.clamp(resource_ratio, min=self.p)
        loss = torch.log((abs_rv / (self.p)))

        # resource_ratio = (sum_flops / self.t_flops)
        # abs_rv = torch.clamp(resource_ratio, min=self.p+0.01)
        # loss = torch.log((abs_rv / self.p))
        # # loss = torch.log(((resource_ratio/self.p).pow(0.9)-1).abs()+1)
        # loss = torch.log((abs_rv/self.p).pow(1.2))
        #

        # + torch.abs(sum_flops/self.t_flops- self.last_flops/self.t_flops)**2

        # diff = torch.abs(sum_flops/self.t_flops - (self.p))**2
        # loss = torch.log(diff+1)
        #self.last_flops = sum_flops.detach()
        return self.weight*loss

class Flops_constraint_mobnetv3(nn.Module):
    def __init__(self, p, all_dict, in_channel=3, w=2, HN=False, structure=None):
        super(Flops_constraint_mobnetv3, self).__init__()

        self.p = p
        self.all_dict = all_dict

        self.k_size = all_dict['size_kernel']
        self.out_size = all_dict['size_out']
        self.g_size = all_dict['size_group']
        self.in_csize = all_dict['size_inchannel']
        self.out_csize = all_dict['size_outchannel']
        self.hswish_list = all_dict['hswish_list']
        self.se_list = all_dict['se_list']


        self.weight = w
        self.inc_1st = in_channel
        self.HN = HN
        self.structure = structure

        self.t_flops = self.init_total_flops()
    def init_total_flops(self):
        total_flops = 0
        detail_flops = []

        # index = 0
        for i in range(len(self.k_size)//3):
            # if i%3 == 0:
            current_se = self.se_list[i]
            current_act = self.hswish_list[i]


            if current_se:
                adpool_flops = self.out_size[3*i+1]*self.out_csize[3*i+1]
                se_flops = self.out_csize[3*i+1]*(self.out_csize[3*i+1]//4)*2 + self.out_csize[3*i+1] + adpool_flops
            else:
                se_flops = 0

            if current_act:
                act_mul = 4.0
            else:
                act_mul = 1.0

            conv1_flops = self.k_size[3*i] * (self.in_csize[3*i] / self.g_size[3*i]) * self.out_csize[3*i] * \
                            self.out_size[3*i] + (2+act_mul) * self.out_csize[3*i] * self.out_size[3*i]
            conv2_flops = self.k_size[3*i+1] * (self.in_csize[3*i+1] / self.g_size[3*i+1]) * self.out_csize[3*i+1] * \
                            self.out_size[3*i+1] + (2+act_mul) * self.out_csize[3*i+1] * self.out_size[3*i+1]
            conv3_flops = self.k_size[3*i+2] * (self.in_csize[3*i+2] / self.g_size[3*i+2]) * self.out_csize[3*i+2] * \
                            self.out_size[3*i+2] + (2+act_mul) * self.out_csize[3*i+2] * self.out_size[3*i+2]


            current_flops = conv1_flops + conv2_flops + conv3_flops + se_flops
            total_flops += current_flops

            detail_flops.append(current_flops)
        detail_flops = [f/1e9 for f in detail_flops]
        #print(detail_flops)
        print('+ Number of FLOPs: %.5fG'%(total_flops/1e9))
        self.last_flops = total_flops
        return total_flops

    def print_current_FLOPs(self, input):
        sum_flops = 0
        #print(len(self.k_size))
        if self.HN:
            #self.h = self.input.register_hook(lambda grad: grad)
            arch_vector = []
            start = 0
            for i in range(len(self.structure)):
                end = start + self.structure[i]
                input = input.squeeze()
                # print(input[start:end].size())
                arch_vector.append(input[start:end])
                start = end

            length = len(arch_vector)
            print(length)
        else:
            length = len(input)
        #print(length)
        for i in range(length):
            #print(i)
            if self.HN is False:
                current_tensor = custom_STE.apply(input[i].detach(), False)
            # if i >0:
            #     c_in = custom_STE.apply(input[i-1], False).sum()
            else:
                current_tensor = arch_vector[i].detach()
            #current_tensor = custom_STE.apply(input[i].detach().cpu(), False)
            # if i >0:
            #     c_in = custom_STE.apply(input[i-1], False).sum()
            channels = current_tensor.sum()
            #two layer as a group

            current_se = self.se_list[i]
            current_act = self.hswish_list[i]

            if current_se:
                adpool_flops = self.out_size[3 * i + 1] * channels
                se_flops = channels * (self.out_csize[3 * i + 1] // 4) * 2 + self.out_csize[
                    3 * i + 1] + adpool_flops
            else:
                se_flops = 0

            if current_act:
                act_mul = 4.0
            else:
                act_mul = 1.0

            conv1_flops = self.k_size[3 * i] * (self.in_csize[3 * i] / self.g_size[3 * i]) * channels * \
                          self.out_size[3 * i] + (2 + act_mul) * channels * self.out_size[3 * i]
            conv2_flops = self.k_size[3 * i + 1] * (channels / self.g_size[3 * i + 1]) *channels * \
                          self.out_size[3 * i + 1] + (2 + act_mul) * channels * self.out_size[
                              3 * i + 1]
            conv3_flops = self.k_size[3 * i + 2] * (channels / self.g_size[3 * i + 2]) * self.out_csize[
                3 * i + 2] * \
                          self.out_size[3 * i + 2] + (2 + act_mul) * self.out_csize[3 * i + 2] * self.out_size[
                              3 * i + 2]
            current_flops = conv1_flops + conv2_flops + conv3_flops + se_flops
            sum_flops += current_flops

        print('+ Current FLOPs: %.5fG'%(sum_flops/1e9))

    def forward(self, input):
        #c_in = self.inc_1st
        sum_flops = 0
        if self.HN:
            #self.h = self.input.register_hook(lambda grad: grad)
            arch_vector = []
            start = 0
            for i in range(len(self.structure)):
                end = start + self.structure[i]
                input = input.squeeze()
                # print(input[start:end].size())
                arch_vector.append(input[start:end])
                start = end

            length = len(arch_vector)
            #print(length)
        else:
            length = len(input)

        #print(len(self.k_size))
        for i in range(length):
            #print(i)

            if self.HN is False:
                current_tensor = custom_STE.apply(input[i], False)

            else:
                current_tensor = arch_vector[i]

            channels = current_tensor.sum()

            current_se = self.se_list[i]
            current_act = self.hswish_list[i]

            if current_se:
                adpool_flops = self.out_size[3 * i + 1] * channels
                se_flops = channels * (self.out_csize[3 * i + 1] // 4) * 2 + self.out_csize[
                    3 * i + 1] + adpool_flops
            else:
                se_flops = 0

            if current_act:
                act_mul = 4.0
            else:
                act_mul = 1.0

            conv1_flops = self.k_size[3 * i] * (self.in_csize[3 * i] / self.g_size[3 * i]) * channels * \
                          self.out_size[3 * i] + (2 + act_mul) * channels * self.out_size[3 * i]
            conv2_flops = self.k_size[3 * i + 1] * (channels / self.g_size[3 * i + 1]) * channels * \
                          self.out_size[3 * i + 1] + (2 + act_mul) * channels * self.out_size[
                              3 * i + 1]
            conv3_flops = self.k_size[3 * i + 2] * (channels / self.g_size[3 * i + 2]) * self.out_csize[
                3 * i + 2] * \
                          self.out_size[3 * i + 2] + (2 + act_mul) * self.out_csize[3 * i + 2] * self.out_size[
                              3 * i + 2]
            current_flops = conv1_flops + conv2_flops + conv3_flops + se_flops

            sum_flops += current_flops

        #ratio regularization
        resource_ratio = (sum_flops / self.t_flops)
        abs_rv = torch.clamp(resource_ratio, min=self.p)
        loss = torch.log((abs_rv / (self.p))+1e-8)

        return self.weight*loss


class push_to_int(nn.Module):
    def __init__(self, num_epoch, cut_off_epoch):
        super(push_to_int, self).__init__()
        self.num_epoch = num_epoch
        self.cut_off_epoch = cut_off_epoch

    def forward(self, input, epoch):
        #loss = 0
        for i in range(len(input)):
            if i == 0:
                #cat_tensor = input[i]

                #current_value = torch.sigmoid(input[i])
                current_value = F.tanh(input[i].abs().pow(1 / 2))
                #binary_value = (current_value > 0.5).detach().float()
                #loss = torch.abs(binary_value - current_value)**2
                loss = binary_loss(current_value)
                loss = loss.mean()
            else:
                #cat_tensor = torch.cat([cat_tensor, input[i]])
                #current_value = torch.sigmoid(input[i])
                current_value = F.tanh(input[i].abs().pow(1 / 2))
                # binary_value = (current_value > 0.5).detach().float()
                # c_loss = torch.abs(binary_value - current_value)**2
                c_loss = binary_loss(current_value)
                c_loss = c_loss.mean()
                loss+=c_loss

        return -loss

def TrainVal_split(dataset, validation_split,shuffle_dataset=True):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(0)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler

def binary_loss(cat_tensor):
    loss = cat_tensor * torch.log(cat_tensor + 1e-8) * (cat_tensor >= 0.5).detach().float() + (
                    1 - cat_tensor) * torch.log(1 - cat_tensor + 1e-8) * (cat_tensor < 0.5).detach().float()
    return loss.mean()

def display_structure(all_parameters, p_flag=False):
    num_layers = len(all_parameters)
    layer_sparsity = []
    for i in range(num_layers):
        if i == 0 and p_flag is True:
            current_parameter = all_parameters[i].cpu().data
            print(current_parameter)
            #print(all_parameters[i].grad.cpu().data)
        current_parameter = all_parameters[i].cpu().data
        layer_sparsity.append((current_parameter>=0.5).sum().float().item()/current_parameter.size(0))

    print_string = ''
    for i in range(num_layers):
        print_string += 'l-%d s-%.3f '%(i+1, layer_sparsity[i])

    print(print_string)

def display_structure_hyper(vectors):
    num_layers = len(vectors)
    layer_sparsity = []
    for i in range(num_layers):

        current_parameter = vectors[i].cpu().data
        if i == 0:
            print(current_parameter)
        layer_sparsity.append(current_parameter.sum().item()/current_parameter.size(0))
    print_string = ''
    for i in range(num_layers):
        print_string += 'l-%d s-%.3f ' % (i + 1, layer_sparsity[i])
    return_string = ''
    for i in range(num_layers):
        return_string += '%.3f ' % (layer_sparsity[i])
    print(print_string)
    return return_string

def loss_fn_kd(outputs, labels, teacher_outputs, T, alpha):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    # T = params[0]
    # alpha = params[1]
    #beta = params[2]

    labels.requires_grad = False
    #teacher_outputs.detach()

    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs.detach()/T, dim=1)) * (alpha ) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    #Teacher_loss = F.cross_entropy(teacher_outputs, labels)
    #KD_loss = KD_loss + beta* Teacher_loss
    return KD_loss

def loss_label_smoothing(outputs, labels, T, alpha):
    uniform = torch.Tensor(outputs.size())
    uniform.fill_(1/outputs.size(1))
    if outputs.is_cuda:
        uniform = uniform.cuda()
    sm_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(uniform/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return sm_loss

def print_model_param_nums(model=None, multiply_adds=True):
    if model == None:
        model = torchvision.models.alexnet()
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    return total


def print_model_param_flops(model=None, input_res=224, multiply_adds=False):
    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        if self.bias is not None:
            bias_ops = self.bias.nelement()
        else:
            bias_ops = 0

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_hswish = []

    def hswish_hook(self, input, output):
        list_hswish.append(5*input[0].nelement())


    list_hsigmoid = []

    def hsigmoid_hook(self, input, output):
        list_hsigmoid.append(4*input[0].nelement())

    list_adpooling = []

    def adpooling_hook(self, input, output):
        kernel = torch.DoubleTensor([*(input[0].shape[2:])]) // torch.DoubleTensor([*(output.shape[2:])])

        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()



        total_add = torch.prod(kernel)
        total_div = 1
        kernel_ops = total_add + total_div
        num_elements = output_channels * output_height * output_width * batch_size
        flops = kernel_ops * num_elements

        list_adpooling.append(flops)

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample = []

    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.AdaptiveAvgPool2d) or isinstance(net, torch.nn.AdaptiveMaxPool2d):
                net.register_forward_hook(adpooling_hook)

            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            if isinstance(net, Hswish):
                net.register_forward_hook(hswish_hook)
            if isinstance(net, Hsigmoid):
                net.register_forward_hook(hsigmoid_hook)
            return
        for c in childrens:
            foo(c)

    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    input = torch.rand(3, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    # print(sum(list_conv)/3/1e9)
    # print(list_conv[0]/3/1e9)
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(
        list_upsample))

    print('  + Number of FLOPs: %.5fG' % (total_flops / 3 / 1e9))

    return total_flops

def get_middle_Fsize(model, input_res=32):
    #size_in = []
    size_out = []
    size_kernel = []
    size_group = []
    size_inchannel = []
    size_outchannel = []
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        size_out.append(output_height*output_width)
        size_kernel.append(self.kernel_size[0] * self.kernel_size[1])
        size_group.append(self.groups)
        size_inchannel.append(input_channels)
        size_outchannel.append(output_channels)
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            return
        for c in childrens:
            foo(c)
    foo(model)
    input = torch.rand(2, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    print(len(size_out))
    print(len(size_kernel))

    return size_out, size_kernel, size_group, size_inchannel, size_outchannel

def get_middle_Fsize_resnet(model, input_res=224):
    size_out = []
    size_kernel = []
    size_group = []
    size_inchannel = []
    size_outchannel = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        size_out.append(output_height * output_width)
        size_kernel.append(self.kernel_size[0] * self.kernel_size[1])
        size_group.append(self.groups) # ???
        size_inchannel.append(input_channels)
        size_outchannel.append(output_channels)

    def foo(net):
        modules = list(net.modules())
        #print(modules)

        for layer_id in range(len(modules)):
            m = modules[layer_id]
            #print(m)
            #if layer_id + 3 <= len(modules):
            if  isinstance(m, virtual_gate):
                #print(m)

                modules[layer_id - 3].register_forward_hook(conv_hook)
                modules[layer_id + 1 ].register_forward_hook(conv_hook)
                # print(modules[layer_id - 3])
                # print(modules[layer_id + 1])
            # elif isinstance(modules[layer_id - 1], soft_gate):
            #     print(m)
            #     m.register_forward_hook(conv_hook)

    foo(model)
    input = torch.rand(2, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    print(len(size_out))
    print(len(size_kernel))

    return size_out, size_kernel, size_group, size_inchannel, size_outchannel



def get_middle_Fsize_resnetbb(model, input_res=224, num_gates=2):
    size_out = []
    size_kernel = []
    size_group = []
    size_inchannel = []
    size_outchannel = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        size_out.append(output_height * output_width)
        size_kernel.append(self.kernel_size[0] * self.kernel_size[1])
        size_group.append(self.groups)
        size_inchannel.append(input_channels)
        size_outchannel.append(output_channels)

    def foo(net):
        modules = list(net.modules())
        #print(modules)
        soft_gate_count=0
        for layer_id in range(len(modules)):
            m = modules[layer_id]
            #print(m)
            #if layer_id + 3 <= len(modules):
            if  isinstance(m, virtual_gate):
                if num_gates==2:
                    modules[layer_id - 2].register_forward_hook(conv_hook)
                    if soft_gate_count%2 == 1:

                        modules[layer_id + 1].register_forward_hook(conv_hook) # ???
                    soft_gate_count += 1
                else:
                    modules[layer_id - 4].register_forward_hook(conv_hook)
                    modules[layer_id - 2].register_forward_hook(conv_hook)
                    modules[layer_id + 1].register_forward_hook(conv_hook)


    foo(model)
    input = torch.rand(2, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    print(len(size_out))
    print(len(size_kernel))

    return size_out, size_kernel, size_group, size_inchannel, size_outchannel

def get_middle_Fsize_densenet(model, input_res=224):
    size_out = []
    size_kernel = []
    size_group = []
    size_inchannel = []
    size_outchannel = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        size_out.append(output_height * output_width)
        size_kernel.append(self.kernel_size[0] * self.kernel_size[1])
        size_group.append(self.groups)
        size_inchannel.append(input_channels)
        size_outchannel.append(output_channels)
    def foo(net):
        modules = list(net.modules())
        #print(modules)
        truncate_module = []
        for layer_id in range(len(modules)):
            m0 = modules[layer_id]
            if isinstance(m0, nn.BatchNorm2d) or isinstance(m0, nn.Conv2d) or isinstance(m0, nn.Linear) or isinstance(m0, nn.ReLU) or isinstance(m0, virtual_gate):
                truncate_module.append(m0)

        #print(truncate_module)

        for layer_id in range(len(truncate_module)):
            m = truncate_module[layer_id]
            #print(m)
            #if layer_id + 3 <= len(modules):
            #print(truncate_module[layer_id + 2])
            if isinstance(m, virtual_gate):
                #print(m)
                #print(truncate_module[layer_id - 1])
                truncate_module[layer_id - 1].register_forward_hook(conv_hook)
                truncate_module[layer_id + 3].register_forward_hook(conv_hook)
                #truncate_module[layer_id + 4].register_forward_hook(conv_hook)


    foo(model)
    input = torch.rand(2, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    print(len(size_out))
    print(len(size_kernel))

    return size_out, size_kernel, size_group, size_inchannel, size_outchannel

def get_middle_Fsize_mobnetv3(model, input_res=224):
    all_dict = {}

    all_dict['size_out'] = []
    all_dict['size_kernel'] = []
    all_dict['size_group'] = []
    all_dict['size_inchannel'] = []
    all_dict['size_outchannel'] = []
    all_dict['se_list'] = []
    all_dict['hswish_list'] = []



    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        all_dict['size_out'].append(output_height * output_width)
        all_dict['size_kernel'].append(self.kernel_size[0] * self.kernel_size[1])
        all_dict['size_group'].append(self.groups)
        all_dict['size_inchannel'].append(input_channels)
        all_dict['size_outchannel'].append(output_channels)

    # def linear_hook(self, input, output):
    #     #batch_size, input_channels  = input[0].size()final_gate
    #     batch_size = input[0].size(0) if input[0].dim() == 2 else 1
    #     input_size = input[0].size(1) if input[0].dim() == 2 else input[0].size(0)
    #     #print(output.size())
    #     #output_size = output[0].size(1) if input[0].dim() == 2 else input[0].size(0)
    #     output_size = output.size(1)
    #
    #     weight_ops = self.weight.nelement()
    #     assert weight_ops == input_size*output_size
    #     #bias_ops = self.bias.nelement()
    #     size_out.append(-1)
    #     size_kernel.append(-1)
    #     size_group.append(-1)
    #     size_inchannel.append(input_size)
    #     size_outchannel.append(output_size)
        #flops = batch_size * (weight_ops + bias_ops)
    def foo(net):
        modules = list(net.modules())
        #print(modules)
        truncate_module = []
        for layer_id in range(len(modules)):
            m0 = modules[layer_id]
            if isinstance(m0, nn.BatchNorm2d) or isinstance(m0, nn.Conv2d) or isinstance(m0, nn.Linear) or isinstance(m0, nn.ReLU) or \
                    isinstance(m0, nn.ReLU6) or isinstance(m0, virtual_gate) or isinstance(m0, Hswish):
                truncate_module.append(m0)
        # or isinstance(m0, Hsigmoid)
        # print(truncate_module)

        for layer_id in range(len(truncate_module)):
            m = truncate_module[layer_id]
            #print(m)
            #if layer_id + 3 <= len(modules):


            if isinstance(m, virtual_gate) and layer_id+4<len(truncate_module)-1:
                #print(m)
                if isinstance(truncate_module[layer_id - 1], Hswish):
                    all_dict['hswish_list'].append(True)
                elif isinstance(truncate_module[layer_id - 1], nn.ReLU):
                    all_dict['hswish_list'].append(False)


                truncate_module[layer_id - 3].register_forward_hook(conv_hook)

                truncate_module[layer_id + 1].register_forward_hook(conv_hook)

                if isinstance(truncate_module[layer_id + 4], nn.Conv2d):
                    truncate_module[layer_id + 4].register_forward_hook(conv_hook)
                    all_dict['se_list'].append(False)
                else:
                    truncate_module[layer_id + 7].register_forward_hook(conv_hook)
                    all_dict['se_list'].append(True)

    foo(model)
    input = torch.rand(2, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    print(len(all_dict['size_out']))
    print(len(all_dict['size_kernel']))
    print(all_dict['se_list'])
    print(all_dict['hswish_list'])

    return all_dict

def transfer_weights(model, my_model):
    mymbnet_v2_ms = list(my_model.modules())
    mbnet_v2_ms = list(model.modules())
    # mbnet_v2.set_training_flag(False)
    # print(resnet_50_ms)
    # print(mymbnet_v2_ms)
    for m in mymbnet_v2_ms:
        if isinstance(m, virtual_gate):
            mymbnet_v2_ms.remove(m)

    print(len(mymbnet_v2_ms))
    print(len(mbnet_v2_ms))

    # print(mbnet_v2)
    for layer_id in range(len(mymbnet_v2_ms)):
        m0 = mbnet_v2_ms[layer_id]
        m1 = mymbnet_v2_ms[layer_id]
        if isinstance(m0, nn.BatchNorm2d):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
        elif isinstance(m0, nn.Conv2d):
            m1.weight.data = m0.weight.data.clone()
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()

def get_middle_Fsize_mobnet(model, input_res=224,last_gate=False):
    size_out = []
    size_kernel = []
    size_group = []
    size_inchannel = []
    size_outchannel = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        size_out.append(output_height * output_width)
        size_kernel.append(self.kernel_size[0] * self.kernel_size[1])
        size_group.append(self.groups)
        size_inchannel.append(input_channels)
        size_outchannel.append(output_channels)

    def linear_hook(self, input, output):
        #batch_size, input_channels  = input[0].size()final_gate
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        input_size = input[0].size(1) if input[0].dim() == 2 else input[0].size(0)
        #print(output.size())
        #output_size = output[0].size(1) if input[0].dim() == 2 else input[0].size(0)
        output_size = output.size(1)

        weight_ops = self.weight.nelement()
        assert weight_ops == input_size*output_size
        #bias_ops = self.bias.nelement()
        size_out.append(-1)
        size_kernel.append(-1)
        size_group.append(-1)
        size_inchannel.append(input_size)
        size_outchannel.append(output_size)
        #flops = batch_size * (weight_ops + bias_ops)
    def foo(net):
        modules = list(net.modules())
        #print(modules)
        truncate_module = []
        for layer_id in range(len(modules)):
            m0 = modules[layer_id]
            if isinstance(m0, nn.BatchNorm2d) or isinstance(m0, nn.Conv2d) or isinstance(m0, nn.Linear) or isinstance(m0, nn.ReLU6) or isinstance(m0, virtual_gate):
                truncate_module.append(m0)

        #print(truncate_module)

        for layer_id in range(len(truncate_module)):
            m = truncate_module[layer_id]
            #print(m)
            #if layer_id + 3 <= len(modules):
            if isinstance(m, virtual_gate) and layer_id+4<len(truncate_module)-1:
                #print(m)

                truncate_module[layer_id - 3].register_forward_hook(conv_hook)
                truncate_module[layer_id + 1].register_forward_hook(conv_hook)
                truncate_module[layer_id + 4].register_forward_hook(conv_hook)
                # print(modules[layer_id - 3])
                # print(modules[layer_id + 1])
            # elif isinstance(modules[layer_id - 1], soft_gate):
            #     print(m)
            #     m.register_forward_hook(conv_hook)
            if last_gate:
                if isinstance(m, nn.Linear):
                    truncate_module[layer_id-4].register_forward_hook(conv_hook)
                    truncate_module[layer_id].register_forward_hook(linear_hook)
    foo(model)
    input = torch.rand(2, 3, input_res, input_res)
    input.require_grad = True
    out = model(input)
    print(len(size_out))
    print(len(size_kernel))

    return size_out, size_kernel, size_group, size_inchannel, size_outchannel

def moving_average(net1, net2, alpha=1):

    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha

def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda()
        input_var = input
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def one_hot(y, num_classes, smoothing_eps=None):
    if smoothing_eps is None:
        one_hot_y = F.one_hot(y, num_classes).float()
        return one_hot_y
    else:
        one_hot_y = F.one_hot(y, num_classes).float()
        v1 = 1 - smoothing_eps + smoothing_eps / float(num_classes)
        v0 = smoothing_eps / float(num_classes)
        new_y = one_hot_y * (v1 - v0) + v0
        return new_y

def cross_entropy_onehot_target(logit, target):
    # target must be one-hot format!!
    prob_logit = F.log_softmax(logit, dim=1)
    loss = -(target * prob_logit).sum(dim=1).mean()
    return loss

def mixup_func(input, target, alpha=0.2):
    gamma = np.random.beta(alpha, alpha)
    # target is onehot format!
    perm = torch.randperm(input.size(0))
    perm_input = input[perm]
    perm_target = target[perm]
    return input.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(1 - gamma, perm_target)


def group_weight(module, wegith_norm=True):
    group_decay = []
    group_no_decay = []
    #group_no_decay.append(module.inputs)
    if hasattr(module, 'inputs'):
        group_no_decay.append(module.inputs)
    count = 0
    #if isinstance(module, torch.nn.DataParallel):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

        elif isinstance(m, nn.Conv2d):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.GRU):

            for k in range(m.num_layers):
                # getattr(m,'weight_ih_l%d'%(k))
                # getattr(m, 'weight_hh_l%d' % (k))
                group_decay.append(getattr(m,'weight_ih_l%d'%(k)))
                group_decay.append(getattr(m, 'weight_hh_l%d' % (k)))

                if getattr(m, 'bias_hh_l%d' % (k)) is not None:
                    group_no_decay.append(getattr(m, 'bias_hh_l%d' % (k)))
                    group_no_decay.append(getattr(m, 'bias_ih_l%d' % (k)))

                if m.bidirectional:
                    group_decay.append(getattr(m, 'weight_ih_l%d_reverse' % (k)))
                    group_decay.append(getattr(m, 'weight_hh_l%d_reverse' % (k)))

                    if getattr(m, 'bias_hh_l%d_reverse' % (k)) is not None:
                        group_no_decay.append(getattr(m, 'bias_hh_l%d_reverse' % (k)))
                        group_no_decay.append(getattr(m, 'bias_ih_l%d_reverse' % (k)))

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m,nn.LayerNorm):
            if m.bias is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    print(len(list(module.parameters())))
    print(len(group_decay))
    print(len(group_no_decay))
    print(count)
    #print(module)
    assert len(list(module.parameters()))-count == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups
