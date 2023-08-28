import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.function
class soft_gate(nn.Module):
    def __init__(self, width, base_width=-1, concrete_flag=False, margin=0):
        super(soft_gate, self).__init__()
        #self.weights = nn.Parameter(torch.ones(width)*0.3017 + 0.1*torch.randn(width))
        # init_tensor = 0.5 + torch.randn(width)
        # init_tensor[init_tensor<0]=0
        # init_tensor[init_tensor > 1] = 1
        if base_width == -1:
            base_width = width
        self.weights = nn.Parameter(torch.ones(width))
        #self.gate_function =torch.sigmoid(nn.Parameter(torch.zeros(width)))
        self.training_flag = True

        #self.register_buffer('gate_function',torch.sigmoid(self.weights))
        #self.tanh = nn.Tanh()
        self.concrete_flag = concrete_flag

        self.g_w = torch.Tensor([float(base_width)/float(width)])

        self.margin = margin
        if concrete_flag:
            self.margin = 0
    def forward(self, input):
        if not self.training_flag:
            return input
        self.weights.data.copy_(self.clip_value(self.weights.data))
        if len(input.size())==2:

            if self.concrete_flag:
                gate_f = custom_STE.apply(self.weights, False)
            else:
                gate_f = custom_STE.apply(self.weights, self.training,self.g_w)

            gate_f = gate_f.unsqueeze(0)



            if input.is_cuda:
                gate_f=gate_f.cuda()

            input = gate_f.expand_as(input)*input
            return input

        elif len(input.size())==4:

            if self.concrete_flag:
                gate_f = custom_STE.apply(self.weights, False)
            else:
                gate_f = custom_STE.apply(self.weights, self.training, self.g_w)
            #gate_f = custom_STE.apply(self.weights, self.training)
            gate_f = gate_f.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            if input.is_cuda:
                gate_f=gate_f.cuda()
            # print(input.type())
            # print(gate_f.type())
            # print(gate_f.size())
            # print(input.size())
            input = gate_f.expand_as(input) * input

            return input

    def clip_value(self,x):
        x[x > 1-self.margin] = 1-self.margin
        x[x < 0+self.margin] = self.margin


        return x

# def scaled_tanh(x,T=3):
#     return 0.5*torch.tanh(T*(x-0.5))+0.5
class virtual_gate(nn.Module):
    def __init__(self,  width):
        super(virtual_gate, self).__init__()
        self.g_w = 1
        self.width = width
        self.gate_f = torch.ones(width)
     #   self.grads = None
        #self.gate_f.requires_grad=True
    def forward(self, input):
        # orignal_norm = input.detach().view(input.size(0),-1).norm(p=2, dim=1).squeeze()
        # print(orignal_norm.size())
        orignal_input = input.detach() # Important
        if len(input.size())==2:

            # if self.training:
            #     self.h = self.gate_f.register_hook(lambda grad: grad)

            gate_f = self.gate_f.unsqueeze(0)

            #gate_f = custom_grad_weight.apply(gate_f,self.g_w)
            if input.is_cuda:
                gate_f=gate_f.cuda()
            input = gate_f.expand_as(input)*input


            return input

        elif len(input.size())==4:
            # if self.training:
            #     self.h = self.gate_f.register_hook(lambda grad: grad)
            gate_f = self.gate_f.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            #gate_f = custom_grad_weight.apply(gate_f, self.g_w)

            if input.is_cuda:
                gate_f=gate_f.cuda()
            # print(input.size())
            # print(gate_f.size())
            # input = gate_f.expand_as(input) * input
            pruned_input = gate_f.expand_as(input) * input
            # pruned_norm = pruned_input.detach().view(input.size(0),-1).norm(p=2, dim=1)
            # adjusted_factor = orignal_norm/pruned_norm
            # adjusted_factor = adjusted_factor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            # input = adjusted_factor.expand_as(input)* gate_f.expand_as(input) * input
            self.mse = F.mse_loss(pruned_input, orignal_input)
            return pruned_input

    def set_structure_value(self, value):
        self.gate_f = value
        #self.gate_f.requires_grad = True
    # def get_grads(self):
    #     return self.gate_f.grad.data
    # def mse_collector(self):
    def collect_mse(self):
        return self.mse

    def reset_value(self):
        self.gate_f = torch.ones(self.width )

def tanh_gradient(x, T=4, b=0.5):
    value_pos = torch.exp(T*(x-b))
    value_neg = torch.exp(-T*(x-b))

    return 2*T/(value_pos+value_neg)

class AC_layer(nn.Module):
    def __init__(self, num_class=10):
        super(AC_layer, self).__init__()
        #self.weights = nn.Parameter(torch.ones(width)*0.3017 + 0.1*torch.randn(width))
        self.fc = nn.Linear(num_class, num_class)
        #self.register_buffer('gate_function',torch.sigmoid(self.weights))
        #self.tanh = nn.Tanh()
        self.num_class = num_class
    def forward(self, input):
        b_size, n_c, w, h = input.size()
        input = input.view(b_size, 1, -1)
        input = F.adaptive_avg_pool1d(input, self.num_class)
        out = self.fc(input.squeeze())
        return out

class custom_STE(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, train, grad_w=1):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        train = train
        ctx.grad_w = grad_w
        if train is True:
            ctx.save_for_backward(input)
            input_clone = input.clone()
            # input_clone[input >= 0.5] = 1
            # input_clone[input < 0.5] = 0

            input_clone = prob_round_torch(input_clone)
        else:
            ctx.save_for_backward(input)
            input_clone = input.clone()
            input_clone[input >= 0.5] = 1
            input_clone[input < 0.5] = 0

        return input_clone.float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        #print(grad_input.mean())

        #mul = tanh_gradient(input)
        #grad_input = grad_input*mul

        #print(grad_input.mean())
        grad_input[input > 1] = 0
        grad_input[input < 0] = 0
        gw = ctx.grad_w
        #print(gw)
        if grad_input.is_cuda and type(gw) is not int:
            gw = gw.cuda()

        return grad_input*gw, None, None

class custom_grad_weight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grad_w=1):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        #train = train
        ctx.grad_w = grad_w

        input_clone = input.clone()
        return input_clone.float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = grad_output.clone()

        gw = ctx.grad_w
        # print(gw)
        if grad_input.is_cuda and type(gw) is not int:
            gw = gw.cuda()

        return grad_input * gw, None, None


# def prob_round(x):
#
#     stochastic_round = np.random.rand(10) < (x - x.astype(int))
#     round_value = np.floor(x)+stochastic_round
#     return round_value

def prob_round_torch(x):
    if x.is_cuda:
        stochastic_round = torch.rand(x.size(0)).cuda() < x
    else:
        stochastic_round = torch.rand(x.size(0)) < x
    #round_value = x.floor() + stochastic_round
    return stochastic_round