import torch
# from only_train_once import OTO
import torchvision
from tqdm import tqdm
import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from imgnet_models.resnet_gate import my_resnet50, my_resnet101, my_resnet34
from imgnet_models.hypernet import HyperStructure
from warm_up.Warmup_Sch import GradualWarmupScheduler
from alignment_functions import SelectionBasedRegularization, SelectionBasedRegularization_MobileNet, SelectionBasedRegularization_MobileNetV3
from torch.utils.data.dataset import random_split

from repeat_dataloader import MultiEpochsDataLoader
from utils import *
from train import *

#############
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
                    # choices=model_names,
                    # help='model architecture: ' +
                    #     ' | '.join(model_names) +
                    #     ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--lmd', default=1e-4, type=float, metavar='W', help='group lasso lamd (default: 1e-4)',
                    dest='lmd')
parser.add_argument('--epsilon', default=0.1, type=float, metavar='M',
                    help='epsilon in OTO')
parser.add_argument('--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--p', default=0.37, type=float,
                    help='Pruning Rate')
parser.add_argument('--stage', default='train-gate', type=str,
                    help='Which stage to choose')
parser.add_argument('--ls', default=True, type=str2bool)
parser.add_argument('--mix_up', default=False, type=str2bool)
parser.add_argument('--gates', default=2, type=int)
parser.add_argument('--pruning_method', default='flops', type=str)
parser.add_argument('--base', default=3.0, type=float)
parser.add_argument('--interval', default=30, type=int)
parser.add_argument('--base_p', default=1.0, type=float)
parser.add_argument('--scratch', type=str2bool, default=False)
parser.add_argument('--bn_decay',type=str2bool, default=False)
parser.add_argument('--cos_anneal',type=str2bool, default=False)
parser.add_argument('--opt_name',type=str,default='SGD')
parser.add_argument('--project',type=str,default='gl')
parser.add_argument('--hyper_step', default=20, type=int)
parser.add_argument('--grad_mul', default=10.0, type=float)
parser.add_argument('--reg_w', default=4.0, type=float)  # 4.0 
parser.add_argument('--gl_lam', default=0.0001, type=float)
parser.add_argument('--start_epoch_hyper', default=20, type=int)
parser.add_argument('--start_epoch_gl', default=100, type=int)
parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
parser.add_argument('--auto-augment', default=None, help='auto augment policy (default: None)')
parser.add_argument('--random-erase', default=0.0, type=float, help='random erasing probability (default: 0.0)')

best_acc1 = 0
args = parser.parse_args()

torch.manual_seed(args.seed)
cudnn.deterministic = True
print(args)
############

args.gates = 2
gate_string = '_2gates'

model = my_resnet50()
args.model_name = 'resnet'
args.block_string = model.block_string
print_model_param_flops(model)

width, structure = model.count_structure()

hyper_net = HyperStructure(structure=structure, T=0.4, base=3,args=args)
hyper_net.cuda()
tmp = hyper_net()
print("Mask", tmp, tmp.size())

args.structure = structure
sel_reg = SelectionBasedRegularization(args)

if args.pruning_method == 'flops':
    size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_resnetbb(model)
    resource_reg = Flops_constraint_resnet_bb(args.p, size_kernel, size_out, size_group, size_inchannel,
                                           size_outchannel, w=args.reg_w, HN=True,structure=structure)                
model = torch.nn.DataParallel(model).cuda()

args.selection_reg = sel_reg
args.resource_constraint = resource_reg
# model = torchvision.models.resnet50(weights=True).cuda()
# model = torch.nn.DataParallel(model).cuda()
# dummy_input = torch.zeros(1, 3, 224, 224).cuda() # one data with H*W*C = 224 * 224 * 3
# oto = OTO(model=model, dummy_input=dummy_input)

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# data_dir = "/data/ILSVRC2012" # Change to your own imagenet path
data_dir = args.data #"/p/federatedlearning/data/ILSVRC2012" # Change to your own imagenet path
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'val')
batch_size = 128

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),   # improve 0.4 - 0.5
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
val_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=transform_test)

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
testloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

import numpy as np
import torch.nn.functional as F

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
    # return input.mul_(gamma).add_(1 - gamma, perm_input), target.mul_(gamma).add_(1 - gamma, perm_target)
    return input.mul_(gamma).add_(perm_input, alpha=1 - gamma), target.mul_(gamma).add_(perm_target, alpha=1 - gamma)

# from utils.utils import check_accuracy
def accuracy_topk(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).view(-1).float().sum(0, keepdim=True)
        res.append(correct_k)
    return res


def check_accuracy(model, testloader, two_input=False):
    correct1 = 0
    correct5 = 0
    total = 0
    model = model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for X, y in tqdm(testloader):
            X = X.to(device)
            y = y.to(device)
            if two_input:
                y_pred = model.forward(X, X)
            else:
                y_pred = model.forward(X)
            total += y.size(0)

            prec1, prec5 = accuracy_topk(y_pred.data, y, topk=(1, 5))
            
            correct1 += prec1.item()
            correct5 += prec5.item()

    model = model.train()
    accuracy1 = correct1 / total
    accuracy5 = correct5 / total
    return accuracy1, accuracy5

def one_step_hypernet(inputs, targets, net, hyper_net, args):
    net.eval()
    hyper_net.train()

    vector = hyper_net() # 有分数
    if hasattr(net,'module'):
        net.module.set_vritual_gate(vector)
    else:
        net.set_vritual_gate(vector)

    outputs = net(inputs)

    res_loss = 2 * args.resource_constraint(hyper_net.resource_output())
    loss = nn.CrossEntropyLoss()(outputs, targets) + res_loss
    loss.backward()

    with torch.no_grad():
        hyper_net.eval()
        vector = hyper_net()
        masks = hyper_net.vector2mask(vector)

    return masks, loss, res_loss, outputs


###########
params_group = group_weight(hyper_net)
print("params_group = group_weight(hyper_net)", len(list(hyper_net.parameters())))
optimizer_hyper = torch.optim.AdamW(params_group, lr=1e-3, weight_decay=1e-2)
scheduler_hyper = torch.optim.lr_scheduler.MultiStepLR(optimizer_hyper, milestones=[int(0.98 * (args.epochs) / 2)], gamma=0.1)

label_smooth = True
mix_up = True
train_time = 2 # Mix-up requires longer training time for better convergence.
# ckpt_dir = './' # Checkpoint save directory

max_epoch = 120
# if not label_smooth:
#     criterion = torch.nn.CrossEntropyLoss()
# else:
criterion = cross_entropy_onehot_target
    
# Every 30 epochs, decay lr by 10.0
# optimizer = oto.dhspg(
#     variant='sgd', 
#     lr=0.1, 
#     target_group_sparsity=0.4,
#     weight_decay=0.0, # Some training set it as 1e-4.
#     first_momentum=0.9,
#     start_pruning_steps=15 * len(trainloader), # start pruning after 15 epochs. Start pruning at initialization stage.
#     epsilon=0.95)
optimizer  = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) 
num_classes = 1000

best_acc_1 = 0.0
params_group = group_weight(hyper_net)

model.lmd = args.lmd
print("====== model.lmd", model.lmd)

print("===================.  Training ============")
cnt = 0

ratio = (len(trainloader)/args.hyper_step)/len(trainloader)
print("Val gate rate %.4f" % ratio)
_, val_gate_dataset = random_split(
    train_dataset,
    lengths=[len(train_dataset) - int(ratio * len(train_dataset)), int(ratio * len(train_dataset))]
)
val_loader_gate = MultiEpochsDataLoader(
    val_gate_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True,)

args.start_epoch_gl = 15
args.start_epoch_hyper = 15

for epoch in range(max_epoch):
    f_avg_val = 0.0
    for t in range(train_time):
        cnt += 1
        print("Training cnt", cnt)

        for param_group in optimizer.param_groups:
            print("current_lr %.4f"%param_group["lr"])
        model.train()
        lr_scheduler.step()  
        scheduler_hyper.step()

        with torch.no_grad():
            hyper_net.eval()
            vector = hyper_net()  # a vector 
            masks = hyper_net.vector2mask(vector)

        for X, y in trainloader:

            X = X.cuda()
            y = y.cuda()
            with torch.no_grad():
                # if label_smooth and not mix_up:
                #     y = one_hot(y, num_classes=num_classes, smoothing_eps=0.1)

                # if not label_smooth and mix_up:
                #     y = one_hot(y, num_classes=num_classes)
                #     X, y = mixup_func(X, y)
                
                # if mix_up and label_smooth:
                y = one_hot(y, num_classes=num_classes, smoothing_eps=0.1)
                X, y = mixup_func(X, y)
    
            # y_pred = model.forward(X)
            y_pred = model(X)
            f = criterion(y_pred, y)
            optimizer.zero_grad()
            f.backward()
            f_avg_val += f
            optimizer.step()

            if epoch >= args.start_epoch_gl:
                with torch.no_grad():
                    if hasattr(model, 'module'):
                        model.module.project_wegit(hyper_net.transfrom_output(vector), args.lmd, model.lr)
                    else:
                        model.project_wegit(hyper_net.transfrom_output(vector), args.lmd, model.lr)


            if epoch >= args.start_epoch_hyper and (epoch < int(args.epochs / 2)):
                if (i + 1) % args.hyper_step == 0:
                    val_inputs, val_targets = next(iter(valid_loader))
                    if args.gpu is not None:
                        val_inputs = val_inputs.cuda(args.gpu, non_blocking=True)
                    val_targets = val_targets.cuda(args.gpu, non_blocking=True)

                    optimizer_hyper.zero_grad()

                    masks, h_loss, res_loss, hyper_outputs = one_step_hypernet(val_inputs, val_targets, model, hyper_net,
                                                                               args)
                    optimizer_hyper.step()

                    if hasattr(model, 'module'):
                        model.module.reset_gates()
                    else:
                        model.reset_gates()

        accuracy1, accuracy5 = check_accuracy(model, testloader)
        f_avg_val = f_avg_val.cpu().item() / len(trainloader)
        print("Epoch: {ep}, loss: {f:.2f}, acc1: {acc:.4f}"\
            .format(ep=epoch, f=f_avg_val, acc=accuracy1))
