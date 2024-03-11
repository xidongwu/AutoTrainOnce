import torch
import time
from tqdm import tqdm
import copy
from utils import loss_fn_kd, display_structure, loss_label_smoothing, display_structure_hyper, LabelSmoothingLoss, one_hot, cross_entropy_onehot_target, mixup_func
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from imgnet_models.sampler import ImbalancedAccuracySampler
import torch.nn as nn

def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

def simple_train(train_loader, model, criterion, optimizer, epoch, args):
    alpha = 0.5
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(input)
        if args.ls:
            #loss=loss_label_smoothing(output, target, T=3, alpha=0.3)
            smooth_loss = LabelSmoothingLoss(classes=1000)(output, target)
            loss = smooth_loss
            # if args.scratch:
            #     loss = smooth_loss
            # else:
            #     entropy_loss = criterion(output, target)
            #     loss = alpha*smooth_loss+(1-alpha)*entropy_loss
        else:
            loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)


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

def one_step_net(inputs, targets, net, masks, args):

    targets = one_hot(targets, num_classes=1000, smoothing_eps=0.1)

    if args.mix_up:
        inputs, targets = mixup_func(inputs, targets)

    net.train()
    sel_loss = torch.tensor(0)
    outputs = net(inputs)

    loss = cross_entropy_onehot_target(outputs, targets)


    # if hasattr(args, 'reg_align') and args.reg_align:
    if hasattr(net,'module'):
        weights = net.module.get_weights()
    else:
        weights = net.get_weights()

    ##### Group lasso remove --- Optional 
    with torch.no_grad():
        sel_loss = args.selection_reg(weights, masks)
    # loss = sel_loss

    loss.backward()

    return sel_loss, loss, outputs

def soft_train(train_loader, model, hyper_net, criterion, valid_loader, optimizer, optimizer_hyper, epoch, cur_maskVec, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    alignments = AverageMeter('AlignmentLoss', ':.4e')
    hyper_losses = AverageMeter('HyperLoss', ':.4e')
    res_losses = AverageMeter('ResLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    h_top1 = AverageMeter('HAcc@1', ':6.2f')
    h_top5 = AverageMeter('HAcc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, alignments, top1,
                             top5, hyper_losses, res_losses, h_top1, h_top5, prefix="Epoch: [{}]".format(epoch))
    # resource_loss = 0
    # hyper_loss = 0

    model.train()
    # sumres_loss = 0
    end = time.time()
    lmdValue = 0

    if epoch < int((args.epochs - 5)/ 2) + 5:
        with torch.no_grad():
            hyper_net.eval()
            vector = hyper_net()  # a vector 
            return_vect = copy.deepcopy(vector)
            masks = hyper_net.vector2mask(vector)
    else:
        print(">>>>> Using fixed mask")
        return_vect = copy.deepcopy(cur_maskVec)
        vector = cur_maskVec
        masks = hyper_net.vector2mask(vector)

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad()
        sel_loss, loss, outputs = one_step_net(input, target, model, masks, args)

        optimizer.step()

        ## project
        if epoch >= args.start_epoch_gl:
            if args.lmd > 0:
                lmdValue = args.lmd
            elif args.lmd == 0:
                if epoch < int((args.epochs - 5)/ 2):
                    lmdValue = 10
                else:
                    lmdValue = 1000 #10000000000

            with torch.no_grad():
                if args.project == 'gl':
                    if hasattr(model, 'module'):
                        model.module.project_wegit(hyper_net.transfrom_output(vector), lmdValue, model.lr)
                    else:
                        model.project_wegit(hyper_net.transfrom_output(vector), lmdValue, model.lr)
                elif args.project == 'oto':
                    model.oto(hyper_net.transfrom_output(vector))

        acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        alignments.update(sel_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # if epoch >= args.start_epoch_hyper:
        if epoch >= args.start_epoch_hyper and (epoch < int((args.epochs - 5)/ 2) + 5):
            if (i + 1) % args.hyper_step == 0:
                val_inputs, val_targets = next(iter(valid_loader))
                if args.gpu is not None:
                    val_inputs = val_inputs.cuda(args.gpu, non_blocking=True)
                val_targets = val_targets.cuda(args.gpu, non_blocking=True)

                optimizer_hyper.zero_grad()

                masks, h_loss, res_loss, hyper_outputs = one_step_hypernet(val_inputs, val_targets, model, hyper_net,
                                                                           args)
                optimizer_hyper.step()

                h_acc1, h_acc5 = accuracy(hyper_outputs, val_targets, topk=(1, 5))
                h_top1.update(h_acc1[0], val_inputs.size(0))
                h_top5.update(h_acc5[0], val_inputs.size(0))
                hyper_losses.update(h_loss.item(), val_inputs.size(0))
                res_losses.update(res_loss.item(),val_inputs.size(0))

                if hasattr(model, 'module'):
                    model.module.reset_gates()
                else:
                    model.reset_gates()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.print(i)


    print("Project Lmd in this Epoch:", lmdValue)
    if epoch >= args.start_epoch:
        if epoch < int((args.epochs - 5)/ 2) + 5: 
            with torch.no_grad():
                    # resource_constraint.print_current_FLOPs(hyper_net.resource_output())
                hyper_net.eval()
                vector = hyper_net()
                display_structure(hyper_net.transfrom_output(vector))
        else:
            display_structure(hyper_net.transfrom_output(vector))


    return return_vect

def simple_validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(tqdm(val_loader)):
            # if args.gpu is not None:
            #     input = input.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def validateMask(val_loader, model, cur_maskVec, criterion, args):

    vector = cur_maskVec

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    if hasattr(model,'module'):
        model.module.set_vritual_gate(vector)
    else:
        model.set_vritual_gate(vector)

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    # reset gates to 1
    if hasattr(model, 'module'):
        model.module.reset_gates()
    else:
        model.reset_gates()

    return top1.avg



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2