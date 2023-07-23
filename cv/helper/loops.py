from __future__ import print_function, division

import sys
import time
import torch
import numpy as np
from .util import AverageMeter, accuracy, cos_sim
import torch.nn.functional as F
from scipy.spatial import distance


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def lable_smooth_target(target, alpha=0.1, K=100):
    logit = np.ones((target.shape[0], K))
    logit *= (alpha/K)
    for i in range(target.shape[0]):
        logit[i][target[i]] = 1-alpha
    return torch.Tensor(logit).cuda()


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt, adapter,
                  ):
    # set modules as train()
    for module in module_list:
        module.train()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    cnt = 1

    for idx, data in enumerate(train_loader):

        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True

        precent = target.shape[0]

        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)

        if opt.distill in ['ddgs']:
            pass
        elif opt.cigma == 0:
            with torch.no_grad():
                feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]
        else:
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        max_idx_s = torch.argmax(logit_s, dim=1)
        max_idx_t = torch.argmax(logit_t, dim=1)
        logit_t_cor = logit_t.clone().detach().cpu().numpy()
        # swap(logit_t_cor[i][max_idx_t[i]], logit_t_cor[i][target[i]])
        for i in range(max_idx_t.size()[0]):
            t = logit_t_cor[i][max_idx_t[i]]
            logit_t_cor[i][max_idx_t[i]] = logit_t_cor[i][target[i]]
            logit_t_cor[i][target[i]] = t
        logit_t_cor = torch.tensor(logit_t_cor).cuda()
        logit_s_cor = logit_s.clone().detach().cpu().numpy()
        for i in range(max_idx_s.size()[0]):
            t = logit_s_cor[i][max_idx_s[i]]
            logit_s_cor[i][max_idx_s[i]] = logit_s_cor[i][target[i]]
            logit_s_cor[i][target[i]] = t
        logit_s_cor = torch.tensor(logit_s_cor).cuda()

        # adapter
        if adapter is not None:
            adapter_t = adapter(feat_t[-1])
            logit_t = logit_t - adapter_t

        # div loss
        if opt.distill == 'kdcl':
            outputs = torch.zeros(size=(2, input.size(0), 1000), dtype=torch.float).cuda()
            outputs[0, ...] = logit_s
            outputs[1, ...] = logit_t
            # backward
            stable_out = outputs.mean(dim=0)
            stable_out = stable_out.detach()
            loss_div = criterion_div(logit_s, stable_out)
        elif opt.distill == 'ddgs':
            loss_div = 0
        elif opt.distill == 'label_smoothing':
            label_smoothing_target = lable_smooth_target(target)
            loss_div = criterion_div(logit_s, label_smoothing_target)
        else:
            loss_div = criterion_div(logit_s, logit_t_cor) + criterion_div(logit_s_cor, logit_t)
            #loss_div = criterion_div(logit_s, logit_t)

        # cls loss
        if opt.distill == 'ddgs':
            if logit_s[:precent].shape[0] != target.shape[0]:
                print(logit_s[:precent].shape, target.shape)
            loss_cls = criterion_cls(logit_s[:precent], target)
            loss_teacher = 0
        else:
            # cls + kl div
            loss_cls = criterion_cls(logit_s, target)
            loss_teacher = criterion_cls(logit_t, target)

        # other kd beyond KL divergence
        if opt.distill == 'kd' or opt.distill == 'dml' or opt.distill == 'kdcl' or opt.distill == 'label_smoothing':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        elif opt.distill == 'ddgs':
            layer_out = feat_s[-1][precent:]
            dis_layer_out = feat_s[-1][:precent]
            if layer_out.shape == dis_layer_out.shape:
                loss_kd = -criterion_kd(layer_out, dis_layer_out)
            else:
                loss_kd = 0
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        top_acc1, top_acc5 = accuracy(logit_t, target, topk=(1, 5))

        if opt.cigma > 0:
            loss += opt.cigma * loss_teacher
        if opt.use_adapter:
            loss += loss_teacher

        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))

        # ===================backward=====================

        loss.backward()
        
        if (idx+1) % 4 == 0:
            optimizer.step()
            optimizer.zero_grad()

        # ===================meters=====================
        end = time.time()

    return top1.avg, losses.avg

def validate(val_loader, model, criterion, opt, output_logit=False, adapter=None):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to evaluate mode
    model.eval()
    logit_sum = torch.zeros(100).cuda()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            if adapter is not None:
                feat_t, output = model(input, is_feat=True, preact=False)
                output += adapter(feat_t[-1])
            else:
                output = model(input)
            loss = criterion(output, target)

            
            # compute output
            if output_logit:
                logit_sum += torch.sum(output, dim=[0])

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    if output_logit:
        return top1.avg, top5.avg, losses.avg, logit_sum
    return top1.avg, top5.avg, losses.avg
