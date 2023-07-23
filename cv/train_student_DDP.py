"""
the general training framework
"""

from __future__ import print_function

import random
import numpy as np
import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import distributed as dist

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser
from models.adapter import Adapter

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.cifar100 import get_cifar100_dataloaders_fewshot
from dataset.otherdataset import fc100
from dataset.imagenet import get_imagenet_dataloader

from helper.util import adjust_learning_rate, LabelSmoothingCrossEntropy

from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss

from helper.loops import train_distill as train, validate
from helper.pretrain import init


def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=11, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    parser.add_argument('--port', type=str, default='5711', help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150, 180, 210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'],
                        help='dataset')

    # model:
    parser.add_argument('--model_s', type=str, default='resnet20',
                        choices=['resnet8', 'resnet14', 'resnet18', 'resnet20', 'resnet32', 'resnet44', 'resnet56',
                                 'resnet110'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')
    parser.add_argument('-c', '--cigma', type=float, default=0, help='weight for teacher losses')

    parser.add_argument('--adapter_type', type=int, default=0, help='weight for teacher losses')

    parser.add_argument('--use_adapter', type=bool, default=False, help='whether to use adapter')
    parser.add_argument('--adapter_lr', type=float, default=2e-5, help='whether to use adapter')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=6, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    parser.add_argument('--seed', default=44, type=int)

    opt = parser.parse_args()

    torch.set_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    # set the path according to the environment
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    print('loading teacher from ', model_path)
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    print('==> done')
    return model


def main():
    best_acc = 0

    opt = parse_option()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = opt.port
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    torch.distributed.init_process_group(backend="nccl", rank=-1)
    local_rank = torch.distributed.get_rank()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data, train_sampler, test_sampler = get_cifar100_dataloaders(rank=local_rank,
                                                                                                 worldsize=dist.get_world_size(),
                                                                                                 batch_size=opt.batch_size,
                                                                                                 num_workers=opt.num_workers,
                                                                                                 is_instance=True)
        n_cls = 100
    elif opt.dataset == 'imagenet':
        train_loader, val_loader, n_data, train_sampler, test_sampler = get_imagenet_dataloader(rank=local_rank,
                                                                                                worldsize=dist.get_world_size(),
                                                                                                batch_size=opt.batch_size,
                                                                                                num_workers=opt.num_workers,
                                                                                                is_instance=True)
        n_cls = 1000

    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)
    print(device)
    model_t.to(device)
    model_s.to(device)

    print(opt.model_s, opt.model_t)

    model_t = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_t)
    model_s = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_s)
    model_t = nn.parallel.DistributedDataParallel(model_t, device_ids=[opt.local_rank],
                                                  output_device=opt.local_rank, find_unused_parameters=False,
                                                  broadcast_buffers=False)
    model_s = nn.parallel.DistributedDataParallel(model_s, device_ids=[opt.local_rank],
                                                  output_device=opt.local_rank, find_unused_parameters=False,
                                                  broadcast_buffers=False)

    if opt.dataset == 'imagenet':
        data = torch.randn(128, 3, 224, 224)
    else:
        data = torch.randn(2, 3, 32, 32)

    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    if not opt.use_adapter:
        adapter = None
    else:
        adapter = Adapter(64, 32, 100).cuda()
        print(feat_t[-1].shape)
        _ = adapter(feat_t[-1])

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    print('distil, ', opt.distill)

    # if opt.cigma>0:
    # trainable_list.append(model_t)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    if opt.distill == 'kd':
        criterion_kd = nn.MSELoss(reduce = True, size_average=True)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    # optimizer
    params = [{'params': model_s.parameters(), 'lr': opt.learning_rate}]
    if opt.cigma > 0:
        params += [{'params': model_t.parameters(), 'lr': 1e-4}]
        trainable_list.append(model_t)

    if adapter is not None:
        params += [{'params': adapter.parameters(), 'lr': opt.adapter_lr}]

    optimizer = optim.SGD(params,
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    avg_batch_time, avg_data_time = 0, 0
    # routine
    for epoch in range(1, opt.epochs + 1):
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()

        train_acc, train_loss = train(epoch, train_loader, module_list, \
                                                criterion_list,
                                                optimizer, opt, adapter)

        # train_loss = reduce_mean(train_loss, dist.get_world_size())
        train_loss /= dist.get_world_size()
        time2 = time.time()
        logger.log_value('epoch_time', time2 - time1, epoch)
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    avg_batch_time /= opt.epochs
    avg_data_time /= opt.epochs
    print('best accuracy:', best_acc)
    print('avg batch time', avg_batch_time)
    print('avg data time', avg_data_time)
    logger.log_value('avg_batch_time', avg_batch_time)
    logger.log_value('avg_data_time', avg_data_time)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
