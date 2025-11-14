# Native Imports
import os, argparse, random, time

# Library Imports
import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

# Custom Imports
from data import Levir_CD as Data

class SAM2():
    def __init__(self, train_loader, net, optimizer, val_loader, args):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.net = net
        self.optimizer = optimizer

    def adjust_lr(self, curr_iter, all_iter):
        return

    def train(self):
        return

    def validate(self, curr_epoch):
        return

class DeepLabV3():
    def __init__(self, train_loader, net, optimizer, val_loader, args):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.net = net
        self.optimizer = optimizer

    def adjust_lr(self, curr_iter, all_iter):
        return

    def train(self):
        return

    def validate(self, curr_epoch):
        return

class MaskRCNN():
    def __init__(self, train_loader, net, optimizer, val_loader, args):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.net = net
        self.optimizer = optimizer

    def adjust_lr(self, curr_iter, all_iter):
        return

    def train(self):
        return

    def validate(self, curr_epoch):
        return

def build_model(args, train_loader, val_loader):
    if args.encoder == 'SAM2':
        from models.SAM2 import SAM2 as Net
        TrainerClass = SAM2
    elif args.encoder == 'DeepLabV3+':
        from models.DeepLabV3 import DeepLabV3 as Net
        TrainerClass = DeepLabV3
    elif args.encoder == 'MaskRCNN':
        from models.MaskRCNN import MaskRCNN as Net
        TrainerClass = MaskRCNN
    else:
        raise ValueError(f'Unknown encoder type: {args.encoder}')

    net = Net()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=True
    )
    model = TrainerClass(train_loader, net, optimizer, val_loader, args)
    return net, model

def get_data_loaders(args):
    train_set = Data.RS(
        'train', random_crop=True, crop_nums=10,
        crop_size=args.crop_size, random_flip=True
    )
    val_set = Data.RS(
        'val', sliding_crop=False,
        crop_size=args.crop_size, random_flip=False
    )
    train_loader = DataLoader(
        train_set, batch_size=args.train_batch_size,
        num_workers=args.data_loader_num_workers, shuffle=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.val_batch_size,
        num_workers=args.data_loader_num_workers, shuffle=False
    )
    return train_loader, val_loader

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--encoder', required=True, choices=['SAM2', 'DeepLabV3+', 'MaskRCNN']) 
    parser.add_argument('--train_batch_size', required=False, default=128, type=int)
    parser.add_argument('--val_batch_size', required=False, default=32, type=int)
    parser.add_argument('--lr', required=False, default=0.1, type=float)
    parser.add_argument('--epochs', required=False, default=100, type=int)
    parser.add_argument('--gpu', required=False, default=True, action='store_true')
    parser.add_argument('--dev_id', required=False, default=0, type=int)
    parser.add_argument('--multi_gpu', required=False, default=None, type=str)
    parser.add_argument('--data_loader_num_workers', required=False, default=16, type=int)
    parser.add_argument('--weight_decay', required=False, default=5e-4, type=float)
    parser.add_argument('--momentum', required=False, default=0.9, type=float)
    parser.add_argument('--print_freq', required=False, default=200, type=int)
    parser.add_argument('--predict_step', required=False, default=5, type=int)
    parser.add_argument('--crop_size', required=False, default=512, type=int)
    parser.add_argument('--ftuned_dir', required=False, default='../models/finetuned')

    return parser.parse_args()

def main():
    args = parse_arguments()
    train_loader, val_loader = get_data_loaders(args)
    net, model = build_model(args, train_loader, val_loader)

    if args.multi_gpu:
        net = torch.nn.DataParallel(net, [int(id) for id in args.multi_gpu.split(',')])
    net.to(device=torch.device('cuda', int(args.dev_id)))

    print(f'Training {args.encoder} started')
    model.train()
    print(f'Training {args.encoder} finished')

if __name__ == '__main__':
    main()