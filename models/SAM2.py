# Native Imports
import os, time

# Library Imports
# import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom Imports
from utils.utils import binary_accuracy

class SAM2():
    def __init__(self, args, net, device, train_loader=None, val_loader=None, test_loader=None, optimizer=None, lr_scheduler=None):
        self.args = args
        self.net = net
        self.device = device
        self.train_loader = train_loader
        self.validation_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.net.to(self.device)

    def train(self):
        if not self.train_loader:
            raise ValueError(f'No value provided for train_loader')
        elif not self.optimizer:
            raise ValueError(f'No value provided for optimizer')
        elif not self.lr_scheduler:
            raise ValueError(f'No value provided for lr_scheduler')
        
        

    def validate(self, epoch):
        if not self.validation_loader:
            raise ValueError(f'No value provided for validation_loader')
        
        

        return val_loss, accuracy, F1, IoU
    
    def predict(self):
        
        
        return images, predictions, targets
    
    def save_model(self, epoch, val_accuracy, val_F1, val_IoU, date_str):
        checkpoint_dir = self.args.chkpt_dir + '/MaskRCNN' + f'/{self.args.dataset}'
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.net.state_dict(),
                'val_accuracy': val_accuracy,
                'val_F1': val_F1,
                'val_IoU': val_IoU,
                'date_str': date_str
            },
            os.path.join(
                checkpoint_dir, 
                f"{self.args.model}_e{epoch}_OA{val_accuracy * 100:.2f}_F{val_F1:.3f}_IoU{val_IoU:.3f}_{date_str}.pth"
            )
        )