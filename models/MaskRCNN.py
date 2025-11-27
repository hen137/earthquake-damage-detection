# Native Imports
import os, time

# Library Imports
# import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom Imports
from utils.utils import binary_accuracy

class MaskRCNN():
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
            raise Exception(f'No value provided for train_loader')
        elif not self.optimizer:
            raise Exception(f'No value provided for optimizer')
        elif not self.lr_scheduler:
            raise Exception(f'No value provided for lr_scheduler')
        
        best_epoch_accuracy = 0.0
        best_validation_accuracy = 0.0
        best_validation_loss = 0
        best_F1 = 0.0
        best_IoU = 0.0

        begin_time = time.time()
        current_time = time.localtime(begin_time)
        date_str = time.strftime("%d-%m-%Y_%H-%M", current_time)
        
        scaler = torch.amp.GradScaler() if self.device == 'cuda' and self.args.use_scaler else None
        
        for epoch in range(self.args.epochs):
            if self.args.gpu: torch.cuda.empty_cache()
            
            epoch_loss = 0  
            epoch_accuracy = 0 
            
            for i, (images, targets) in enumerate(self.train_loader):
                images = images.to(self.device)
                if isinstance(targets, list):
                    for sample in targets:
                        for key in list(sample.keys()):
                            val = sample[key]
                            if isinstance(val, torch.Tensor):
                                sample[key] = val.to(self.device)
                elif isinstance(targets, dict):
                    for key in list(targets.keys()):
                        val = targets[key]
                        if isinstance(val, torch.Tensor):
                            targets[key] = val.to(self.device)
                
                
                self.net.train()
                self.optimizer.zero_grad()
                with torch.autocast(self.device):
                    losses = self.net(images, targets)
                    train_loss = sum([loss for loss in losses.values()])

                if scaler:
                    scaler.scale(train_loss).backward()
                    scaler.step(self.optimizer)
                    # old_scaler = scaler.get_scale()
                    scaler.update()
                    # new_scaler = scaler.get_scale()
                    # if new_scaler >= old_scaler:
                else:
                    train_loss.backward()
                    self.optimizer.step()

                self.lr_scheduler.step()
                
                epoch_loss += train_loss.item()
                
                if i % self.args.print_freq == 0:
                    self.net.eval()
                    with torch.no_grad():
                        predictions = self.net(images)
                        
                        thresh_idx = (predictions[0]['scores'] > self.args.mask_confidence_thresh).nonzero().squeeze(1)
                        mask = torch.einsum('bcij->cij', (predictions[0]['masks'][thresh_idx] > self.args.pixel_confidence_thresh).float()) # assumes batch_size is 1
                        accuracy, _, _, f1, _ = binary_accuracy(mask, targets[0]['masks'].squeeze(0))
                        
                        epoch_accuracy += accuracy
                        
                    print(f'[Train] [Epoch {epoch + 1}] [Iter. {i}] [Learning Rate {self.optimizer.param_groups[0]['lr']:.2e}] [Loss {train_loss.item():.4f}, Accuracy {accuracy * 100:.2f}%, F1 {f1:.3f}]')
                    
            epoch_loss /= len(self.train_loader)
            epoch_accuracy /= len(self.train_loader)
            
            # VALIDATE
            if epoch % self.args.val_freq == 0:
                val_loss, val_accuracy, val_F1, val_IoU = self.validate(epoch)
                
                if val_F1 > best_F1:
                    best_validation_loss = val_loss
                    best_validation_accuracy = val_accuracy
                    best_F1 = val_F1
                    best_IoU = val_IoU
                    
                    self.save_model(epoch, val_accuracy, val_F1, val_IoU, date_str)
                
                if epoch_accuracy > best_epoch_accuracy: best_epoch_accuracy = epoch_accuracy
                
                print(f'[Epoch {epoch}/{self.args.epochs}, Exec Time {time.time() - begin_time:.2f}s] [Best] [vAccuracy {best_validation_accuracy * 100:.2f}%, vLoss {best_validation_loss:.4f}, F1 {best_F1:.3f}, IoU {best_IoU:.3f}]')

    def validate(self, epoch):
        if not self.validation_loader:
            raise Exception(f'No value provided for validation_loader')
        
        # the following code is written assuming that batch size is 1
        if self.args.gpu: torch.cuda.empty_cache()
        
        start_time = time.time()

        val_loss = 0
        accuracy = 0
        F1 = 0
        IoU = 0

        self.net.eval()
        with torch.no_grad():
            for i, (image, targets) in enumerate(self.validation_loader):
                image = image.to(self.device)
                if isinstance(targets, list):
                    for sample in targets:
                        for key in list(sample.keys()):
                            val = sample[key]
                            if isinstance(val, torch.Tensor):
                                sample[key] = val.to(self.device)
                                
                elif isinstance(targets, dict):
                    for key in list(targets.keys()):
                        val = targets[key]
                        if isinstance(val, torch.Tensor):
                            targets[key] = val.to(self.device)
                
                predictions = self.net(image)
                
                thresh_idx = (predictions[0]['scores'] > self.args.mask_confidence_thresh).nonzero().squeeze(1)
                mask = torch.einsum('bcij->cij', predictions[0]['masks'][thresh_idx])
                loss = F.binary_cross_entropy_with_logits(mask, targets[0]['masks'].float())
                
                mask = torch.einsum('bcij->cij', (predictions[0]['masks'][thresh_idx] > self.args.pixel_confidence_thresh).float())
                acc, _, _, f1, iou = binary_accuracy(mask, targets[0]['masks'].squeeze(0))
                
                val_loss += loss.item()
                accuracy += acc
                F1 += f1
                IoU += iou

            val_loss /= len(self.validation_loader)
            accuracy /= len(self.validation_loader)
            F1 /= len(self.validation_loader)
            IoU /= len(self.validation_loader)
        
        print(f'[Validation] [Epoch {epoch}, Exec Time {time.time() - start_time:.2f}s] [Loss {val_loss:.4f}, Accuracy {accuracy * 100:.2f}%, F1 {F1:.3f}, IoU {IoU:.3f}]')

        return val_loss, accuracy, F1, IoU
    
    def predict(self):
        if not self.test_loader:
            raise Exception(f'No value provided for test_loader')
        images = []
        targets = []
        predictions = []
        
        self.net.eval()
        with torch.no_grad():
            for i, (image_batch, targets_batch) in enumerate(self.test_loader):
                preds = self.net(image_batch.to(self.device))
                
                for image, target in zip(image_batch, targets_batch):
                    images.append(image.cpu())
                    targets.append(target)
                    
                for pred in preds:
                    thresh_idx = (pred['scores'] > self.args.mask_confidence_thresh).nonzero().squeeze(1)
                    predictions.append({'mask': torch.einsum('bcij->cij', (pred['masks'][thresh_idx] > self.args.pixel_confidence_thresh)).bool().cpu()})
        
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