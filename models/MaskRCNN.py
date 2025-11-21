# Native Imports
import os, time

# Library Imports
# import tqdm
import torch
import torch.nn.functional as F

# Custom Imports
from utils.utils import binary_accuracy

class MaskRCNN():
    def __init__(self, args, net, device, train_loader, val_loader, optimizer, lr_scheduler):
        self.args = args
        self.net = net
        self.device = device
        self.train_loader = train_loader
        self.validation_loader = val_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.net.to(self.device)

    def train(self):
        best_epoch_accuracy = 0.0
        best_validation_accuracy = 0.0
        best_validation_loss = 1.0
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
                targets = {key: value.to(self.device) for key, value in zip(targets.keys(), targets.values())}
                
                self.net.train()
                with torch.autocast(self.device):
                    losses = self.net(images, [targets])
                    train_loss = sum([loss for loss in losses.values()]) # assumes batch_size is 1

                if scaler:
                    scaler.scale(train_loss).backward()
                    scaler.step(self.optimizer)
                    old_scaler = scaler.get_scale()
                    scaler.update()
                    new_scaler = scaler.get_scale()
                    if new_scaler >= old_scaler:
                        self.lr_scheduler.step()
                else:
                    train_loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                epoch_loss += train_loss.item()
                
                if i % self.args.print_freq == 0:
                    self.net.eval()
                    with torch.no_grad():
                        predictions = self.net(images)
                        
                        thresh_idx = (predictions[0]['scores'] > self.args.mask_confidence_thresh).nonzero().squeeze(1)
                        mask = torch.einsum('bcij->cij', (predictions[0]['masks'][thresh_idx] > self.args.pixel_confidence_thresh).float()) # assumes batch_size is 1
                        accuracy, _, _, _, _ = binary_accuracy(mask, targets['masks'].squeeze(0))
                        
                        epoch_accuracy += accuracy
                        
                    print(f'[Train] [Epoch {epoch + 1}] [Iter. {i}] [Learning Rate {self.optimizer.param_groups[0]['lr']:.2e}] [Loss {train_loss.item():.4f}, Accuracy {accuracy * 100:.2f}%]')
                    
            epoch_loss /= len(self.train_loader)
            epoch_accuracy /= len(self.train_loader)
            
            # VALIDATE
            if epoch % self.args.val_freq == 0:
                val_loss, val_accuracy, val_F1, val_IoU = self.validate(epoch)
                
                if val_F1 > best_F1: # Consider other metrics to determine 'best'
                    best_validation_loss = val_loss
                    best_validation_accuracy = val_accuracy
                    best_F1 = val_F1
                    best_IoU = val_IoU
                    
                    self.save_model(epoch, val_accuracy, val_F1, val_IoU, date_str)
                
                if epoch_accuracy > best_epoch_accuracy: best_epoch_accuracy = epoch_accuracy
                
                print(f'[Epoch {epoch + 1}/{self.args.epochs}, Exec Time {time.time() - begin_time:.2f}s] [Best] [vAccuracy {best_validation_accuracy * 100:.2f}%, vLoss {best_validation_loss:.4f}, F1 {best_F1:.3f}, IoU {best_IoU:.3f}]')

    def validate(self, epoch):
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
                targets = {key: value.to(self.device) for key, value in zip(targets.keys(), targets.values())}

                with torch.no_grad():
                    predictions = self.net(image)
                    
                    thresh_idx = (predictions[0]['scores'] > self.args.mask_confidence_thresh).nonzero().squeeze(1)
                    mask = torch.einsum('bcij->cij', predictions[0]['masks'][thresh_idx])
                    loss = F.binary_cross_entropy_with_logits(mask, targets['masks'].squeeze(0).float())
                    
                    mask = torch.einsum('bcij->cij', (predictions[0]['masks'][thresh_idx] > self.args.pixel_confidence_thresh).float())
                    acc, precision, recall, f1, iou = binary_accuracy(mask, targets['masks'].squeeze(0))
                
                val_loss += loss.item()
                accuracy += acc
                F1 += f1
                IoU += iou

            val_loss /= len(self.validation_loader)
            accuracy /= len(self.validation_loader)
            F1 /= len(self.validation_loader)
            IoU /= len(self.validation_loader)
        
        print(f'[Validation] [Epoch {epoch + 1}, Exec Time {time.time() - start_time:.2f}s] [Loss {val_loss:.4f}, Accuracy {accuracy * 100:.2f}%, F1 {F1:.3f}, IoU {IoU:.3f}]')

        return val_loss, accuracy, F1, IoU
    
    def save_model(self, epoch, val_accuracy, val_F1, val_IoU, date_str):
        if not os.path.exists(self.args.chkpt_dir): os.mkdir(self.args.chkpt_dir)
        
        torch.save(
            self.net.state_dict(), 
            os.path.join(
                self.args.chkpt_dir, 
                f"{self.args.model}_e{epoch}_OA{val_accuracy * 100:.2f}_F{val_F1:.3f}_IoU{val_IoU:.3f}_{date_str}.pth"
            )
        )