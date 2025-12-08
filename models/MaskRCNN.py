# Native Imports
import os, time

# Library Imports
# import tqdm
import torch
import torch.nn.functional as F

# Custom Imports
from utils.utils import binary_accuracy, save_hist_graphs
from utils.attributes import hist_attributes, train_attributes, predict_attributes

MaskRCNN_attributes = {
    'pixel_confidence_thresh': 'Float',
    'mask_confidence_thresh': 'Float'
}

class MaskRCNN():
    def __init__(self, net, device, init_from_chkpt=False, **kwargs):
        self.net = net
        self.device = device
        
        self.net.to(self.device)
        
        for attr, value in kwargs.items():
            if attr in list(hist_attributes.keys()): continue
            setattr(self, attr, value)
            
        if init_from_chkpt:
            if not hasattr(self, 'checkpoint'):
                raise ValueError(f'No checkpoint provided for initializing MaskRCNN(init_from_chkpt=True, checkpoint=...)')
            
            self.net.load_state_dict(kwargs['checkpoint'])
            
            for hist in hist_attributes.keys():
                setattr(self, hist, kwargs[hist])
        
        else:
            for hist in hist_attributes.keys():
                setattr(self, hist, [])

    
    def train(self, epochs):
        self._confirm_attributes({**MaskRCNN_attributes, **train_attributes}) 
        
        best_epoch_accuracy = 0.0
        best_validation_accuracy = 0.0
        best_validation_loss = 0
        best_F1 = 0.0
        best_IoU = 0.0

        begin_time = time.time()
        current_time = time.localtime(begin_time)
        date_str = time.strftime("%d-%m-%Y_%H-%M", current_time)
        
        scaler = torch.amp.GradScaler() if self.device == 'cuda' and self.use_scaler else None
        
        print(f'Training started at {date_str}')
        
        for epoch in range(epochs):
            
            epoch_loss = 0  
            epoch_accuracy = 0
            epoch_precision = 0
            epoch_recall = 0
            epoch_f1 = 0
            epoch_iou = 0
            
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
                
                if i % self.print_freq == 0:
                    self.net.eval()
                    with torch.no_grad():
                        predictions = self.net(images)
                        
                        thresh_idx = (predictions[0]['scores'] > self.mask_confidence_thresh).nonzero().squeeze(1)
                        mask = torch.einsum('bcij->cij', (predictions[0]['masks'][thresh_idx] > self.pixel_confidence_thresh).float()) # assumes batch_size is 1
                        accuracy, precision, recall, f1, iou = binary_accuracy(mask, targets[0]['masks'].squeeze(0))
                        
                        epoch_accuracy += accuracy
                        epoch_precision += precision
                        epoch_recall += recall
                        epoch_f1 += f1
                        epoch_iou += iou
                        
                    print(f'[Train] [Epoch {epoch}] [Iter. {i}] [Learning Rate {self.optimizer.param_groups[0]['lr']:.2e}] [Loss {train_loss.item():.4f}, Accuracy {accuracy * 100:.2f}%, F1 {f1:.3f}]')
                    
            epoch_loss /= len(self.train_loader) / self.print_freq
            epoch_accuracy /= len(self.train_loader) / self.print_freq
            epoch_precision /= len(self.train_loader) / self.print_freq
            epoch_recall /= len(self.train_loader) / self.print_freq
            epoch_f1 /= len(self.train_loader) / self.print_freq
            epoch_iou /= len(self.train_loader) / self.print_freq
            
            self.train_loss_hist.append(epoch_loss)
            self.train_accuracy_hist.append(epoch_accuracy)
            self.train_precision_hist.append(epoch_precision)
            self.train_recall_hist.append(epoch_recall)
            self.train_f1_hist.append(epoch_f1)
            self.train_iou_hist.append(epoch_iou)
            
            # VALIDATE
            if epoch % self.val_freq == 0:
                val_loss, val_accuracy, val_F1, val_IoU = self._validate(epoch)
                
                if val_F1 > best_F1:
                    best_validation_loss = val_loss
                    best_validation_accuracy = val_accuracy
                    best_F1 = val_F1
                    best_IoU = val_IoU
                    
                    self._save_model_incrementaly(epoch, val_accuracy, val_F1, val_IoU, date_str)
                
                if epoch_accuracy > best_epoch_accuracy: best_epoch_accuracy = epoch_accuracy
                
                print(f'[Epoch {epoch}/{epochs}, Exec Time {time.time() - begin_time:.2f}s] [Best] [vAccuracy {best_validation_accuracy * 100:.2f}%, vLoss {best_validation_loss:.4f}, F1 {best_F1:.3f}, IoU {best_IoU:.3f}]')
            
            if self.device == 'cuda': torch.cuda.empty_cache()
        
        self.train_time = time.time() - begin_time
        print(f'Training complete in {self.train_time // 60:.0f}m {self.train_time % 60:.0f}s')
        
        # if self.graph_hists:
        #     for hist in hist_attributes.keys():
        #         make_hist_graphs(getattr(self, hist))
        
        self._save_model(date_str, epochs)
        
    
    def predict(self):
        self._confirm_attributes({**MaskRCNN_attributes, **predict_attributes})
        
        predictions = []
        t_batch_avg = 0.0
        
        self.net.eval()
        with torch.no_grad():
            for i, (image_batch, targets_batch) in enumerate(self.test_loader):
                t = time.time()
                preds = self.net(image_batch.to(self.device))
                t_batch_avg += (time.time() - t)
                
                for pred in preds:
                    thresh_idx = (pred['scores'] > self.mask_confidence_thresh).nonzero().squeeze(1)
                    predictions.append({'mask': torch.einsum('bcij->cij', (pred['masks'][thresh_idx] > self.pixel_confidence_thresh)).bool().cpu()})
        
        t_batch_avg /= len(self.test_loader)
        
        return predictions, t_batch_avg
    
    def _validate(self, epoch):
        # the following code is written assuming that batch size is 1
        if self.device == 'cuda': torch.cuda.empty_cache()
        
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
                
                thresh_idx = (predictions[0]['scores'] > self.mask_confidence_thresh).nonzero().squeeze(1)
                mask = torch.einsum('bcij->cij', predictions[0]['masks'][thresh_idx])
                loss = F.binary_cross_entropy_with_logits(mask, targets[0]['masks'].float())
                
                mask = torch.einsum('bcij->cij', (predictions[0]['masks'][thresh_idx] > self.pixel_confidence_thresh).float())
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
    
    def _confirm_attributes(self, attr_list):
        for attr, type in attr_list.items():
            if not hasattr(self, attr):
                raise ValueError(f'No value provided for {attr} (Expected type: {type})')
            
    def _save_model_incrementaly(self, epoch, val_accuracy, val_F1, val_IoU, date_str):
        checkpoint_dir = self.checkpoint_dir + '/MaskRCNN' + f'/{self.dataset}'
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.net.state_dict(),
                'pixel_confidence_thresh': self.pixel_confidence_thresh,
                'mask_confidence_thresh': self.mask_confidence_thresh,
                
                'val_accuracy': val_accuracy,
                'val_F1': val_F1,
                'val_IoU': val_IoU,
                
                'date_str': date_str,
            },
            os.path.join(
                checkpoint_dir, 
                f"MaskRCNN_{date_str}_E{epoch}_vA{val_accuracy * 100:.2f}_vF{val_F1:.3f}_vIoU{val_IoU:.3f}.pth"
            )
        )
    
    def _save_model(self, date_str, epochs):
        checkpoint_dir = self.checkpoint_dir + '/MaskRCNN' + f'/{self.dataset}'
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        
        torch.save(
            {
                'epoch': 'final',
                'model_state_dict': self.net.state_dict(),
                'pixel_confidence_thresh': self.pixel_confidence_thresh,
                'mask_confidence_thresh': self.mask_confidence_thresh,
                'train_time': self.train_time,
                
                # metrics vs epoch history
                'train_loss_hist': self.train_loss_hist,
                'train_accuracy_hist': self.train_accuracy_hist,
                'train_precision_hist': self.train_precision_hist,
                'train_recall_hist': self.train_recall_hist,
                'train_f1_hist': self.train_f1_hist,
                'train_iou_hist': self.train_iou_hist,
                
                'date_str': date_str,
            },
            os.path.join(
                checkpoint_dir, 
                f"MaskRCNN_{date_str}_Epochs{epochs}.pth"
            )
        )