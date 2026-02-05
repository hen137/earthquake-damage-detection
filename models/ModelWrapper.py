# Native Imports
import os, time

# Library Imports
# import tqdm
import torch

# Local Imports
from utils.utils import binary_accuracy, save_hist_graphs
from utils.attributes import hist_attributes, train_attributes, predict_attributes

class ModelBase:
    def _get_attributes(self):
        '''
        Must return a dictionary of model-specific attributes and their types
        e.g., {'pixel_confidence_thresh': 'Float'}
        '''
        raise NotImplementedError('Child class must implement _get_attributes() method.')
    
    def _set_train_mode(self):
        '''
        Must set the model to training mode
        '''
        raise NotImplementedError('Child class must implement _set_train_mode() method')
    
    def _set_eval_mode(self):
        '''
        Must set the model to evaluation mode
        '''
        raise NotImplementedError('Child class must implement _set_eval_mode() method')
    
    def _prediction(self, images, targets):
        '''
        Must return loss and predicted mask for each image/target pair
        '''
        raise NotImplementedError('Child class must implement _prediction(self, images, targets) method')
    
    # def _threshold_mask(self, pred_mask):
    #     '''
    #     Must return thresholded mask from predicted mask
    #     '''
    #     raise NotImplementedError('Child class must implement _threshold_mask(self, pred_mask) method')
    
    def _get_parameters(self):
        '''
        Must return model parameters as a state_dict
        '''
        raise NotImplementedError('Child class must implement _get_parameters(self) method')
    
    def __init__(self, name, device, init_from_checkpoint=False, **kwargs):
        self.name = name
        self.device = device
        
        for attr, value in kwargs.items():
            if attr in list(hist_attributes.keys()): continue
            setattr(self, attr, value)
            
        if init_from_checkpoint:
            if not hasattr(self, 'checkpoint'):
                raise ValueError(f'No checkpoint provided for initializing MaskRCNN(init_from_chkpt=True, checkpoint=...)')
            
            for hist in hist_attributes.keys():
                setattr(self, hist, kwargs[hist])
        
        else:
            for hist in hist_attributes.keys():
                setattr(self, hist, [])
    
    def train(self, epochs):
        self._confirm_attributes({**self._get_attributes(), **train_attributes})

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
        
                self._set_train_mode()
                self.optimizer.zero_grad()
                with torch.autocast(self.device):
                    train_loss, pred_mask = self._prediction(images, targets)
                
                if scaler:
                    scaler.scale(train_loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    train_loss.backward()
                    self.optimizer.step()
                
                self.lr_scheduler.step()
                
                epoch_loss += train_loss.item()
                
                if i % self.print_freq == 0:
                    #TODO: comply with batches
                    accuracy, precision, recall, f1, iou = binary_accuracy(pred_mask, targets[0]['masks'].squeeze(0))
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
                start_time = time.time()
                val_loss, val_accuracy, val_F1, val_IoU = self._validate()
                
                print(f'[Validation] [Epoch {epoch}, Exec Time {time.time() - start_time:.2f}s] [Loss {val_loss:.4f}, Accuracy {accuracy * 100:.2f}%, F1 {val_F1:.3f}, IoU {val_IoU:.3f}]')

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
        self._confirm_attributes({**self._get_attributes(), **predict_attributes})
        
        predictions = []
        t_batch_avg = 0.0
        
        self._set_eval_mode()
        with torch.no_grad():
            for i, (image_batch, targets_batch) in enumerate(self.test_loader):
                t = time.time()
                _, pred_masks = self._prediction(image_batch, targets_batch)
                t_batch_avg += time.time() - t
                
                for pred_mask in pred_masks:
                    predictions.append({'mask': pred_mask})
        
        t_batch_avg /= len(self.test_loader)
        
        return predictions, t_batch_avg
    
    def _validate(self):
        if self.device == 'cuda': torch.cuda.empty_cache()

        val_loss = 0
        accuracy = 0
        F1 = 0
        IoU = 0

        self._set_eval_mode()
        with torch.no_grad():
            for i, (images, targets) in enumerate(self.validation_loader):
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
                
                loss, pred_masks = self._prediction(images, targets)
                
                #TODO: comply with batches
                acc, _, _, f1, iou = binary_accuracy(pred_masks, targets[0]['masks'].squeeze(0))
                
                val_loss += loss
                accuracy += acc
                F1 += f1
                IoU += iou

            val_loss /= len(self.validation_loader)
            accuracy /= len(self.validation_loader)
            F1 /= len(self.validation_loader)
            IoU /= len(self.validation_loader)
        
        return val_loss, accuracy, F1, IoU
    
    def _confirm_attributes(self, attr_list):
        for attr, type in attr_list.items():
            if not hasattr(self, attr):
                raise ValueError(f'No value provided for {attr} (Expected type: {type})')
            
    def _save_model_incrementaly(self, epoch, val_accuracy, val_F1, val_IoU, date_str):
        checkpoint_dir = self.checkpoint_dir + f'/{self.name}/{self.dataset}'
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self._get_parameters(),
                
                **{attr_name: getattr(self, attr_name) for attr_name in self._get_attributes().keys()},
                
                'val_accuracy': val_accuracy,
                'val_F1': val_F1,
                'val_IoU': val_IoU,
                
                'date_str': date_str,
            },
            os.path.join(
                checkpoint_dir, 
                f"{self.name}_{date_str}_E{epoch}_vA{val_accuracy * 100:.2f}_vF{val_F1:.3f}_vIoU{val_IoU:.3f}.pth"
            )
        )
    
    def _save_model(self, date_str, epochs):
        checkpoint_dir = self.checkpoint_dir + f'/{self.name}/{self.dataset}'
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        
        torch.save(
            {
                'epoch': 'final',
                'model_state_dict': self._get_parameters(),
                'train_time': self.train_time,
                
                **{attr_name: getattr(self, attr_name) for attr_name in self._get_attributes().keys()},
                
                # metrics vs epochs history
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
                f"{self.name}_{date_str}_Epochs{epochs}.pth"
            )
        )