# Native Imports
import os, time

# Library Imports
# import tqdm
import torch
import torch.nn.functional as F

# Custom Imports
from models.ModelWrapper import ModelBase
from utils.utils import binary_accuracy, save_hist_graphs
from utils.attributes import hist_attributes, train_attributes, predict_attributes

MaskRCNN_attributes = {
    'pixel_confidence_thresh': 'Float',
    'mask_confidence_thresh': 'Float'
}

class MaskRCNN(ModelBase):
    def __init__(self, net, device, init_from_checkpoint=False, **kwargs):
        super('MaskRCNN', device, init_from_checkpoint, **kwargs)
        
        self.net = net
        self.net.to(self.device)
        
        if init_from_checkpoint:
            self.net.load_state_dict(kwargs['checkpoint'])
    
    def _get_attributes(self):
        return MaskRCNN_attributes
    
    def _set_train_mode(self):
        self.model.train()
        
    def _set_eval_mode(self):
        self.model.eval()
    
    def _prediction(self, images, targets):
        return loss, pred_mask.detach()
        
    def train(self, epochs):
                with torch.autocast(self.device):
                    losses = self.net(images, targets)
                    train_loss = sum([loss for loss in losses.values()])

                
                if i % self.print_freq == 0:
                    self.net.eval()
                    with torch.no_grad():
                        predictions = self.net(images)
                        
                        thresh_idx = (predictions[0]['scores'] > self.mask_confidence_thresh).nonzero().squeeze(1)
                        mask = torch.einsum('bcij->cij', (predictions[0]['masks'][thresh_idx] > self.pixel_confidence_thresh).float()) # assumes batch_size is 1
                        
    def predict(self):
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
                
                predictions = self.net(image)
                
                thresh_idx = (predictions[0]['scores'] > self.mask_confidence_thresh).nonzero().squeeze(1)
                mask = torch.einsum('bcij->cij', predictions[0]['masks'][thresh_idx])
                loss = F.binary_cross_entropy_with_logits(mask, targets[0]['masks'].float())
                
                mask = torch.einsum('bcij->cij', (predictions[0]['masks'][thresh_idx] > self.pixel_confidence_thresh).float())
                acc, _, _, f1, iou = binary_accuracy(mask, targets[0]['masks'].squeeze(0))
                
    