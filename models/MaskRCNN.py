# Native Imports
import time

# Library Imports
# import tqdm
import torch
import torch.nn.functional as F

# Local Imports
from models.ModelWrapper import ModelBase

MaskRCNN_attributes = {
    'pixel_confidence_thresh': 'Float',
    'mask_confidence_thresh': 'Float'
}

class MaskRCNN(ModelBase):
    def __init__(self, net, device, init_from_checkpoint=False, **kwargs):
        super().__init__('MaskRCNN', device, init_from_checkpoint=init_from_checkpoint, **kwargs)
        
        self.net = net
        self.net.to(self.device)
        
        if init_from_checkpoint:
            self.net.load_state_dict(kwargs['checkpoint'])
        
        self.flag_predict = False
    
    def _get_attributes(self):
        return MaskRCNN_attributes
    
    def _set_train_mode(self):
        self.net.train()
        self.flag_predict = False
        
    def _set_eval_mode(self):
        self.net.eval()
        self.flag_predict = True
    
    def _prediction(self, images, targets):
        #TODO: make compatible with batches
        
        if self.flag_predict:
            preds = self.net(images)
            
            thresh_idx = (preds[0]['scores'] > self.mask_confidence_thresh).nonzero().squeeze(1)
            pred_mask = torch.einsum('bcij->cij', (preds[0]['masks'][thresh_idx] > self.pixel_confidence_thresh).float()) # assumes batch_size is 1
            
            mask = torch.einsum('bcij->cij', preds[0]['masks'][thresh_idx])
            loss = F.binary_cross_entropy_with_logits(mask, targets[0]['masks'].float())
        
        else:
            loss = sum([loss for loss in self.net(images, targets).values()])
            
            self.net.eval()
            with torch.no_grad():
                preds = self.net(images)
                thresh_idx = (preds[0]['scores'] > self.mask_confidence_thresh).nonzero().squeeze(1)
                pred_mask = torch.einsum('bcij->cij', (preds[0]['masks'][thresh_idx] > self.pixel_confidence_thresh).float()) # assumes batch_size is 1
        
            self.net.train()
        
        return loss, pred_mask.detach()
    
    # def _threshold_mask(self, pred_mask):
    #     thresh_idx = (pred_mask['scores'] > self.mask_confidence_thresh).nonzero().squeeze(1)
    #     mask = torch.einsum('bcij->cij', (pred_mask['masks'][thresh_idx] > self.pixel_confidence_thresh).float())
    #     return mask.bool().cpu()

    def _get_parameters(self):
        return self.net.state_dict()