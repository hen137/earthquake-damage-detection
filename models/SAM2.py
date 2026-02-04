# Native Imports
import time

# Library Imports
# import tqdm
import torch
import torch.nn.functional as F

# Custom Imports
from models.ModelWrapper import ModelBase

class SAM2(ModelBase):
    def __init__(self, predictor, device, init_from_checkpoint=False, **kwargs):
        super().__init__('SAM2', device, init_from_checkpoint, **kwargs)
        
        self.predictor = predictor
        
        if init_from_checkpoint:
            self.predictor.model.load_state_dict(kwargs['checkpoint'])
        
        self.flag_predict = False
    
    def _get_attributes(self):
        return {}
    
    def _set_train_mode(self):
        self.predictor.model.sam_mask_decoder.train()
        self.predictor.model.sam_prompt_encoder.train()
        self.flag_predict = False
        
    def _set_eval_mode(self):
        self.predictor.model.sam_mask_decoder.eval()
        self.predictor.model.sam_prompt_encoder.eval()
        self.flag_predict = True
    
    def _prediction(self, images, targets):
        #TODO: make compatible with batches
        
        # image, mask, input_point, input_label = read_batch(data) # load data batch
        # if mask.shape[0] == 0: continue # ignore empty batches
        
        self.predictor.set_image(images[0].permute(1, 2, 0).cpu().numpy()) # apply SAM image encoder to the image
        
        if self.flag_predict:
            pred_mask, _, _ = self.predictor.predict()
            pred_mask = torch.tensor(pred_mask).to(self.device)
            mask = torch.einsum('cij->ij', pred_mask).to(self.device)
            loss = F.binary_cross_entropy_with_logits(mask, targets[0]['masks'].squeeze(0).float())
        
        else:
            # mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
            # sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None)
            sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(points=None,boxes=None,masks=None)
            
            batched_mode = False #unnorm_coords.shape[0] > 1 # multi object prediction
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in self.predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features)
            pred_mask = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[-1])# Upscale the masks to the original image resolution

            # Segmentaion Loss caclulation

            gt_mask = targets[0]['masks'].float()  # Ground truth mask
            pred_mask = torch.sigmoid(pred_mask[:, 0])# Turn logit map to probability map
            seg_loss = (-gt_mask * torch.log(pred_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - pred_mask) + 0.00001)).mean() # cross entropy loss

            # Score loss calculation (intersection over union) IOU

            inter = (gt_mask * (pred_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (pred_mask > 0.5).sum(1).sum(1) - inter)
            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05  # mix losses
        
        pred_mask = (torch.einsum('cij->ij', pred_mask) > 0.5)
        
        return loss, pred_mask.detach()
    
    # def _threshold_mask(self, pred_mask):
    #     return ((torch.einsum('cij->ij', torch.tensor(pred_mask)) > 0.5).cpu())

    def _get_parameters(self):
        return self.predictor.model.state_dict()