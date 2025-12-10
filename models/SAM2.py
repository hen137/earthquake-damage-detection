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

class SAM2(ModelBase):
    def __init__(self, predictor, device, init_from_checkpoint=False, **kwargs):
        super(device, init_from_checkpoint, **kwargs)
        
        self.predictor = predictor
        
        if init_from_checkpoint:
            self.predictor.model.load_state_dict(kwargs['checkpoint'])
    
    def _get_attributes(self):
        return {}
    
    def _set_train_mode(self):
        self.predictor.model.sam_mask_decoder.train()
        self.predictor.model.sam_prompt_encoder.train()
        
    def _set_eval_mode(self):
        self.predictor.model.sam_mask_decoder.eval()
        self.predictor.model.sam_prompt_encoder.eval()
    
    def _get_prediction(self, images, targets):
        return loss, pred_mask.detach()
        
    def train(self, epochs):
                with torch.autocast(self.device): # cast to mix precision
                    # image, mask, input_point, input_label = read_batch(data) # load data batch
                    # if mask.shape[0] == 0: continue # ignore empty batches
                    
                    self.predictor.set_image(images[0].permute(1, 2, 0).cpu().numpy()) # apply SAM image encoder to the image
                    
                    # mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
                    # sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None)
                    sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(points=None,boxes=None,masks=None)
                    
                    batched_mode = False #unnorm_coords.shape[0] > 1 # multi object prediction
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in self.predictor._features["high_res_feats"]]
                    low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features)
                    prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[-1])# Upscale the masks to the original image resolution

                    # Segmentaion Loss caclulation

                    gt_mask = targets[0]['masks'].float()  # Ground truth mask
                    prd_mask = torch.sigmoid(prd_masks[:, 0])# Turn logit map to probability map
                    seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

                    # Score loss calculation (intersection over union) IOU

                    inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                    iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
                    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                    train_loss = seg_loss+score_loss * 0.05  # mix losses

        
    def predict(self):
        with torch.no_grad():
            for i, (image_batch, targets_batch) in enumerate(self.test_loader):
                self.predictor.set_image(image_batch[0].permute(1, 2, 0).cpu().numpy()) # apply SAM image encoder to the image
                
                t = time.time()
                masks, scores, logits = self.predictor.predict()
                t_img_avg = time.time() - t
                
                predictions.append({'mask': (torch.einsum('cij->ij', torch.tensor(masks)) > 0.5).cpu()})
        
        t_img_avg /= len(self.test_loader)
        
        return predictions, t_img_avg
    
    def _validate(self, epoch):
                self.predictor.set_image(image[0].permute(1, 2, 0).cpu().numpy()) # apply SAM image encoder to the image
                
                masks, scores, logits = self.predictor.predict()
                
                # masks = masks[:,0].astype(bool)
                # shorted_masks = torch.tensor(masks[torch.argsort(torch.tensor(scores[0]))][::-1].astype(bool))
                
                # seg_map = torch.zeros_like(shorted_masks)
                # occupancy_mask = torch.zeros_like(shorted_masks, dtype=bool)
                
                # print(masks.shape)
                # for i in range(shorted_masks.shape[0]):
                #     mask = shorted_masks[i]
                #     if (mask * occupancy_mask).sum() / mask.sum() > 0.15: continue 
                #     mask[occupancy_mask] = 0
                #     seg_map[mask] = i + 1
                #     occupancy_mask[mask] = 1
                mask = torch.einsum('cij->ij', torch.tensor(masks)).to(self.device)
                loss = F.binary_cross_entropy_with_logits(mask, targets[0]['masks'].squeeze(0).float())
                
                mask = (torch.einsum('cij->ij', torch.tensor(masks)) > 0.5).to(self.device)
                acc, _, _, f1, iou = binary_accuracy(mask, targets[0]['masks'].squeeze(0))
                