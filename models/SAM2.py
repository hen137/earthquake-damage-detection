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
    def __init__(self, args, predictor, device, train_loader=None, val_loader=None, test_loader=None, optimizer=None, lr_scheduler=None):
        self.args = args
        self.predictor = predictor
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
            if self.args.device: torch.cuda.empty_cache()
            
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
                
                '''
                predictor.model.image_encoder.train(True) # enable training of image encoder
                #Note that for this case, you will also need to scan the SAM2 code for “no_grad” commands and remove them (“ no_grad” blocks the gradient collection, which saves memory but prevents training).
                '''
                self.predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder 
                self.predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder
        
                self.optimizer.zero_grad()
                with torch.autocast(self.device): # cast to mix precision
                    image, mask, input_point, input_label = read_batch(data) # load data batch
                    if mask.shape[0] == 0: continue # ignore empty batches
                    
                    self.predictor.set_image(image) # apply SAM image encoder to the image
                    
                    mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(input_point, input_label, box=None, mask_logits=None, normalize_coords=True)
                    sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None)
                    
                    batched_mode = unnorm_coords.shape[0] > 1 # multi object prediction
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in self.predictor._features["high_res_feats"]]
                    low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=batched_mode,high_res_features=high_res_features)
                    prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[-1])# Upscale the masks to the original image resolution

                    # Segmentaion Loss caclulation

                    gt_mask = torch.tensor(mask.astype(torch.float32)).cuda()
                    prd_mask = torch.sigmoid(prd_masks[:, 0])# Turn logit map to probability map
                    seg_loss = (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001)).mean() # cross entropy loss

                    # Score loss calculation (intersection over union) IOU

                    inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                    iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
                    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                    train_loss = seg_loss+score_loss * 0.05  # mix losses

                    # apply back propogation

                    if scaler:
                        scaler.scale(train_loss).backward()  # Backpropogate
                        scaler.step(self.optimizer)
                        scaler.update() # Mix precision
                    else:
                        train_loss.backward()
                        self.optimizer.step()
                    
                    self.lr_scheduler.step()
                
                epoch_loss += train_loss.item()
                
                if i % self.args.print_freq == 0:
                    # self.predictor.model.eval()
                    with torch.no_grad():
                        # predictions = self.net(images)
                        
                        # thresh_idx = (predictions[0]['scores'] > self.args.mask_confidence_thresh).nonzero().squeeze(1)
                        # mask = torch.einsum('bcij->cij', (predictions[0]['masks'][thresh_idx] > self.args.pixel_confidence_thresh).float()) # assumes batch_size is 1
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
            raise ValueError(f'No value provided for validation_loader')
        
        

        return val_loss, accuracy, F1, IoU
    
    def predict(self):
        
        
        return images, predictions, targets
    
    def save_model(self, epoch, val_accuracy, val_F1, val_IoU, date_str):
        checkpoint_dir = self.args.chkpt_dir + '/SAM2' + f'/{self.args.dataset}'
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