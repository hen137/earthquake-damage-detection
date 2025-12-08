# Native Imports
import os, time

# Library Imports
# import tqdm
import torch
import torch.nn.functional as F

# Custom Imports
from utils.utils import binary_accuracy, save_hist_graphs
from utils.attributes import hist_attributes, train_attributes, predict_attributes

class SAM2():
    def __init__(self, predictor, device, init_from_chkpt=False, **kwargs):
        self.predictor = predictor
        self.device = device
        
        # self.net.to(self.device)
        
        for attr, value in kwargs.items():
            if attr in list(hist_attributes.keys()): continue
            setattr(self, attr, value)
            
        if init_from_chkpt:
            if not hasattr(self, 'checkpoint'):
                raise ValueError(f'No checkpoint provided for initializing SAM2(init_from_chkpt=True, checkpoint=...)')
            
            self.predictor.model.load_state_dict(kwargs['checkpoint'])
            
            for hist in hist_attributes.keys():
                setattr(self, hist, kwargs[hist])
        
        else:
            for hist in hist_attributes.keys():
                setattr(self, hist, [])
    
    def train(self, epochs):
        self._confirm_attributes(train_attributes)
        
        '''
        This code is written assuming a batch size of 1 for test loader
        '''
        
        best_epoch_accuracy = 0.0
        best_validation_accuracy = 0.0
        best_validation_loss = 0
        best_F1 = 0.0
        best_IoU = 0.0

        begin_time = time.time()
        current_time = time.localtime(begin_time)
        date_str = time.strftime("%d-%m-%Y_%H-%M", current_time)
        
        scaler = torch.amp.GradScaler() if self.device == 'cuda' and self.use_scaler else None
        
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
                
                '''
                predictor.model.image_encoder.train(True) # enable training of image encoder
                #Note that for this case, you will also need to scan the SAM2 code for “no_grad” commands and remove them (“ no_grad” blocks the gradient collection, which saves memory but prevents training).
                '''
                self.predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder 
                self.predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder
        
                self.optimizer.zero_grad()
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
                
                if i % self.print_freq == 0:
                    # self.predictor.model.eval()
                    # with torch.no_grad():
                        # predictions = self.net(images)
                        
                        # thresh_idx = (predictions[0]['scores'] > self.args.mask_confidence_thresh).nonzero().squeeze(1)
                        # mask = torch.einsum('bcij->cij', (predictions[0]['masks'][thresh_idx] > self.args.pixel_confidence_thresh).float()) # assumes batch_size is 1
                    accuracy, precision, recall, f1, iou = binary_accuracy(prd_mask, targets[0]['masks'].squeeze(0))
                    print(f1, iou)
                    epoch_accuracy += accuracy
                    epoch_precision += precision
                    epoch_recall += recall
                    epoch_f1 += f1
                    epoch_iou += iou
                
                    # print(f'[Train] [Epoch {epoch + 1}] [Iter. {i}] [Learning Rate {self.optimizer.param_groups[0]['lr']:.2e}] [Loss {train_loss.item():.4f}, IoU {iou.item():.3f}]')
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
            
            print(self.train_f1_hist)
            print(self.train_iou_hist)
            
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
        self._confirm_attributes(predict_attributes)
        
        '''
        This code is written assuming a batch size of 1 for test loader
        '''
        
        predictions = []
        t_img_avg = 0.0
        
        # self.predictor.model.eval()
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
        
        '''
        This code is written assuming a batch size of 1 for test loader
        '''
        
        if self.device == 'cuda': torch.cuda.empty_cache()
        
        start_time = time.time()

        val_loss = 0
        accuracy = 0
        F1 = 0
        IoU = 0

        # self.net.eval()
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
                
                # predictions = self.net(image)
                
                # thresh_idx = (predictions[0]['scores'] > self.args.mask_confidence_thresh).nonzero().squeeze(1)
                # mask = torch.einsum('bcij->cij', predictions[0]['masks'][thresh_idx])
                # loss = F.binary_cross_entropy_with_logits(mask, targets[0]['masks'].float())
                
                # mask = torch.einsum('bcij->cij', (predictions[0]['masks'][thresh_idx] > self.args.pixel_confidence_thresh).float())
                # acc, _, _, f1, iou = binary_accuracy(mask, targets[0]['masks'].squeeze(0))
                
                self.predictor.set_image(image[0].permute(1, 2, 0).cpu().numpy()) # apply SAM image encoder to the image
                
                masks, scores, logits = self.predictor.predict()
                
                # print(scores)
                
                # plt.imshow(mask.cpu().numpy())
                # plt.show()
                # exit()
                
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
        checkpoint_dir = self.checkpoint_dir + '/SAM2' + f'/{self.dataset}'
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.predictor.model.state_dict(),
                
                'val_accuracy': val_accuracy,
                'val_F1': val_F1,
                'val_IoU': val_IoU,
                
                # metrics vs epoch history
                'train_loss_hist': self.train_loss_hist,
                'train_accuracy_hist': self.train_accuracy_hist,
                'train_precision_hist': self.train_precision_hist,
                'train_recall_hist': self.train_recall_hist,
                'train_f1_hist': self.train_f1_hist,
                'train_iou_hist': self.train_iou_hist,
                
                'date_str': date_str
            },
            os.path.join(
                checkpoint_dir, 
                f"SAM2_{date_str}_E{epoch}_vA{val_accuracy * 100:.2f}_vF{val_F1:.3f}_vIoU{val_IoU:.3f}.pth"
            )
        )
        
    def _save_model(self, date_str, epochs):
        checkpoint_dir = self.checkpoint_dir + '/SAM2' + f'/{self.dataset}'
        if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
        
        hists = {hist: getattr(self, hist) for hist in hist_attributes.keys()}
        
        torch.save(
            {
                'epoch': epochs,
                'model_state_dict': self.predictor.model.state_dict(),
                'train_time': self.train_time,
                # metrics vs epoch history
                'train_loss_hist': self.train_loss_hist,
                'train_accuracy_hist': self.train_accuracy_hist,
                'train_precision_hist': self.train_precision_hist,
                'train_recall_hist': self.train_recall_hist,
                'train_f1_hist': self.train_f1_hist,
                'train_iou_hist': self.train_iou_hist,
                
                'date_str': date_str
            },
            os.path.join(
                checkpoint_dir, 
                f"SAM2_{date_str}_E{epochs}.pth"
            )
        )
    