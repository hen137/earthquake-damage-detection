# Native Imports
import os

# Library Imports
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision.ops import masks_to_boxes
from torchvision.io import decode_image

# Custom Imports

# Classes
class KATE_CD(Dataset):
  def __init__(self, data, split, transforms=None):
    # self.data = data
    # self.split = split
    self.data = data[split]
    self.transforms = transforms
    self.length = self.data.num_rows

    self.data.set_format('torch', self.data.column_names)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    sample = self.data[idx]
    
    image = self.transforms(sample['post_image'])
    
    masks = sample['label']
    if masks.dim() == 2:
        masks = masks.unsqueeze(0)
    masks = self.transforms(masks)
    masks = (masks > 0.5).to(torch.uint8)
    
    bboxes = masks_to_boxes(masks)
    
    labels = torch.tensor([1], dtype=torch.int64)

    return image, {'boxes': bboxes, 'masks': masks, 'labels': labels}

class KATE_PD(Dataset):
  def __init__(self, data, split, transforms=None):
    # self.data = data
    # self.split = split
    self.data = data[split]
    self.transforms = transforms
    self.length = self.data.num_rows

    self.data.set_format('torch', self.data.column_names)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    sample = self.data[idx]
    image = self.transforms(sample['image'])
    
    masks = sample['mask']
    if masks.dim() == 2:
        masks = masks.unsqueeze(0)
    masks = self.transforms(masks)
    masks = (masks > 0.5).to(torch.uint8)
    
    bboxes = masks_to_boxes(masks)
    
    labels = torch.tensor([1], dtype=torch.int64)

    return image, {'boxes': bboxes, 'masks': masks, 'labels': labels}
  
class FLOOD(Dataset):
  def __init__(self, data_dir, transforms):
    self.image_dir = os.path.join(data_dir, 'Image')
    self.mask_dir = os.path.join(data_dir, 'Mask')
    
    self.size = len([name for name in os.listdir(self.image_dir)])
    
    self.transforms = transforms
  
  def __len__(self):
    return self.size
  
  def __getitem__(self, idx):
    image = decode_image(self.image_dir + f'/{idx}.jpg', mode='RGB')
    image = self.transforms(image)

    masks = decode_image(self.mask_dir + f'/{idx}.png', mode='GRAY')
    masks = self.transforms(masks)
    masks = (masks > 0.5).to(torch.uint8)

    bboxes = masks_to_boxes(masks)
    labels = torch.tensor([1], dtype=torch.int64)

    return image, {'boxes': bboxes, 'masks': masks, 'labels': labels}

# Functions
def detection_collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets
