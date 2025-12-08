# Native Imports
import argparse

# Library Imports
# import tqdm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from datasets import load_dataset

# Custom Imports
from data.data import detection_collate

def build_model(args, train_loader, val_loader):
    device = 'cuda' if args.device and torch.cuda.is_available() else 'cpu'
    
    if args.model == 'SAM2':
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from models.SAM2 import SAM2 as TrainerClass
        
        sam2_model = build_sam2_hf("facebook/sam2.1-hiera-small", device=device)
        predictor = SAM2ImagePredictor(sam2_model) # load net
        
        net = predictor
        grad_params = [p for p in predictor.model.parameters() if p.requires_grad]
        
        kwargs = {}

    elif args.model == 'MaskRCNN':
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
        # from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        
        from models.MaskRCNN import MaskRCNN as TrainerClass
        
        net = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
        
        in_features_box = net.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = net.roi_heads.mask_predictor.conv5_mask.in_channels
        dim_reduced = net.roi_heads.mask_predictor.conv5_mask.out_channels
        net.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=2)
        net.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced, num_classes=2)
        
        kwargs = {
            'pixel_confidence_thresh': args.pixel_confidence_thresh,
            'mask_confidence_thresh': args.mask_confidence_thresh,
        }
        
        grad_params = [p for p in net.parameters() if p.requires_grad]
    
    else:
        raise ValueError(f'Unknown encoder type: {args.model}')

    # optimizer = optim.SGD(
    #     [p for p in net.parameters() if p.requires_grad],
    #     lr=args.lr,
    #     weight_decay=args.weight_decay,
    #     momentum=args.momentum,
    #     nesterov=True
    # )
    optimizer = torch.optim.AdamW(
        grad_params, 
        lr=args.lr, 
        weight_decay=args.weight_decay,
    )
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr, 
        total_steps=args.epochs * len(train_loader)
    )
    
    kwargs = {
        **kwargs,
        'train_loader': train_loader,
        'validation_loader': val_loader,
        'optimizer': optimizer,
        'lr_scheduler': lr_scheduler,
        'use_scaler': args.use_scaler,
        'dataset': args.dataset,
        'print_freq': args.print_freq,
        'val_freq': args.val_freq,
        'checkpoint_dir': args.chkpt_dir,
    }
    
    model = TrainerClass(
        net, 
        device,
        **kwargs
    )
    
    return net, model

def get_data_loaders(args):
    transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((512, 512))])
    
    if args.dataset == 'KATE_CD':
        from data.data import KATE_CD
        
        data = load_dataset('CSCRS/kate-cd')

        trainset = KATE_CD(data, 'train', transforms)
        train_loader = DataLoader(trainset, args.train_batch_size, shuffle=True, collate_fn=detection_collate)
        
        validationset = KATE_CD(data, 'validation', transforms)
        validation_loader = DataLoader(validationset, args.val_batch_size, collate_fn=detection_collate)
        
    elif args.dataset == 'KATE_PD':
        from data.data import KATE_PD
        
        data = load_dataset('CSCRS/kate-pd')

        trainset = KATE_PD(data, 'train', transforms)
        train_loader = DataLoader(trainset, args.train_batch_size, shuffle=True, collate_fn=detection_collate)
        
        validationset = KATE_PD(data, 'validation', transforms)
        validation_loader = DataLoader(validationset, args.val_batch_size, collate_fn=detection_collate)
        
    elif args.dataset == 'Flood':
        from data.data import FLOOD
        
        trainset = FLOOD('./data/flood/train', transforms=transforms)
        train_loader = DataLoader(trainset, args.train_batch_size, shuffle=True, collate_fn=detection_collate)
        
        validationset = FLOOD('./data/flood/validation', transforms=transforms)
        validation_loader = DataLoader(validationset, args.val_batch_size, collate_fn=detection_collate)
    
    else:
        raise ValueError(f'Unknown dataset type: {args.dataset}')
    
    return train_loader, validation_loader

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training")
    
    parser.add_argument('--model', required=True, choices=['SAM2', 'DeepLabV3+', 'MaskRCNN'])
    parser.add_argument('--dataset', required=True, choices=['KATE_CD', 'KATE_PD', 'Flood'])
    
    parser.add_argument('--device', required=False, default='gpu', action='store_true')
    # parser.add_argument('--multi_gpu', required=False, default=None, type=str)
    # parser.add_argument('--dev_id', required=False, default=0, type=int)
    # parser.add_argument('--data_loader_num_workers', required=False, default=16, type=int)
    parser.add_argument('--use_scaler', required=False, default=False, type=bool)
    
    parser.add_argument('--epochs', required=False, default=20, type=int)
    parser.add_argument('--train_batch_size', required=False, default=1, type=int)
    parser.add_argument('--val_batch_size', required=False, default=1, type=int)
    parser.add_argument('--lr', required=False, default=1e-3, type=float)
    
    parser.add_argument('--weight_decay', required=False, default=5e-4, type=float)
    parser.add_argument('--momentum', required=False, default=0.9, type=float)
    
    parser.add_argument('--pixel_confidence_thresh', required=False, default=0.75, type=float)
    parser.add_argument('--mask_confidence_thresh', required=False, default=0.65, type=float)
    
    parser.add_argument('--print_freq', required=False, default=20, type=int)
    parser.add_argument('--val_freq', required=False, default=2, type=int)
    parser.add_argument('--chkpt_dir', required=False, default='./models/checkpoints')
    parser.add_argument('--graph_hists', required=False, default=False, type=bool)

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    train_loader, val_loader = get_data_loaders(args)
    
    net, model = build_model(args, train_loader, val_loader)

    # if args.multi_gpu:
    #     net = torch.nn.DataParallel(net, [int(id) for id in args.multi_gpu.split(',')])
    # net.to(device=torch.device('cuda', int(args.dev_id)))

    print(f'Training {args.model} started')
    model.train(args.epochs)
    print(f'Training {args.model} finished')

if __name__ == '__main__':
    main()