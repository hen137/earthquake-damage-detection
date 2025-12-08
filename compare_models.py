# Native Imports
import argparse

# Library Imports
# import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from datasets import load_dataset

# Custom Imports
from data.data import detection_collate
from utils.utils import compare_predictions

def build_model(args, test_loader, model):
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    
    if model == 'SAM2':
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from models.SAM2 import SAM2 as TrainerClass
        
        sam2_model = build_sam2_hf("facebook/sam2.1-hiera-small", device=device)
        predictor = SAM2ImagePredictor(sam2_model) # load net
        
        net = predictor
        
        checkpoint = torch.load(args.chkpt_file, map_location=device, weights_only=False)
        # predictor.model.load_state_dict(checkpoint['model_state_dict'])
        
        kwargs = {
            
        }
        
    elif model == 'MaskRCNN':
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
        
        checkpoint = torch.load(args.chkpt_file, map_location=device, weights_only=False)
        # net.load_state_dict(checkpoint['model_state_dict'])
        
        kwargs = {
            'pixel_confidence_thresh': checkpoint['pixel_confidence_thresh'],
            'mask_confidence_thresh': checkpoint['mask_confidence_thresh'],
        }
    
    else:
        raise ValueError(f'Unknown model type: {model}')
    
    kwargs = {
        **kwargs,
        'test_loader': test_loader,
        'checkpoint': checkpoint['model_state_dict'],
        'train_loss_hist': checkpoint['train_loss_hist'],
        'train_accuracy_hist': checkpoint['train_accuracy_hist'],
        'train_precision_hist': checkpoint['train_precision_hist'],
        'train_recall_hist': checkpoint['train_recall_hist'],
        'train_f1_hist': checkpoint['train_f1_hist'],
        'train_iou_hist': checkpoint['train_iou_hist'],
        'train_time': checkpoint['train_time'],
    }
    
    model = TrainerClass(
        net, 
        device,
        init_from_chkpt=True,
        **kwargs
    )
    
    return net, model

def get_test_loaders(args):
    if args.dataset == 'KATE_CD':
        from data.data import KATE_CD
        data = load_dataset('CSCRS/kate-cd')

        transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        testset = KATE_CD(data, 'test', transforms)
        test_loader = DataLoader(testset, args.test_batch_size, collate_fn=detection_collate)
        
    elif args.dataset == 'KATE_PD':
        from data.data import KATE_PD
        data = load_dataset('CSCRS/kate-pd')

        transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        testset = KATE_PD(data, 'test', transforms)
        test_loader = DataLoader(testset, args.test_batch_size, collate_fn=detection_collate)
        
    elif args.dataset == 'Flood':
        from data.data import FLOOD

        transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((512, 512))])
        
        testset = FLOOD(data_dir='./data/flood/test', transforms=transforms)
        test_loader = DataLoader(testset, args.test_batch_size, collate_fn=detection_collate)
    
    else:
        raise ValueError(f'Unknown dataset type: {args.dataset}')
    
    return test_loader

def parse_arguments():
    parser = argparse.ArgumentParser(description="Predicting")

    parser.add_argument('--dataset', required=True, choices=['KATE_CD', 'KATE_PD', 'Flood'])

    parser.add_argument('--gpu', required=False, default=True, action='store_true')
    # parser.add_argument('--multi_gpu', required=False, default=None, type=str)
    # parser.add_argument('--dev_id', required=False, default=0, type=int)
    # parser.add_argument('--data_loader_num_workers', required=False, default=16, type=int)

    parser.add_argument('--test_batch_size', required=False, default=1, type=int)

    parser.add_argument('--MaskRCNN_chkpt_file', required=True)
    parser.add_argument('--SAM2_chkpt_file', required=True)
    # parser.add_argument('--MaskRCNN_chkpt_file', required=False, default='')
    # parser.add_argument('--SAM2_chkpt_file', required=False, default='')
    
    parser.add_argument('--num_samples', required=False, default=5, type=int)
    
    parser.add_argument('--output_dir', required=False, default='outputs/comparisons')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    output_dir = args.output_dir + '/' + args.dataset
    
    test_loader = get_test_loaders(args)
    
    images = []
    targets = []
    
    for image_batch, target_batch in test_loader:
        for image, target in zip(image_batch, target_batch):
            images.append(image)
            targets.append(target)
    
    _, MaskRCNN = build_model(args, test_loader, 'MaskRCNN')
    _, SAM2 = build_model(args, test_loader, 'SAM2')
    
    # print(MaskRCNN.train_f1_hist)
    model_hists = {model_name: {hist: getattr(model, hist) for hist in hist_attributes} for model_name, model in zip(['MaskRCNN', 'SAM2'], [MaskRCNN, SAM2])}
    
    save_hist_graphs(output_dir, **model_hists)
    
    t = time.time()
    MaskRCNN_predictions, MaskRCNN_time = MaskRCNN.predict()
    MaskRCNN_total_time = time.time() - t
    
    t = time.time()
    SAM2_predictions, SAM2_time = SAM2.predict()
    SAM2_total_time = time.time() - t
    
    model_predictions = {
        'MaskRCNN': MaskRCNN_predictions,
        'SAM2': SAM2_predictions
    }
    
    compare_predictions(images, targets, args.num_samples, **model_predictions)
    
    print(f'MaskRCNN Total Train Time: {MaskRCNN.train_time // 60:.0f}m {MaskRCNN.train_time % 60:.0f}s')
    print(f'MaskRCNN Average Inference Time per Batch: {MaskRCNN_time:.4f} seconds')
    print(f'MaskRCNN Total Inference Time for {len(images)} images: {MaskRCNN_total_time:.4f} seconds')
    print()
    print(f'SAM2 Total Train Time: {SAM2.train_time // 60:.0f}m {SAM2.train_time % 60:.0f}s')
    print(f'SAM2 Average Inference Time per Batch: {SAM2_time:.4f} seconds')
    print(f'SAM2 Total Inference Time for {len(images)} images: {SAM2_total_time:.4f} seconds')

if __name__ == "__main__":
    main()