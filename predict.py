# Native Imports
import argparse, time

# Library Imports
# import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from datasets import load_dataset

# Local Imports
from data.data import detection_collate
from utils.utils import save_hist_graphs, visualize_predictions
from utils.attributes import hist_attributes

def build_model(model, gpu, test_loader, chkpt_file):
    device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
    
    kwargs = {}
    
    if model == 'SAM2':
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from models.SAM2 import SAM2 as TrainerClass
        
        sam2_model = build_sam2_hf("facebook/sam2.1-hiera-small", device=device)
        predictor = SAM2ImagePredictor(sam2_model) # load net
        
        net = predictor
        
        checkpoint = torch.load(chkpt_file, map_location=device, weights_only=False)
        # predictor.model.load_state_dict(checkpoint['model_state_dict'])
        
        # kwargs = {
        #     **kwargs,
        # }
        
    elif model == 'MaskRCNN':
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
        # from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        from models.MaskRCNN import MaskRCNN as TrainerClass
         
        net = maskrcnn_resnet50_fpn_v2()
        
        in_features_box = net.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = net.roi_heads.mask_predictor.conv5_mask.in_channels
        dim_reduced = net.roi_heads.mask_predictor.conv5_mask.out_channels
        net.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=2)
        net.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced, num_classes=2)
        
        checkpoint = torch.load(chkpt_file, map_location=device, weights_only=False)
        # net.load_state_dict(checkpoint['model_state_dict'])
        
        kwargs = {
            **kwargs,
            'pixel_confidence_thresh': checkpoint['pixel_confidence_thresh'],
            'mask_confidence_thresh': checkpoint['mask_confidence_thresh'],
        }
    
    else:
        raise ValueError(f'Unknown encoder type: {model}')
    
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

def get_test_loaders(dataset, test_batch_size):
    if dataset == 'KATE_CD':
        from data.data import KATE_CD
        data = load_dataset('CSCRS/kate-cd')

        transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        testset = KATE_CD(data, 'test', transforms)
        test_loader = DataLoader(testset, test_batch_size, collate_fn=detection_collate)
        
    elif dataset == 'KATE_PD':
        from data.data import KATE_PD
        data = load_dataset('CSCRS/kate-pd')

        transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

        testset = KATE_PD(data, 'test', transforms)
        test_loader = DataLoader(testset, test_batch_size, collate_fn=detection_collate)
        
    elif dataset == 'Flood':
        from data.data import FLOOD

        transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((512, 512))])
        
        testset = FLOOD(data_dir='./data/flood/test', transforms=transforms)
        test_loader = DataLoader(testset, test_batch_size, collate_fn=detection_collate)
    
    else:
        raise ValueError(f'Unknown dataset type: {dataset}')
    
    return test_loader

def parse_arguments():
    parser = argparse.ArgumentParser(description="Predicting")

    parser.add_argument('--model', required=True, choices=['SAM2', 'MaskRCNN']) 
    parser.add_argument('--dataset', required=True, choices=['KATE_CD', 'KATE_PD', 'Flood'])

    parser.add_argument('--gpu', required=False, default=True, action='store_true')
    # parser.add_argument('--multi_gpu', required=False, default=None, type=str)
    # parser.add_argument('--dev_id', required=False, default=0, type=int)
    # parser.add_argument('--data_loader_num_workers', required=False, default=16, type=int)

    parser.add_argument('--test_batch_size', required=False, default=1, type=int)

    parser.add_argument('--chkpt_file', required=True, type=str)
    # parser.add_argument('--chkpt_file', required=False, default='models\checkpoints\MaskRCNN\KATE_CD\MaskRCNN_04-12-2025_01-45_E8_vA97.03_vF0.173_vIoU0.126.pth')
    
    parser.add_argument('--output_frmt', required=True, choices=['overlay', 'rand_5'])
    parser.add_argument('--output_dir', required=False, default='outputs/predictions')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    output_dir = args.output_dir + '/' + args.model + '/' + args.dataset
    
    test_loader = get_test_loaders(args.dataset, args.test_batch_size)
    
    images = []
    targets = []
    
    for img_batch, targets_batch in test_loader:
        for (image, target) in zip(img_batch, targets_batch):
            images.append(image)
            targets.append(target)
    
    net, model = build_model(args.model, args.gpu, test_loader, args.chkpt_file)
    
    t = time.time()
    predictions, avg_time = model.predict()
    t_total = time.time() - t
    
    visualize_predictions(args.output_frmt, images, predictions, targets)
    
    model_hist = {args.model: {hist: getattr(model, hist) for hist in hist_attributes}}
    
    save_hist_graphs(output_dir, **model_hist)
    
    print(f'Total Train Time: {model.train_time // 60:.0f}m {model.train_time % 60:.0f}s')
    print(f'Average inference time per image: {avg_time:.4f} seconds')
    print(f'Total inference time for {len(images)} images: {t_total:.4f} seconds')
    
if __name__ == '__main__':
    main()
     