# Native Imports
import argparse

# Library Imports
# import tqdm
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.utils import draw_segmentation_masks
from datasets import load_dataset

# Custom Imports
from data.data import KATE

def build_model(args, test_loader):
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    
    if args.model == 'SAM2':
        from models.SAM2 import SAM2 as TrainerClass
        #TODO: setup SAM2
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
        
        # checkpoint = torch.load(args.chkpt_file, map_location=device)
        # net.load_state_dict(checkpoint['model_state_dict'])
        
    # elif args.model == 'DeepLabV3+':
    #     from models.DeepLabV3 import DeepLabV3 as TrainerClass
    else:
        raise ValueError(f'Unknown encoder type: {args.encoder}')
    
    model = TrainerClass(
        args, 
        net, 
        device, 
        test_loader=test_loader
    )
    
    return net, model

def get_data_loaders(args):
    data = load_dataset('CSCRS/kate-cd')

    transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    testset = KATE(data, 'test', transforms)
    test_loader = DataLoader(testset, args.test_batch_size)
    
    return test_loader

def parse_arguments():
    parser = argparse.ArgumentParser(description="Predicting")

    parser.add_argument('--model', required=True, choices=['SAM2', 'DeepLabV3+', 'MaskRCNN']) 

    parser.add_argument('--gpu', required=False, default=True, action='store_true')
    # parser.add_argument('--multi_gpu', required=False, default=None, type=str)
    # parser.add_argument('--dev_id', required=False, default=0, type=int)
    # parser.add_argument('--data_loader_num_workers', required=False, default=16, type=int)

    parser.add_argument('--test_batch_size', required=False, default=1, type=int)

    parser.add_argument('--pixel_confidence_thresh', required=False, default=0.75, type=float)
    parser.add_argument('--mask_confidence_thresh', required=False, default=0.65, type=float)

    # parser.add_argument('--chkpt_file', required=True)

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    test_loader = get_data_loaders(args)
    
    net, model = build_model(args, test_loader)
    
    images, predictions, targets = model.predict()
    
    idx = 0
    
    gt = draw_segmentation_masks(image=images[idx], masks=(targets[idx]['masks'].squeeze(0) > 0), alpha=0.3, colors='red')
    pred = draw_segmentation_masks(image=images[idx], masks=predictions[idx]['mask'], alpha=0.3, colors='blue')
    
    # plt.imshow(gt.permute(1, 2, 0))
    plt.imshow(torch.cat((gt, pred), 1).permute(1, 2, 0))
    plt.show()
    
    # for i in []:
    #     ...
    
if __name__ == '__main__':
    main()
    
# class PredOptions():
#     def __init__(self):
#         """Reset the class; indicates the class hasn't been initailized"""
#         self.initialized = False
#     # ResNet_CD_e4_OA98.74_F78.17_IoU68.77.pth    
#     def initialize(self, parser):
#         parser.add_argument('--crop_size', required=False, default=(512, 512), help='cropping size')
#         parser.add_argument('--TTA', required=False, default=True, help='Test time augmentation')
#         parser.add_argument('--test_dir', required=True, type=pathlib.Path, help='directory to test images')
#         parser.add_argument('--pred_dir', required=True, type=pathlib.Path, help='directory to output masks')
#         parser.add_argument('--chkpt_path', required=True, type=pathlib.Path )
#         parser.add_argument('--dev_id', required=False, default=0, help='Device id')
#         parser.add_argument('--encoder', required=True, choices=['ResNet', 'SAM', 'effSAM'])
#         self.initialized = True
#         return parser
        
#     def gather_options(self):
#         if not self.initialized:  # check if it has been initialized
#             parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#             parser = self.initialize(parser)
#         self.parser = parser
#         return parser.parse_args()

#     def parse(self):
#         self.opt = self.gather_options()
#         return self.opt

# def create_crops(imgA, imgB, size):
#     imgA_crops = []
#     imgB_crops = []
#     h = imgA.shape[0]
#     w = imgA.shape[1]
#     c_h = size[0]
#     c_w = size[1]
#     if h < c_h or w < c_w:
#         print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
#         return 1
#     h_rate = h/c_h
#     w_rate = w/c_w
#     rows = math.ceil(h_rate)
#     cols = math.ceil(w_rate)
#     stride_h = int((c_h*rows-h)/(rows-1))
#     stride_w = int((c_w*cols-w)/(cols-1))
#     for j in range(rows):
#         for i in range(cols):
#             s_h = int(j*c_h - j*stride_h)
#             if(j==(rows-1)): s_h = h - c_h
#             e_h = s_h + c_h
#             s_w = int(i*c_w - i*stride_w)
#             if(i==(cols-1)): s_w = w - c_w
#             e_w = s_w + c_w
#             imgA_crops.append(imgA[s_h:e_h, s_w:e_w, :])
#             imgB_crops.append(imgB[s_h:e_h, s_w:e_w, :])
#     print('Sliding crop finished. %d images created.' %len(imgA_crops))
#     return imgA_crops, imgB_crops

# def stitch_pred(patch_list, size_stitch):
#     H, W = size_stitch
#     h, w = patch_list[0].shape
#     stitch_rows = math.ceil(H/h)
#     stitch_cols = math.ceil(W/w)
#     assert stitch_rows*stitch_cols == len(patch_list), "Stitching patch number mismatch." 
    
#     h_overlap = int((h*stitch_rows-H)/(stitch_rows-1))
#     w_overlap = int((w*stitch_cols-W)/(stitch_cols-1))
    
#     for r in range(stitch_rows):
#         crop_t = math.ceil(h_overlap/2)
#         crop_b = h_overlap-crop_t
#         crop_l = math.ceil(w_overlap/2)
#         crop_r = w_overlap-crop_l
#         if r == 0: crop_t=0
#         if r == stitch_rows-1:
#             crop_b=0
#             crop_t = stitched_img.shape[0]-H
#         stitched_r = patch_list[r*stitch_cols][crop_t:h-crop_b, 0:w-crop_r]
#         for c in range(1,stitch_cols):
#             if c == stitch_cols-1:
#                 crop_r = 0
#                 crop_l = stitched_r.shape[1]-W
#             patch_croped = patch_list[r*stitch_cols+c][crop_t:h-crop_b, crop_l:w-crop_r]
#             stitched_r = np.concatenate((stitched_r, patch_croped), axis=1)           
#         if r==0: stitched_img = stitched_r
#         else: stitched_img = np.concatenate((stitched_img, stitched_r), axis=0)
#     #sH, sW = stitched_img.shape
#     #if sH>H or sW>W: stitched_img = stitched_img[:H, :W]
#     print('Pred Stitched (%d, %d)'%(stitched_img.shape[0], stitched_img.shape[1]))
#     return stitched_img

# def compare_models(model_1, model_2):
#     models_differ = 0
#     for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
#         if torch.equal(key_item_1[1], key_item_2[1]):
#             pass
#         else:
#             models_differ += 1
#             if (key_item_1[0] == key_item_2[0]):
#                 print('Mismtach found at', key_item_1[0])
#             else:
#                 raise Exception
#     if models_differ == 0:
#         print('Models match perfectly! :)')   

# def main():
#     begin_time = time.time()
#     opt = PredOptions().parse()
#     if opt.encoder == 'ResNet':
#         from models.SAM2 import ResNet_CD as Net
#     elif opt.encoder == 'SAM':
#         from models.DeepLabV3 import SAM_CD as Net
#     elif opt.encoder == 'effSAM':
#         from models.MaskRCNN import SAM_CD as Net
#     else:
#         raise Exception(f'Unknown encoder {opt.encoder}')
#     net = Net()
#     state_dict = torch.load(opt.chkpt_path, map_location="cpu", weights_only=True)
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         # name = k[7:] # remove `module.`
#         if 'module.' in k:
#             new_state_dict[k[7:]] = v
#         else:
#             new_state_dict = state_dict
#     net.load_state_dict(new_state_dict)  
#     net.to(torch.device('cuda', int(opt.dev_id))).eval()
    
#     predict(net, opt)
#     time_use = time.time() - begin_time
#     print('Total time: %.2fs'%time_use)

# def predict(net, opt):
#     if not os.path.exists(opt.pred_dir): os.makedirs(opt.pred_dir)
#     imgA_dir = os.path.join(opt.test_dir, 'A')
#     imgB_dir = os.path.join(opt.test_dir, 'B')
#     data_list = os.listdir(imgA_dir)
#     valid_list = []
#     for it in data_list:
#         if (it[-4:]=='.png'): valid_list.append(it)
    
#     for it in valid_list:
#         imgA_path = os.path.join(imgA_dir, it)
#         imgB_path = os.path.join(imgB_dir, it)
#         imgA = io.imread(imgA_path)
#         imgB = io.imread(imgB_path)
#         imgA = Data.normalize_image(imgA)
#         imgB = Data.normalize_image(imgB)
        
#         with torch.no_grad():
#               if imgA.shape[0]>opt.crop_size[0] or imgA.shape[1]>opt.crop_size[1]:
#                   imgA_crops, imgB_crops = create_crops(imgA, imgB, opt.crop_size)
#                   crop_num = len(imgA_crops)
#                   print(it+' (%d, %d, %d) cropped into %d patches.'%(imgA.shape[0], imgA.shape[1], imgA.shape[2], crop_num))        
#                   preds = []
#                   for idx in range(crop_num):
#                       cropA = imgA_crops[idx]
#                       cropB = imgB_crops[idx]
#                       tensorA = transF.to_tensor(cropA).unsqueeze(0).to(torch.device('cuda', int(opt.dev_id))).float()
#                       tensorB = transF.to_tensor(cropB).unsqueeze(0).to(torch.device('cuda', int(opt.dev_id))).float()
#                       if opt.encoder == 'ResNet':
#                         output = net(tensorA, tensorB)
#                       else:
#                         output, _, _ = net(tensorA, tensorB) 
#                       output = F.sigmoid(output)
#                       if opt.TTA:
#                           tensorA_v = torch.flip(tensorA, [2])
#                           tensorB_v = torch.flip(tensorB, [2])
#                           if opt.encoder == 'ResNet':
#                               output_v = net(tensorA_v, tensorB_v)
#                           else:
#                               output_v, _, _ = net(tensorA_v, tensorB_v)
#                           output_v = torch.flip(output_v, [2])
#                           output += F.sigmoid(output_v)
                                      
#                           tensorA_h = torch.flip(tensorA, [3])
#                           tensorB_h = torch.flip(tensorB, [3])
#                           if opt.encoder == 'ResNet':
#                               output_h = net(tensorA_h, tensorB_h)
#                           else:
#                               output_h, _, _ = net(tensorA_h, tensorB_h)
#                           output_h = torch.flip(output_h, [3])
#                           output += F.sigmoid(output_h)
                          
#                           tensorA_hv = torch.flip(tensorA, [2,3])
#                           tensorB_hv = torch.flip(tensorB, [2,3])
#                           if opt.encoder == 'ResNet':
#                               output_hv = net(tensorA_hv, tensorB_hv)
#                           else:
#                               output_hv, _, _ = net(tensorA_hv, tensorB_hv)
#                           output_hv = torch.flip(output_hv, [2,3])
#                           output += F.sigmoid(output_hv)            
#                           output = output/4.0
#                       pred = output.cpu().detach().numpy().squeeze()>0.5
#                       preds.append(pred)
#                   print('%d preds calculated...'%len(preds))            
#                   pred = stitch_pred(preds, size_stitch=imgA.shape[:-1])
#               else:
#                   tensorA = transF.to_tensor(imgA).unsqueeze(0).to(torch.device('cuda', int(opt.dev_id))).float()
#                   tensorB = transF.to_tensor(imgB).unsqueeze(0).to(torch.device('cuda', int(opt.dev_id))).float()    
#                   if opt.encoder == 'ResNet':
#                     output = net(tensorA, tensorB)
#                   else:
#                     output, _, _ = net(tensorA, tensorB)   
#                   output = F.sigmoid(output)
#                   if opt.TTA:
#                       tensorA_v = torch.flip(tensorA, [2])
#                       tensorB_v = torch.flip(tensorB, [2])
#                       if opt.encoder == 'ResNet':
#                         output_v = net(tensorA_v, tensorB_v)
#                       else:
#                         output_v, _, _ = net(tensorA_v, tensorB_v)
#                       output_v = torch.flip(output_v, [2])
#                       output += F.sigmoid(output_v)
                                  
#                       tensorA_h = torch.flip(tensorA, [3])
#                       tensorB_h = torch.flip(tensorB, [3])
#                       if opt.encoder == 'ResNet':
#                         output_h = net(tensorA_h, tensorB_h)
#                       else:
#                         output_h, _, _ = net(tensorA_h, tensorB_h)
#                       output_h = torch.flip(output_h, [3])
#                       output += F.sigmoid(output_h)
                      
#                       tensorA_hv = torch.flip(tensorA, [2,3])
#                       tensorB_hv = torch.flip(tensorB, [2,3])
#                       if opt.encoder == 'ResNet':
#                         output_hv = net(tensorA_hv, tensorB_hv)
#                       else:
#                         output_hv, _, _ = net(tensorA_hv, tensorB_hv)
#                       output_hv = torch.flip(output_hv, [2,3])
#                       output += F.sigmoid(output_hv)            
#                       output = output/4.0
#                   pred = output.cpu().detach().numpy().squeeze()>0.5
#         pred_path = os.path.join(opt.pred_dir, it)
#         pred = pred.astype(np.uint8) * 255
#         io.imsave(pred_path, pred)
