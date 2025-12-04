# Native Imports
import random

# Library Imports
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from torchvision.utils import draw_segmentation_masks
from scipy import stats

# Custom Imports

train_attributes = {
    'train_loader': 'Pytorch Dataloader',
    'validation_loader': 'Pytorch Dataloader',
    'optimizer': 'Pytorch Optimizer',
    'lr_scheduler': 'Pytorch LR Scheduler',
    'use_scaler': 'Boolean',
    'print_freq': 'Integer',
    'val_freq': 'Integer',
    'checkpoint_dir': 'Directory Path',
    'dataset': 'Dataset Name',
}

predict_attributes = {
    'test_loader': 'Pytorch Dataloader',
}

def align_dims(np_input, expected_dims=2):
    dim_input = len(np_input.shape)
    np_output = np_input
    
    if dim_input>expected_dims:
        np_output = np_input.squeeze(0)
    elif dim_input<expected_dims:
        np_output = np_input.unsqueeze(0)   
         
    assert len(np_output.shape) == expected_dims
    
    return np_output

def get_confusion_matrix_elements(pred, label):
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)
    pred = (pred>= 0.5)
    label = (label>= 0.5)
    
    TP = (pred * label)
    FP = (pred * (~label))
    FN = ((~pred) * (label))
    TN = ((~pred) * (~label))
    
    return TP, TN, FP, FN

def binary_accuracy(pred, label):
    TP, TN, FP, FN = get_confusion_matrix_elements(pred, label)
    
    TP = float(TP.sum())
    TN = float(TN.sum())
    FP = float(FP.sum())
    FN = float(FN.sum())
    
    precision = TP / (TP+FP+1e-10)
    recall = TP / (TP+FN+1e-10)
    IoU = TP / (TP+FP+FN+1e-10)
    acc = (TP+TN) / (TP+FP+FN+TN)
    F1 = 0
    
    if acc>0.999 and TP==0:
        precision=1
        recall=1
        IoU=1
    if precision>0 and recall>0:
        F1 = stats.hmean([precision, recall])
        
    return acc, precision, recall, F1, IoU

def get_pred_mask_as_rgb(pred_mask, gt_mask):
    TP, TN, FP, FN = get_confusion_matrix_elements(pred_mask, gt_mask)
    
    h, w = gt_mask.shape[1:]
    rgb_mask = torch.zeros((h, w, 3))
    rgb_mask[TP == 1] = [1, 1, 1]  
    rgb_mask[TN == 1] = [0, 0, 0]  
    rgb_mask[FP == 1] = [1, 0, 0]  
    rgb_mask[FN == 1] = [0, 0, 1]
    
    return rgb_mask

def visualize_predictions(output_format, images, predictions, targets):
    if output_format == 'overlay':
        for i in range(len(images)):
            gt = draw_segmentation_masks(image=images[i], masks=(targets[i]['masks'].squeeze(0) > 0), alpha=0.3, colors='red')
            pred = draw_segmentation_masks(image=images[i], masks=predictions[i]['mask'], alpha=0.3, colors='blue')
            
            plt.imshow(torch.cat((gt, pred), 2).permute(1, 2, 0))
            plt.show()
            
    elif output_format == 'rand_5':
        row_labels = ['Image', 'Ground Truth', 'Prediction']
        legend_labels = ["True Negative", "True Positive", "False Positive", "False Negative"]
        legend_colors = ["black", "white", "red", "blue"]
        
        idxs = random.sample(range(len(images)), 5)
        idxs.sort()
        
        predictions = [predictions[i] for i in idxs]
        
        sample_names = [f'{i}.png' for i in idxs]
        images = [images[i] for i in idxs]
        gt_masks = [targets[i]['masks'] for i in idxs]
        pred_masks = [get_pred_mask_as_rgb(prediction['mask'], gt_mask) for prediction, gt_mask in zip(predictions, gt_masks)]
        
        fig, axes = plt.subplots(len(row_labels), len(idxs), figsize=(2 * len(idxs), 6))
        
        for i in range(len(idxs)):
            axes[0, i].imshow(images[i].permute(1, 2, 0))
            axes[0, i].axis('off')
            
            axes[1, i].imshow((gt_masks[i] > 0).permute(1, 2, 0), cmap='gray')
            axes[1, i].axis('off')

            axes[2, i].imshow(pred_masks[i])
            axes[2, i].axis('off')
        
        for row, label in enumerate(row_labels):
            axes[row, 0].annotate(
                label, 
                xy=(-0.1, 0.5), 
                xycoords="axes fraction", 
                fontsize=12, 
                fontweight="normal", 
                rotation=0, 
                ha="right", 
                va="center"
            )

        for col, label in enumerate(sample_names):
            axes[0, col].annotate(
                label, 
                xy=(0.5, 1.05), 
                xycoords="axes fraction", 
                fontsize=14, 
                fontweight="normal", 
                rotation=0, 
                ha="center", 
                va="bottom"
            )
        
        patches = [mlines.Line2D([], [], color=legend_colors[i], marker='s', markersize=6, 
                markeredgecolor='black', markeredgewidth=1.5, linestyle='None', 
                label=legend_labels[i]) for i in range(len(legend_labels))]

        fig.legend(
            handles=patches, 
            bbox_to_anchor=(0.5, 0.05), 
            loc="lower center",  
            ncol=4,  
            fontsize=12,  
            frameon=True,
            markerscale=2 
        )
        
        plt.show()

def compare_predictions(images, targets, num_samples, **model_predictions):
    row_labels = ['Image', 'Ground Truth'] + list(model_predictions.keys())
    legend_labels = ["True Negative", "True Positive", "False Positive", "False Negative"]
    legend_colors = ["black", "white", "red", "blue"]
    
    idxs = random.sample(range(len(images)), num_samples)
    idxs.sort()
    
    sample_names = [f'{i}.png' for i in idxs]
    images = [images[i] for i in idxs]
    gt_masks = [targets[i]['masks'] for i in idxs]
    
    pred_masks = {model: [get_pred_mask_as_rgb(prediction['mask'], gt_mask) for prediction, gt_mask in zip(predictions, gt_masks)] for model, predictions in model_predictions.items()}
    
    fig, axes = plt.subplots(len(row_labels), len(idxs), figsize=(2 * len(idxs), 2.5 * len(row_labels)))
    
    for row in range(len(row_labels)):
        if row == 0:
            for i in range(len(idxs)):
                axes[0, i].imshow(images[i].permute(1, 2, 0))
                axes[0, i].axis('off')
        elif row == 1:
            for i in range(len(idxs)):
                axes[1, i].imshow((gt_masks[i] > 0).permute(1, 2, 0), cmap='gray')
                axes[1, i].axis('off')
        else:
            model_name = row_labels[row]
            pred_masks_model = pred_masks[model_name]
            for i in range(len(idxs)):
                axes[row, i].imshow(pred_masks_model[i])
                axes[row, i].axis('off')
        
    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(
            label, 
            xy=(-0.1, 0.5), 
            xycoords="axes fraction", 
            fontsize=12, 
            fontweight="normal", 
            rotation=0, 
            ha="right", 
            va="center"
        )

    for col, label in enumerate(sample_names):
        axes[0, col].annotate(
            label, 
            xy=(0.5, 1.05), 
            xycoords="axes fraction", 
            fontsize=14, 
            fontweight="normal", 
            rotation=0, 
            ha="center", 
            va="bottom"
        )
    
    patches = [mlines.Line2D([], [], color=legend_colors[i], marker='s', markersize=6, 
            markeredgecolor='black', markeredgewidth=1.5, linestyle='None', 
            label=legend_labels[i]) for i in range(len(legend_labels))]

    fig.legend(
        handles=patches, 
        bbox_to_anchor=(0.5, 0.05), 
        loc="lower center",  
        ncol=4,  
        fontsize=12,  
        frameon=True,
        markerscale=2 
    )
    
    plt.show()