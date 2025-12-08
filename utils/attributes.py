hist_attributes = {
    'train_loss_hist': 'List of Floats',
    'train_accuracy_hist': 'List of Floats',   
    'train_precision_hist': 'List of Floats',   
    'train_recall_hist': 'List of Floats',
    'train_f1_hist': 'List of Floats',
    'train_iou_hist': 'List of Floats',
    'train_time': 'Duration in Seconds',
}

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
    'graph_hists': 'Boolean',
}

predict_attributes = {
    'test_loader': 'Pytorch Dataloader',
}