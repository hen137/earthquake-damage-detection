from scipy import stats

def align_dims(np_input, expected_dims=2):
    dim_input = len(np_input.shape)
    np_output = np_input
    
    if dim_input>expected_dims:
        np_output = np_input.squeeze(0)
    elif dim_input<expected_dims:
        np_output = np_input.unsqueeze(0)   
         
    assert len(np_output.shape) == expected_dims
    
    return np_output

def binary_accuracy(pred, label):
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)
    pred = (pred>= 0.5)
    label = (label>= 0.5)
    
    TP = float((pred * label).sum())
    FP = float((pred * (~label)).sum())
    FN = float(((~pred) * (label)).sum())
    TN = float(((~pred) * (~label)).sum())
    
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
