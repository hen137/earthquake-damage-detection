# Native Imports
import argparse

# Library Imports
import torch

# Custom Imports
from utils.utils import make_hist_graphs
from utils.attributes import hist_attributes

def parse_arguments():
    parser = argparse.ArgumentParser(description="Predicting")

    # parser.add_argument('--model', required=True, choices=['SAM2', 'MaskRCNN']) 
    # parser.add_argument('--dataset', required=True, choices=['KATE_CD', 'KATE_PD', 'Flood'])

    parser.add_argument('--chkpt_file', required=True)
    # parser.add_argument('--chkpt_file', required=False, default='')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    checkpoint = torch.load(args.chkpt_file, weights_only=False)
    
    for hist in hist_attributes:
        make_hist_graphs(checkpoint[hist])
    
if __name__ == '__main__':
    main()
    