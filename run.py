import sys
import os
import json

sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix

import torch 
from  torchvision import transforms, models
import torch.nn as nn
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



from create_dataloader import ImportData
from create_dataloader import create_loader

from build_model import resnet152
from build_model import vgg19
from build_model import vgg16
from build_model import train_val_model

from evaluate_test import plot_both_loss
from evaluate_test import test_model
from evaluate_test import test_mae
from evaluate_test import plot_pearson_r

TEST_PATH = '/test/testdata/test.csv'
TRAIN_PATH = '/test/testdata/train.csv'
VAL_PATH = '/test/testdata/val.csv'

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

    model_name = targets[1]
    if model_name == 'vgg16':
        model_test = vgg16()
    elif model_name == 'vgg19':
        model_test = vgg19()
    else:
        model_test = resnet152()
    
    
    if 'test' in targets:
        train_loader, val_loader, test_loader = create_loader(TRAIN_PATH, VAL_PATH, TEST_PATH)
        
        model_trained, train_loss, val_loss = train_val_model(model = model_test,
                                                              batch_size = 8,
                                                              num_epochs = 20,  
                                                              learning_rate = 8e-4,
                                                              train_loader = train_loader, val_loader = val_loader)
        
        plot_both_loss(train_loss, val_loss)
        
        y_test, y_true = test_model(model_trained, test_loader)
        test_mae_out = test_mae(y_test, y_true)
        
        plot_pearson_r(y_test, y_true)
        
        with open('results.txt', 'w') as f:
            f.write('The training loss are: ' + str(train_loss))
            f.write('The validation loss are: ' + str(val_loss))
            f.write('The true log BNPP value are: ' + str(y_true))
            f.write('The inferred log BNPP value are: ' + str(y_test))
            f.write('The test MAE is: ' + str(test_mae_out))
        
    return 

        


if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)