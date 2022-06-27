import datasets as data
import argparse
import os
import logging

parser = argparse.ArgumentParser()

####### Main Parameter: Dataset to use for Training
parser.add_argument('--dataset',      default='cub200',   type=str, help='Dataset to use.')
parser.add_argument('--hub_train_dir',      default='',   type=str, help='hub train data dir')
parser.add_argument('--source_path',  default=os.getcwd()+'/Datasets',         type=str, help='Path to training data.')
parser.add_argument('--bs',                default=112 ,     type=int,   help='Mini-Batchsize to use.')
parser.add_argument('--samples_per_class', default=4,        type=int,   help='Number of samples in one class drawn before choosing the next class. Set to >1 for losses other than ProxyNCA.')
parser.add_argument('--arch',         default='resnet50',  type=str,   help='Network backend choice: resnet50, googlenet.')
parser.add_argument('--kernels',           default=8,        type=int,   help='Number of workers for pytorch dataloader.')

##### Read in parameters
opt = parser.parse_args()

logger = logging.getLogger('MAIN')
dataloaders = data.give_dataloaders(opt.dataset, opt)
for batch in dataloaders['training']:
    print(f"batch type: {type(batch)}, batch: {batch}")