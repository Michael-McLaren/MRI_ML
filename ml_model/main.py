
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_generation import uterus
from arg_extractor import get_args
from experiment_build import ExperimentBuilder
from model_architectures import BasicNet
import os 

'''
example command for running on terminal
ython ml_model/main.py --num_epochs 100 --experiment_name trained_models/curve
'''


def main():
    args = get_args()  # get arguments from command line
    #rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
    #torch.manual_seed(seed=args.seed)  # sets pytorch's seed
    
    # set up data augmentation transforms for training and testing

    #figure out how to do this
    #transform_test = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #])
    name = args.experiment_name.split('/')
    name = name[-1]
    
    num_train = 100000
    batch_train = args.batch_size

    
    num_val = 30000
    batch_val = args.batch_size_test

    '''
    batch_train = args.batch_size
    batch_val = args.batch_size_test

    num_train = 1
    uterus_train_data = uterus(num_train)
    real_x = uterus_train_data.real_x
    real_y = uterus_train_data.real_y

    train_ind = int(real_x.shape[0]*0.6)
    val_ind = int(real_x.shape[0]*0.8)
    
    train_x = real_x[:train_ind]
    train_x = uterus_train_data.normalise(train_x)
    train_y = real_y[:train_ind]
    
    val_x = real_x[train_ind:val_ind]
    val_x = uterus_train_data.normalise(val_x)
    val_y = real_y[train_ind:val_ind]
    
    train_data_loader = uterus_train_data.create_dataloader(train_x, train_y, batch_train)
    val_data_loader = uterus_train_data.create_dataloader(val_x, val_y, batch_val)
    '''
    
    custom_conv_net = BasicNet()
    
    mri_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                        experiment_name=args.experiment_name,
                                        num_epochs=args.num_epochs,
                                        weight_decay_coefficient=args.weight_decay_coefficient,
                                        pk_weight = args.pk_weight, 
                                        curve_weight = args.curve_weight,
                                        lr = args.lr,
                                        train_data=None, 
                                        val_data=None,
                                        test_data=None)  # build an experiment object
    
    mri_experiment.create_data(batch_train, batch_val, num_train, num_val)
    
    folder_path, model_path = mri_experiment.run_experiment()  # run experiment and return experiment metrics
    print('\n stat_path: ', folder_path)
    print('\n model_path: ', model_path)

    
    mri_experiment.loss_plot()
    
    mri_experiment.pk_dist()
    
    
    in_epoch = int(input('Input epoch 0-99: '))
    mri_experiment.testing(epoch=in_epoch)
    
if __name__ == '__main__':
    main()
