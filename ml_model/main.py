
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
    batch_train = args.batch_size

    batch_val = args.batch_size_test
    
    num = 100000
    
    uterus_data = uterus(num)
    train, val, test, scaler = uterus_data.create_data(batch_train, batch_val, shuffle = True)
    
    custom_conv_net = BasicNet()
    
    mri_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                        experiment_name=args.experiment_name,
                                        num_epochs=args.num_epochs,
                                        weight_decay_coefficient=args.weight_decay_coefficient,
                                        pk_weight = args.pk_weight, 
                                        curve_weight = args.curve_weight,
                                        lr = args.lr,
                                        train_data=train, 
                                        val_data=val,
                                        test_data=test)  # build an experiment object
    

    
    best_val_model_loss, best_val_model_idx = mri_experiment.run_experiment()  # run experiment and return experiment metrics
    print('\n best val model loss: ', best_val_model_loss)
    print('\n best val model idx: ', best_val_model_idx)

    
    mri_experiment.loss_plot()
    
    mri_experiment.pk_dist()
    
    
    mri_experiment.testing(epoch=best_val_model_idx)
    
if __name__ == '__main__':
    main()
