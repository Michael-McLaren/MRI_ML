
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
python ml_model/main.py --num_epochs 100 --experiment_name trained_models/testing
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
    
    num_train = 10000
    batch_train = args.batch_size
    uterus_train_data = uterus(num_train)
    train_data_loader = uterus_train_data.return_dataloader(batch_train)
    
    num_val = 3000
    batch_val = args.batch_size_test
    uterus_val_data = uterus(num_val)
    val_data_loader = uterus_val_data.return_dataloader(batch_val)

    


    
    custom_conv_net = BasicNet()
    
    mri_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                        experiment_name=args.experiment_name,
                                        num_epochs=args.num_epochs,
                                        weight_decay_coefficient=args.weight_decay_coefficient,
                                        train_data=train_data_loader, val_data=val_data_loader,
                                        test_data=None)  # build an experiment object
    
    stat_path, model_path = mri_experiment.run_experiment()  # run experiment and return experiment metrics
    
    
    mri_experiment.loss_plot()
    
    mri_experiment.pk_dist()
    
if __name__ == '__main__':
    main()
