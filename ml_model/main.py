
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_generation import uterus
from arg_extractor import get_args
from experiment_build_rnn import ExperimentBuilder
from model_architectures import BasicNet, NeuralNet, NeuralNet_Dropout, GRUModel
import os 

'''
example command for running on terminal
ython ml_model/main.py --num_epochs 100 --experiment_name trained_models/curve
'''


def main():
    
    args = get_args()  
    
    batch_train = args.batch_size

    batch_val = args.batch_size_test
    
    num = 100000
    
    uterus_data = uterus(num, smooth = True)
    train, val, test, scaler = uterus_data.create_synth_data(batch_train, batch_val, shuffle = True)
    
    enc_sizes = [150, 200, 200, 200, 200, 200]
    input_dim = 1
    hidden_dim =  100
    layer_dim = 3
    output_dim = 3
    dropout_prob = 0.5
    custom_conv_net = GRUModel(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob)
    print(custom_conv_net)
    
    mri_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                        experiment_name=args.experiment_name,
                                        num_epochs=args.num_epochs,
                                        weight_decay_coefficient=args.weight_decay_coefficient,
                                        pk_weight = args.pk_weight, 
                                        curve_weight = args.curve_weight,
                                        lr = args.lr,
                                        train_data=train, 
                                        val_data=val,
                                        test_data=test, 
                                        scaler = scaler,
                                        save = True,
                                        use_gpu = True)  # build an experiment object
    

    
    best_val_model_loss, best_val_model_idx = mri_experiment.run_experiment()  # run experiment and return experiment metrics
    print('\n best val model loss: ', best_val_model_loss)
    print('\n best val model idx: ', best_val_model_idx)
    '''
    experiment_saved_models = os.path.abspath(os.path.join('trained_models/gru', "saved_models"))
    mri_experiment.load_model(experiment_saved_models, 'train_model', 9)
    '''
    mri_experiment.loss_plot()
    
    #larger is better
    stats, res = mri_experiment.pk_dist()
    str_ = '\n E stat: {:.3f} \n Fp stat: {:.3f} \n vp stat: {:.3f}'.format(stats[0], stats[1], stats[2])
    str_1 = '\n E res: {:.3f} \n Fp res: {:.3f} \n vp res: {:.3f}'.format(res[0], res[1], res[2])
    print(str_)
    print(str_1)
    
    
    
    mri_experiment.testing1()
    
    
    
if __name__ == '__main__':
    main()
