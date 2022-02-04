import argparse


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to my new helper script boi')

    parser.add_argument('--batch_size', nargs="?", type=int, default=100, help='Batch_size for experiment')
    
    parser.add_argument('--batch_size_test', nargs="?", type=int, default=1000, help='Batch_size for testing')

    
    parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='Total number of epochs for model training')
    
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=0.01,
                        help='Weight decay to use for Adam')

    args = parser.parse_args()
    print(args)
    return args