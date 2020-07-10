import argparse

def opt_args():
    parser = argparse.ArgumentParser(description='Crowd counting base model')
    parser.add_argument(
        '--path',
        default='/mnt/hdd02/ShanghaiTech',
        type=str,
        help='Dataset Directry'
    )
    parser.add_argument(
        '--lr',
        default=0.1, #1e-1
        type=float,
        help='Learning Rate'
    )
    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help='Batch Size'
    )
    parser.add_argument(
        '--weight_decay',
        default=1e-3,
        type=float,
        help='Weight Decay'
    )
    parser.add_argument(
        '--start_epoch',
        default=1,
        type=int,
        help='Number of the begining epoch'
    )
    parser.add_argument(
        '--num_epochs',
        default=100,
        type=int,
        help='Number of total epochs'
    )

    args = parser.parse_args()

    return args