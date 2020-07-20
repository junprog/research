import argparse

def opt_args():
    parser = argparse.ArgumentParser(description='Crowd counting base model')
    parser.add_argument(
        '--root_path',
        default='/mnt/hdd02/ShanghaiTech',
        type=str,
        help='Dataset Directry'
    )
    parser.add_argument(
        '--results_path',
        default='/mnt/hdd02/crownd_counting_results/res_d4_freeze_weight_to40_outkernel_1'
    )
    parser.add_argument(
        '--ST_part',
        default='part_B',
        type=str,
        help='The part of ShanghaiTech Dataset'
    )
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        help='Training phase : train, Test phase : test',
    )
    parser.add_argument(
        '--train_json',
        default='train.json',
        type=str,
        help='json file name of training data'
    )
    parser.add_argument(
        '--val_json',
        default='val.json',
        type=str,
        help='json file name of validation data'
    )
    parser.add_argument(
        '--gaussian_std',
        default=15,
        type=int,
        help='standard deviation of gaussian filter'
    )
    parser.add_argument(
        '--crop_scale',
        default=0.5,
        type=float,
        help='parameter of scaledown : (100,80) -- x crop_scale --> (50,40)'
    )
    parser.add_argument(
        '--crop_position',
        default=None,
        type=str,
        help='c:center, tr:top right, tl:top left, br:bottom right, bl:bottom left'
    )
    parser.add_argument(
        '--crop_size_w',
        default=448,
        type=int,
        help='width of crop size'
    )
    parser.add_argument(
        '--crop_size_h',
        default=448,
        type=int,
        help='height of crop size'
    )
    parser.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='Number of threads for multi-thread loading',
    )
    parser.add_argument(
        '--down_scale_num',
        default=4,
        type=int,
        help='Number of Downsampling (e.g. want to use feature maps of Block2 in ResNet: 3)'
    )
    parser.add_argument(
        '--lr',
        default=0.001, #1e-1
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
    parser.add_argument(
        '--checkpoint',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs'
    )

    args = parser.parse_args()

    return args