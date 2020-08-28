import sys
import json
import os
from os.path import join
import glob

def create_json(opts):
    # path to folder that contains images
    dataset_name = opts.dataset     ### Datasetの種類 (ST_A, ST_B, UCF-QNRF)
    root_path = opts.root_path      ### Datasetがおいてある場所
    phase = opts.phase              ### phase (train or test)

    if os.path.isdir(dataset_name) == False:
        os.mkdir(dataset_name)

    # path to the final json file
    if phase == 'train':
        if os.path.exists(os.path.join(dataset_name, 'train.json')) == False:
            if dataset_name == 'ST_B':
                img_dir = os.path.join(root_path, 'ShanghaiTech/part_B/train_data/images')
            elif dataset_name == 'ST_A':
                img_dir = os.path.join(root_path, 'ShanghaiTech/part_A/train_data/images')
            elif dataset_name == 'UCF-QNRF':
                img_dir = os.path.join(root_path, 'UCF-QNRF_ECCV18/Train')

            output_train_json = os.path.join(dataset_name, 'train.json')
            output_val_json = os.path.join(dataset_name, 'val.json')

            img_list_tr = []
            img_list_vl = []

            for i, img_path in enumerate(glob.glob(join(img_dir,'*.jpg'))):

                if i % 4 == 0: # validation ratio
                    img_list_vl.append(img_path)
                else:
                    img_list_tr.append(img_path)

            with open(output_train_json,'w') as f:
                json.dump(img_list_tr,f)

            with open(output_val_json,'w') as f:
                json.dump(img_list_vl,f)

    elif phase == 'test':
        if os.path.exists(os.path.join(dataset_name, 'test.json')) == False:
            if dataset_name == 'ST_B':
                img_dir = os.path.join(root_path, 'ShanghaiTech/part_B/test_data/images')
            elif dataset_name == 'ST_A':
                img_dir = os.path.join(root_path, 'ShanghaiTech/part_A/test_data/images')
            elif dataset_name == 'UCF-QNRF':
                img_dir = os.path.join(root_path, 'UCF-QNRF_ECCV18/Test')

            output_test_json = os.path.join(dataset_name, 'test.json')

            img_list_test = []

            for i, img_path in enumerate(glob.glob(join(img_dir,'*.jpg'))):
                img_list_test.append(img_path)

            with open(output_test_json,'w') as f:
                json.dump(img_list_test,f)
