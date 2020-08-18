import sys
import json
import os
from os.path import join
import glob


if __name__ == '__main__':
    # path to folder that contains images
    img_folder = sys.argv[1] ### datas dir path (~/images)
    phase = sys.argv[2] ### phase (train or test)

    if os.path.isdir('json_file') == False:
        os.mkdir('json_file')

    # path to the final json file
    if phase == 'train':
        output_train_json = 'json_file/train.json'
        output_val_json = 'json_file/val.json'

        img_list_tr = []
        img_list_vl = []

        for i, img_path in enumerate(glob.glob(join(img_folder,'*.jpg'))):

            if i % 4 == 0: # validation ratio
                img_list_vl.append(img_path)
            else:
                img_list_tr.append(img_path)

        with open(output_train_json,'w') as f:
            json.dump(img_list_tr,f)

        with open(output_val_json,'w') as f:
            json.dump(img_list_vl,f)
    elif phase == 'test':
        output_test_json = 'json_file/test.json'

        img_list_test = []

        for i, img_path in enumerate(glob.glob(join(img_folder,'*.jpg'))):
            img_list_test.append(img_path)

        with open(output_test_json,'w') as f:
            json.dump(img_list_test,f)
