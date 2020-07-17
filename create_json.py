import json
from os.path import join
import glob


if __name__ == '__main__':
    # path to folder that contains images
    img_folder = '/mnt/hdd02/ShanghaiTech/part_B/train_data/images'

    # path to the final json file
    output_train_json = '/mnt/hdd02/ShanghaiTech/part_B/train_data/train.json'
    output_val_json = '/mnt/hdd02/ShanghaiTech/part_B/train_data/val.json'

    img_list_tr = []
    img_list_vl = []

    for i, img_path in enumerate(glob.glob(join(img_folder,'*.jpg'))):

        if i % 4 == 0:
            img_list_vl.append(img_path)
        else:
            img_list_tr.append(img_path)

    with open(output_train_json,'w') as f:
        json.dump(img_list_tr,f)

    with open(output_val_json,'w') as f:
        json.dump(img_list_vl,f)
