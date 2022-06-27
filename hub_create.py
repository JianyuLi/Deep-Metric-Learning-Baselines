# Copyright 2019 Karsten Roth and Biagio Brattoli
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

################# LIBRARIES ###############################
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import hub
import numpy as np


def create_CUB200_hub_datasets(source_path):
    """
    This function generates a training, testing and evaluation dataloader for Metric Learning on the CUB-200-2011 dataset.
    For Metric Learning, the dataset classes are sorted by name, and the first half used for training while the last half is used for testing.
    So no random shuffling of classes.

    Args:
        opt: argparse.Namespace, contains all traininig-specific parameters.
    Returns:
        dict of PyTorch datasets for training, testing and evaluation.
    """
    image_sourcepath  = source_path+'/images'
    #Find available data classes.
    image_classes = sorted([x for x in os.listdir(image_sourcepath) if '._' not in x], key=lambda x: int(x.split('.')[0]))
    print(f"image_classes: {image_classes}")
    
    class_names = [x.split('.')[-1] for x in image_classes]

    #Generate a list of tuples (class_label, image_path)
    image_list    = {int(key.split('.')[0]):sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key) if '._' not in x]) for key in image_classes}
    print(f"image_list_keys: {image_list.keys()}")

    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    #Image-dict of shape {class_idx:[list of paths to images belong to this class] ...}
    image_dict    = {}
    for key, img_path in image_list:
        key = key-1
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)

    keys = sorted(list(image_dict.keys()))
    total_num = np.sum([len(image_dict[key]) for key in image_dict.keys()])
    print(f"keys: {keys}, total_num: {total_num}")


    #Following "Deep Metric Learning via Lifted Structured Feature Embedding", we use the first half of classes for training.
    train,test = keys[:len(keys)//2], keys[len(keys)//2:]
    train_image_dict, val_image_dict = {key:image_dict[key] for key in train},{key:image_dict[key] for key in test}

    ds_train = hub.empty("/Users/joanny.li/dev/hub_image/cub200_train", overwrite=True)
    with ds_train:
        ds_train.create_tensor('image', htype = 'image', sample_compression = 'jpeg')
        ds_train.create_tensor('label')

        for label_idx, image_path_list in train_image_dict.items():
            for image_path in image_path_list:
                print(f"image: {image_path}, label: {label_idx}")
                ds_train.append({'image': hub.read(image_path), 'label': np.uint32(label_idx)})
    print(f"train sample num: {len(ds_train)}")

    ds_val = hub.empty("/Users/joanny.li/dev/hub_image/cub200_test", overwrite=True)
    with ds_val:
        ds_val.create_tensor('image', htype = 'image', sample_compression = 'jpeg')
        ds_val.create_tensor('label')

        for label_idx, image_path_list in val_image_dict.items():
            for image_path in image_path_list:
                print(f"image: {image_path}, label: {label_idx}")
                ds_val.append({'image': hub.read(image_path), 'label': np.uint32(label_idx)})
    print(f"val sample num: {len(ds_val)}")

    return


if __name__ == "__main__":
    create_CUB200_hub_datasets(sys.argv[1])
