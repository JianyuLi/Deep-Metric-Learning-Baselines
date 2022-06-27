import hub
import numpy as np
from hub.util.iterable_ordered_dict import IterableOrderedDict
import torchvision.transforms as transforms
import random
import torch
import logging

class TransToImage(object):
    def __init__(self):
        return
    
    def __call__(self, image_tensor):
        img = transforms.ToPILImage()(image_tensor)
        img_2 = img.convert('RGB')
        return img_2

def get_transform(arch):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    transf_list = []
    transf_list.extend([TransToImage(), transforms.RandomResizedCrop(size=224) if arch=='resnet50' else transforms.RandomResizedCrop(size=227),
        transforms.RandomHorizontalFlip(0.5)])
    transf_list.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(transf_list)

class HubTrainDataSet(torch.utils.data.IterableDataset):
    def __init__(self, dataset_path, items_per_class, batch_size, num_workers = 4, arch='resnet50', s3_creds = None):
        self.ds = hub.load(dataset_path)
        #self.ds = hub.load(dataset_path, creds=s3_creds)
        self.ds_len = len(self.ds)
        print(f"ds len: {self.ds_len}")

        self.num_workers = num_workers
        self.items_per_class = items_per_class
        self.batch_size = batch_size

        self.data_loader = self.ds.pytorch(num_workers = self.num_workers, batch_size = self.batch_size, 
            drop_last = False, collate_fn = hub_collate_fn, pin_memory=False, shuffle = True, buffer_size = 1024,
            use_local_cache = True, transform = {'image': get_transform(arch), 'label': None})
        self.batch_iter = iter(self.data_loader)

        self.item_samples_by_class = {}
        self.item_num_by_class = {}
        self.valid_class_set = set()
        self.classes_visited = []

        self.batch_imgs = []
        self.batch_labels = []
        self.batch_output_num = 0

        self.is_init = True
        self.logger = logging.getLogger('MAIN')

    def __iter__(self):
        while True:
            checked_items_total_num = 0
            for _, item_nums in self.item_num_by_class.items():
                checked_items_total_num += item_nums
            self.logger.info(f"num of valid class: {len(self.valid_class_set)}, num of items: {checked_items_total_num}, num of collect batched imgs: {len(self.batch_imgs)}")
            self.logger.info(f"class visited: {self.classes_visited}, batch_labels: {self.batch_labels}")

            if len(self.batch_imgs) >= self.batch_size:
                batch_samples = [torch.Tensor(self.batch_labels[0:self.batch_size]).int(), torch.stack(self.batch_imgs[0:self.batch_size])]
                #batch_samples = [torch.Tensor(self.batch_labels[0:self.batch_size], dtype=torch.int64), torch.stack(self.batch_imgs[0:self.batch_size])]
                self.batch_imgs = self.batch_imgs[self.batch_size:]
                self.batch_labels = self.batch_labels[self.batch_size:]
                self.batch_output_num += 1
                yield batch_samples

            if self.valid_class_set:
                class_picked = True
                # pick valid classes
                if self.is_init:
                    choose_class_id = random.choice(tuple(self.valid_class_set))
                    self.classes_visited = [choose_class_id, choose_class_id]
                    self.is_init = False
                else:
                    class_visited_set = set(self.classes_visited)
                    candidate_class_set = self.valid_class_set - class_visited_set
                    if len(candidate_class_set) > 0:
                        choose_class_id = random.choice(tuple(candidate_class_set))
                        self.classes_visited = self.classes_visited[1:]+[choose_class_id]
                    else:
                        class_picked = False

                if class_picked:
                    self.logger.info(f"new class picked: {choose_class_id}")
                    self.item_num_by_class[choose_class_id] -= self.items_per_class
                    if self.item_num_by_class[choose_class_id] < self.items_per_class:
                        self.valid_class_set.remove(choose_class_id)
                        if self.item_num_by_class[choose_class_id] <= 0:
                            self.item_num_by_class.pop(choose_class_id, None)
                
                    # pop sample
                    choose_items_indexs = np.random.choice(range(len(self.item_samples_by_class[choose_class_id])), self.items_per_class,
                        replace=False)
                    self.batch_imgs.extend([self.item_samples_by_class[choose_class_id][item_idx] for item_idx in choose_items_indexs])
                    self.batch_labels.extend([choose_class_id] * self.items_per_class)
                    for item_idx in sorted(choose_items_indexs, reverse=True):
                        del self.item_samples_by_class[choose_class_id][item_idx]

                    continue

            while True:
                try:
                    next_batch = next(self.batch_iter)
                except StopIteration:
                    if self.batch_output_num * self.batch_size > self.ds_len:
                        self.logger.info(f"hub dataset iterator completed!")
                        return
                    self.batch_iter = iter(self.data_loader)
                    self.logger.info(f"load iterator is over!")
                    continue
                break

            for idx in range(len(next_batch['image'])):
                class_id = next_batch['label'][idx][0].item()
                image = next_batch['image'][idx]

                if class_id not in self.item_samples_by_class:
                    self.item_samples_by_class[class_id] = []

                # add item into array
                self.item_samples_by_class[class_id].append(image)

                if class_id not in self.item_num_by_class:
                    self.item_num_by_class[class_id] = 1
                else:
                    self.item_num_by_class[class_id] += 1

                if self.item_num_by_class[class_id] >= self.items_per_class:
                    if class_id not in self.valid_class_set:
                        self.valid_class_set.add(class_id)
            del next_batch

            
def hub_collate_fn(batch):
    elem = batch[0]

    if isinstance(elem, IterableOrderedDict):
        return IterableOrderedDict(
            (key, hub_collate_fn([d[key] for d in batch])) for key in elem.keys()
        )

    if isinstance(elem, np.ndarray) and elem.size > 0 and isinstance(elem[0], str):
        batch = [it[0] for it in batch]

    return torch.utils.data._utils.collate.default_collate(batch)

def hub_collate_fn_2(batch):
    elem = batch[0]

    if isinstance(elem, IterableOrderedDict):
        return IterableOrderedDict(
            (key, hub_collate_fn_2([d[key] for d in batch])) for key in elem.keys()
        )

    if isinstance(elem, np.ndarray) and elem.size > 0 and isinstance(elem[0], str):
        batch = [it[0] for it in batch]

    if isinstance(elem, list):
        batched_list = []
        for elem in batch:
            batched_list.extend(elem)
        return batched_list

    return torch.utils.data._utils.collate.default_collate(batch)


if __name__ == "__main__":
    logger = logging.getLogger('MAIN')
    data_set = HubTrainDataSet(dataset_path = "/Users/joanny.li/dev/hub_image/cub200_train", items_per_class = 4, batch_size=16)
    data_loader = torch.utils.data.DataLoader(data_set, collate_fn = hub_collate_fn_2, pin_memory = True, drop_last = True)
    data_iter = iter(data_loader)
    while True:
        batch = next(data_iter)
        logger.info(f"type: {type(batch)}, batch: {batch}")
