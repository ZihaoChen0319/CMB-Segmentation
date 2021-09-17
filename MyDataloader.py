import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

from Utils import find_binary_object


def get_train_cases(data_path, train_list, bbox=(20, 20, 16), seed=2021, if_translation=False,
                    random_negatives=100000, aug_num=10, silent=False):
    random.seed(seed)
    aug_list = []
    for pat in train_list:
        for i in range(aug_num):
            aug_list.append(pat + '_aug%d' % i)
    train_cases_pos = []
    train_cases_neg = []
    num_pos = 0
    num_neg = 0

    # positive sample
    for pat in train_list:
        if not silent:
            print(pat)
        pat_path = data_path + '%s/%s_space-T2S_CMB.npy' % (pat, pat)
        cmb = np.load(pat_path)
        shape = cmb.shape
        labelmap, num = find_binary_object(cmb)
        for i in range(num):
            region = (labelmap == i + 1).astype(float)
            xx, yy, zz = np.nonzero(region)
            x, y, z = np.median(xx), np.median(yy), np.median(zz)
            if if_translation:
                xyz = [(i, j, k) for i in [x - 4, x - 2, x - 1, x, x + 1, x + 2, x + 4]
                       for j in [y - 4, y - 2, y - 1, y, y + 1, y + 2, y + 4]
                       for k in [z - 4, z - 2, z - 1, z, z + 1, z + 2, z + 4]]
            else:
                xyz = [(x, y, z)]
            for (x, y, z) in xyz:
                if x > shape[0] - bbox[0] // 2 or x < bbox[0] // 2 or \
                        y > shape[1] - bbox[1] // 2 or y < bbox[1] // 2 or \
                        z > shape[2] - bbox[2] // 2 or z < bbox[2] // 2:
                    continue
                x, y, z = int(x), int(y), int(z)
                if np.sum(cmb[x - 1:x + 1, y - 1:y + 1, z - 1:z + 1]):
                    sample = {'pat': pat, 'start': (x, y, z), 'have cmb': 1}
                else:
                    sample = {'pat': pat, 'start': (x, y, z), 'have cmb': 0}

                if sample in train_cases_pos + train_cases_neg:
                    continue
                else:
                    if sample['have cmb']:
                        num_pos += 1
                        train_cases_pos.append(sample)
                    else:
                        num_neg += 1
                        train_cases_neg.append(sample)

    # negative sample
    if not silent:
        print(num_pos, num_neg)
    num_per_patient = (random_negatives - num_neg) // len(train_list)
    for pat in train_list:
        if not silent:
            print(pat)
        pat_path = data_path + '%s/%s_space-T2S_CMB.npy' % (pat, pat)
        cmb = np.load(pat_path)
        shape = cmb.shape
        i = 0
        while i < num_per_patient:
            x = random.randint(bbox[0] // 2, shape[0] - bbox[0] // 2)
            y = random.randint(bbox[1] // 2, shape[1] - bbox[1] // 2)
            z = random.randint(bbox[2] // 2, shape[2] - bbox[2] // 2)
            if np.sum(cmb[x - 1:x + 1, y - 1:y + 1, z - 1:z + 1]):
                sample = {'pat': pat, 'start': (x, y, z), 'have cmb': 1}
            else:
                sample = {'pat': pat, 'start': (x, y, z), 'have cmb': 0}

            if sample not in train_cases_pos + train_cases_neg:
                i += 1
                if sample['have cmb']:
                    num_pos += 1
                    train_cases_pos.append(sample)
                else:
                    num_neg += 1
                    train_cases_neg.append(sample)

    # augmentation of positive sample
    if not silent:
        print(num_pos, num_neg)
    for pat in aug_list:
        if not silent:
            print(pat)
        pat_path = data_path + '%s/%s_space-T2S_CMB.npy' % (pat, pat)
        cmb = np.load(pat_path)
        shape = cmb.shape
        labelmap, num = find_binary_object(cmb)
        for i in range(num):
            region = (labelmap == i + 1).astype(float)
            xx, yy, zz = np.nonzero(region)
            x, y, z = np.median(xx), np.median(yy), np.median(zz)
            if if_translation:
                xyz = [(i, j, k) for i in [x - 4, x - 2, x, x + 2, x + 4]
                       for j in [y - 4, y - 2, y, y + 2, y + 4]
                       for k in [z - 4, z - 2, z, z + 2, z + 4]]
            else:
                xyz = [(x, y, z)]
            for (x, y, z) in xyz:
                if x > shape[0] - bbox[0] // 2 or x < bbox[0] // 2 or \
                        y > shape[1] - bbox[1] // 2 or y < bbox[1] // 2 or \
                        z > shape[2] - bbox[2] // 2 or z < bbox[2] // 2:
                    continue
                x, y, z = int(x), int(y), int(z)
                if np.sum(cmb[x - 1:x + 1, y - 1:y + 1, z - 1:z + 1]):
                    sample = {'pat': pat, 'start': (x, y, z), 'have cmb': 1}
                else:
                    # sample = {'pat': pat, 'start': (x, y, z), 'have cmb': 0}
                    continue

                if sample not in train_cases_pos + train_cases_neg:
                    i += 1
                    if sample['have cmb']:
                        num_pos += 1
                        train_cases_pos.append(sample)
                    else:
                        num_neg += 1
                        train_cases_neg.append(sample)

    return train_cases_pos, train_cases_neg


class CMBDataset(Dataset):
    def __init__(self, data_path, dataset_index, bbox, modality=('T1', 'T2', 'T2S'), if_seg=False):
        super(CMBDataset, self).__init__()

        self.data_path = data_path
        self.dataset_index = dataset_index
        self.bbox = bbox
        self.modality = modality
        self.if_seg = if_seg

    def __len__(self):
        return len(self.dataset_index)

    def __getitem__(self, item):
        sample = self.dataset_index[item]
        pat = sample['pat']
        position = sample['start']
        data_mmap = []
        for mod in self.modality:
            data_mmap.append(np.load(self.data_path + '%s/%s_space-T2S_%s.npy' % (pat, pat, mod), mmap_mode='r'))
        shape = data_mmap[0].shape
        img_list = []
        if self.bbox[0] // 2 <= position[0] <= shape[0] - self.bbox[0] // 2 and \
                self.bbox[1] // 2 <= position[1] <= shape[1] - self.bbox[1] // 2 and \
                self.bbox[2] // 2 <= position[2] <= shape[2] - self.bbox[2] // 2:
            for data in data_mmap:
                img_list.append(data[position[0] - self.bbox[0] // 2:position[0] + self.bbox[0] // 2,
                                     position[1] - self.bbox[1] // 2:position[1] + self.bbox[1] // 2,
                                     position[2] - self.bbox[2] // 2:position[2] + self.bbox[2] // 2])
            label = sample['have cmb']
            img = np.stack(img_list, axis=0)
            if self.if_seg:
                cmb_mmap = np.load(self.data_path + '%s/%s_space-T2S_CMB.npy' % (pat, pat), mmap_mode='r')
                mask = cmb_mmap[position[0] - self.bbox[0] // 2:position[0] + self.bbox[0] // 2,
                                position[1] - self.bbox[1] // 2:position[1] + self.bbox[1] // 2,
                                position[2] - self.bbox[2] // 2:position[2] + self.bbox[2] // 2]
                return img, label, mask
            else:
                return img, label
        else:
            if self.if_seg:
                return np.zeros((len(self.modality), self.bbox[0], self.bbox[1], self.bbox[2])), 0, \
                       np.zeros((self.bbox[0], self.bbox[1], self.bbox[2]))
            else:
                return np.zeros((len(self.modality), self.bbox[0], self.bbox[1], self.bbox[2])), 0


def get_cmbdataloader(data_path, dataset_index, bbox, batch_size, modality=('T1', 'T2', 'T2S'), if_seg=False,
                      shuffle=True, pin_memory=True, num_workers=8):
    dataset = CMBDataset(data_path=data_path, dataset_index=dataset_index, bbox=bbox, modality=modality, if_seg=if_seg)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers)
    return dataloader



