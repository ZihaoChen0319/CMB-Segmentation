import torch.nn as nn
import os
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as nnf
import SimpleITK as sitk
import json
import random
import time
import medpy.metric.binary as mmb
from scipy import ndimage
from batchgenerators.utilities.file_and_folder_operations import load_pickle, save_pickle
from nnunet.utilities.nd_softmax import softmax_helper
from PairwiseMeasures_modified import PairwiseMeasures
import medpy.io as mio

from MyDataloader import get_train_cases, get_cmbdataloader
from MyNetwork import UNet3Stage
from MyLoss import FocalLoss, SoftDiceLoss, DC_and_Focal_loss
from ScreenTrainer import ScreenTrainer
from DiscriTrainer import DiscriTrainer


class UNetTrainer(nn.Module):
    def __init__(self, data_path, model_save_path, dataset_path, screen_model_path, discri_model_path,
                 load_screen='current', load_discri='current', device='cuda',
                 all_epoch=50, fold=0, bbox=(32, 32, 24), batch_size=32, loss='soft dice',
                 optimizer='sgd', init_lr=1e-3, decay_exponent=0.9, config=None, if_test=False,
                 random_negatives=200000, aug_num=30, add_fp=False,
                 resample_num=(100000, 100000, 100000), modality=('T1', 'T2', 'T2S')):
        super(UNetTrainer, self).__init__()

        self.bbox = bbox
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.decay_exponent = decay_exponent
        self.all_epoch = all_epoch
        self.config = config
        self.resample_num = resample_num
        self.modality = modality
        self.aug_num = aug_num
        self.fold = fold
        self.random_negatives = random_negatives
        self.screen_trainer = ScreenTrainer(
            data_path=data_path,
            model_save_path=screen_model_path,
            dataset_path=dataset_path,
            device=device,
            fold=fold,
            modality=modality,
            if_test=True)
        self.screen_trainer.load_model(load_screen)
        self.discri_trainer = DiscriTrainer(
            data_path=data_path,
            screen_model_path=screen_model_path,
            load_screen=None,
            model_save_path=discri_model_path,
            dataset_path=dataset_path,
            device=device,
            fold=fold,
            modality=modality,
            if_test=True)
        self.discri_trainer.load_model(load_discri)

        # path define
        self.data_path = data_path
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path + 'fold_%d/' % fold
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # device
        self.device = device

        # load division of data
        if os.path.exists(dataset_path + 'fold_division.json'):
            with open(dataset_path + 'fold_division.json', mode='r') as f:
                splits = json.load(f)
            self.train_list_sub = splits[str(fold)]['train']
            self.val_list_sub = splits[str(fold)]['val']
        else:
            self.train_list_sub = []
            self.val_list_sub = []
            print('Data division is empty!')

        # training and validation samples
        if not if_test:
            self.dataset_name = 'fold_%d/bbox-%d-%d-%d_neg-%d_aug-%d/' % \
                                (fold, self.screen_trainer.bbox[0], self.screen_trainer.bbox[1],
                                 self.screen_trainer.bbox[2], random_negatives, aug_num)
            if not os.path.exists(dataset_path + self.dataset_name):
                os.makedirs(dataset_path + self.dataset_name)
            # load or generate the training samples
            if os.path.exists(dataset_path + self.dataset_name + 'pos.json'):
                with open(dataset_path + self.dataset_name + 'pos.json', mode='r') as f:
                    self.train_cases_pos = json.load(f)
            if os.path.exists(dataset_path + self.dataset_name + 'neg.json'):
                with open(dataset_path + self.dataset_name + 'neg.json', mode='r') as f:
                    self.train_cases_neg = json.load(f)
            else:
                self.train_cases_pos, self.train_cases_neg = get_train_cases(
                    data_path=self.data_path, train_list=self.train_list_sub, bbox=self.bbox, seed=2021,
                    if_translation=True, random_negatives=random_negatives, aug_num=aug_num)
                with open(dataset_path + self.dataset_name + 'pos.json', mode='w') as f:
                    json.dump(self.train_cases_pos, f)
                with open(dataset_path + self.dataset_name + 'neg.json', mode='w') as f:
                    json.dump(self.train_cases_neg, f)
            # load false positive samples
            self.train_cases_fp = []
            if add_fp:
                if os.path.exists(dataset_path + 'fold_%d/fp_%s_current.json' % (self.fold, self.screen_trainer.model_name)):
                    with open(dataset_path + 'fold_%d/fp_%s_current.json' % (self.fold, self.screen_trainer.model_name), mode='r') as f:
                        self.train_cases_fp = json.load(f)
            print('Dataset: pos %d, neg %d, fp %d' %
                  (len(self.train_cases_pos), len(self.train_cases_neg), len(self.train_cases_fp)))
        else:
            self.train_cases_fp = []
            self.train_cases_pos = []
            self.train_cases_neg = []

        # model
        self.model = UNet3Stage(in_channel=len(modality), num_class=2)
        self.model.to(self.device)

        # loss function
        if loss == 'soft dice':
            self.loss_seg = SoftDiceLoss(
                **{'apply_nonlin': None, 'batch_dice': True, 'smooth': 1e-5, 'do_bg': True})
        elif loss == 'dice focal':
            self.loss_seg = DC_and_Focal_loss(
                {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False},
                {'alpha': 0.5, 'gamma': 2, 'smooth': 1e-5})
        else:
            raise ValueError('No such seg loss')

        # optimizer
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=init_lr, momentum=0.99, nesterov=True)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=init_lr)
        else:
            raise ValueError('No such optimizer')

        self.epoch = 1
        self.lr = init_lr
        self.train_metric = [0] * 2
        self.test_metric = [0] * 7

    def train_epoch(self):
        self.model.train()
        train_accum = [0] * 4
        train_cases_fp = self.train_cases_fp.copy()
        train_cases_pos = self.train_cases_pos.copy()
        train_cases_neg = self.train_cases_neg.copy()
        # randomly choose training samples, ensuring that the number of samples is fixed under different conditions
        if len(self.resample_num):
            train_cases_pos = np.random.choice(train_cases_pos, size=self.resample_num[0]).tolist()
            train_cases_neg = np.random.choice(train_cases_neg, size=self.resample_num[1]).tolist()
            if len(train_cases_fp):
                train_cases_fp = np.random.choice(train_cases_fp, size=self.resample_num[2]).tolist()
        data_list = train_cases_pos + train_cases_neg + train_cases_fp
        dataloader = get_cmbdataloader(
            data_path=self.data_path,
            dataset_index=data_list,
            bbox=self.bbox,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=2,
            modality=self.modality,
            if_seg=True
        )
        dataloader = tqdm(dataloader)
        for img_batch, label_batch, mask_batch in dataloader:
            img_batch = img_batch.to(self.device).float()
            mask_batch = mask_batch.to(self.device)

            seg_pred_batch = self.model(img_batch)
            loss = self.loss_seg(seg_pred_batch, mask_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_accum[0] += img_batch.shape[0]
            have_cmb = 1 if torch.sum(label_batch) else 0
            train_accum[1] += have_cmb
            loss_value = loss.detach().cpu().numpy()
            train_accum[2] += loss_value + 1
            train_accum[3] += - loss_value * have_cmb

            self.train_metric[0] = train_accum[2] / train_accum[0]  # loss
            self.train_metric[1] = train_accum[3] / train_accum[1]  # dice
            dataloader.set_description('Epoch: %d, ' % self.epoch + 'train loss %.4f, ' % self.train_metric[0] +
                                       'train dice %.4f, ' % self.train_metric[1])

        return self.train_metric

    def val_epoch(self):
        self.screen_trainer.model.eval()
        self.discri_trainer.model.eval()
        self.model.eval()
        test_accum = [0] * 9
        for pat in self.val_list_sub:
            data_list = []
            for mod in self.modality:
                data_list.append(np.load(self.data_path + '%s/%s_space-T2S_%s.npy' % (pat, pat, mod)))
            cmb, h = mio.load(self.data_path + '%s/%s_space-T2S_CMB.nii.gz' % (pat, pat))
            img = np.stack(data_list, axis=0)
            pred, pred_post, n_obj, pred_init_space, candidates_list, score_init_space = \
                self.screen_trainer.inference(img, patch_size=(160, 160, 80), thresh=0.1, size=2, if_nms=True)
            pred_fp_reduced, reduc_candidates_list, num = self.discri_trainer.inference(img, candidates_list, size=2, thresh=0.5)
            seg_pred = self.inference(img, reduc_candidates_list)
            pe_seg = PairwiseMeasures(ref_img=cmb, seg_img=seg_pred, analysis='microbleeds',
                                      measures=('f1_score', 'tp', 'fn', 'fp', 'mean_diceover',
                                                'absolute_count_difference', 'absolute_volume_difference'),
                                      connectivity=3, pixdim=h.get_voxel_spacing(), empty=True,
                                      threshold=0.5, thresh_assign=1)
            tp_seg, fn_seg, fp_seg, f1_seg = pe_seg.m_dict['tp'][0](), pe_seg.m_dict['fn'][0](), \
                                             pe_seg.m_dict['fp'][0](), pe_seg.m_dict['f1_score'][0]()
            dice = pe_seg.m_dict['mean_diceover'][0]()
            vol_diff = pe_seg.m_dict['absolute_volume_difference'][0]()
            count_diff = pe_seg.m_dict['absolute_count_difference'][0]()
            test_accum[0] += 1  # number of cases
            test_accum[1] += 1 if np.sum(cmb) else 0  # number of cases with CMB
            test_accum[2] += tp_seg
            test_accum[3] += fn_seg
            test_accum[4] += fp_seg
            test_accum[5] += count_diff
            test_accum[6] += f1_seg if np.sum(cmb) else 0
            test_accum[7] += dice if np.sum(cmb) else 0
            test_accum[8] += vol_diff
            print('%s: TP %d, FN %d, FP %d, count diff %.2f, F1 %.2f, Dice %.2f, volume diff %.2f' %
                  (pat, tp_seg, fn_seg, fp_seg, count_diff, f1_seg, dice, vol_diff))

        self.test_metric[0] = test_accum[2]
        self.test_metric[1] = test_accum[3]
        self.test_metric[2] = test_accum[4] / test_accum[0]
        self.test_metric[3] = test_accum[5] / test_accum[0]
        self.test_metric[4] = test_accum[6] / test_accum[1]
        self.test_metric[5] = test_accum[7] / test_accum[1]
        self.test_metric[6] = test_accum[8] / test_accum[0]

        print('Epoch: %d, TP %d, FN %d, avg FP %.2f, count diff %.2f, F1 %.2f, Dice %.2f, volume diff %.2f' %
              (self.epoch, self.test_metric[0], self.test_metric[1], self.test_metric[2],
               self.test_metric[3], self.test_metric[4], self.test_metric[5], self.test_metric[6]))

        return self.test_metric

    def adjust_lr(self):
        """Adjust the learning rate following ‘poly’ policy"""
        self.lr = self.init_lr * (1 - self.epoch / self.all_epoch) ** self.decay_exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        return self.lr

    def save_model(self, force=False):
        """Save the model every epoch(current) and every 5 epochs(epoch_xx)"""
        state = {
            'epoch': self.epoch,
            'state_dict': self.model.state_dict(),
            'config': self.config,
        }
        torch.save(state, self.model_save_path + 'current.pth.tar')
        if self.epoch % 5 == 0 or force:
            torch.save(state, self.model_save_path + 'epoch_%d_%.2f_%.2f_%.2f_%.2f.pth.tar' %
                       (self.epoch, self.test_metric[3], self.test_metric[4], self.test_metric[5], self.test_metric[6]))

    def load_model(self, model_name='current', silent=False):
        all_saved_models = os.listdir(self.model_save_path)
        matched_model = [model for model in all_saved_models if model.startswith(model_name)]
        if len(matched_model) == 1:
            checkpoint = torch.load(self.model_save_path + matched_model[0], map_location={'cuda:0': self.device})
            self.epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(self.device)
            # self.config = checkpoint['config']
            self.adjust_lr()
        elif len(matched_model) > 1:
            raise ValueError('Too many matched models!')
        if not silent:
            print('Segmentation model: %s, device: %s, epoch: %d'
                  % (self.model_save_path + model_name, self.device, self.epoch))

    def inference(self, data: np.ndarray, candidates_list, if_translate=False):
        init_shape = data.shape[1:]
        enlarged_data = np.pad(data, ((0, 0), (self.bbox[0], self.bbox[0]), (self.bbox[1], self.bbox[1]), (self.bbox[2], self.bbox[2])),
                               mode='constant', constant_values=0)
        shape = enlarged_data.shape[1:]
        seg_pred = np.zeros(shape)
        overlap = np.zeros(shape)
        for position in candidates_list:
            position = np.array(position, dtype=int)
            if if_translate:
                x, y, z = position
                position_enlarged = [[i, j, k] for i in [x - 1, x, x + 1] for j in [y - 1, y, y + 1] for k in
                                     [z - 1, z, z + 1]]
            else:
                position_enlarged = [position]
            regions = np.zeros((len(position_enlarged), len(self.modality), self.bbox[0], self.bbox[1], self.bbox[2]))
            for i, pos in enumerate(position_enlarged):
                pos_new = pos + self.bbox
                neighbour = self.get_neighbour(enlarged_data, pos_new)
                regions[i] = neighbour
                # print(neighbour.shape, pos_new, shape)
            regions = torch.tensor(regions, dtype=torch.float32, device=self.device)
            out_seg = self.model(regions).detach()[:, 1]
            for i, pos in enumerate(position_enlarged):
                pos_new = pos + self.bbox
                seg_pred[pos_new[0]-self.bbox[0]//2:pos_new[0]+self.bbox[0]//2,
                         pos_new[1]-self.bbox[1]//2:pos_new[1]+self.bbox[1]//2,
                         pos_new[2]-self.bbox[2]//2:pos_new[2]+self.bbox[2]//2] += out_seg[i].cpu().numpy()
                overlap[pos_new[0]-self.bbox[0]//2:pos_new[0]+self.bbox[0]//2,
                        pos_new[1]-self.bbox[1]//2:pos_new[1]+self.bbox[1]//2,
                        pos_new[2]-self.bbox[2]//2:pos_new[2]+self.bbox[2]//2] += 1
        seg_pred = seg_pred[self.bbox[0]:self.bbox[0]+init_shape[0],
                            self.bbox[1]:self.bbox[1]+init_shape[1],
                            self.bbox[2]:self.bbox[2]+init_shape[2]]
        overlap = overlap[self.bbox[0]:self.bbox[0]+init_shape[0],
                          self.bbox[1]:self.bbox[1]+init_shape[1],
                          self.bbox[2]:self.bbox[2]+init_shape[2]]
        seg_pred /= np.clip(overlap, a_min=1e-5, a_max=1e10)
        return seg_pred

    def get_neighbour(self, data: np.ndarray, position):
        return data[:, position[0]-self.bbox[0]//2:position[0]+self.bbox[0]//2,
                       position[1]-self.bbox[1]//2:position[1]+self.bbox[1]//2,
                       position[2]-self.bbox[2]//2:position[2]+self.bbox[2]//2]



