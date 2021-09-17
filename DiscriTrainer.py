import torch.nn as nn
import os
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as nnf
import json
from PairwiseMeasures_modified import PairwiseMeasures
import medpy.io as mio

from ScreenTrainer import ScreenTrainer
from MyDataloader import get_train_cases, get_cmbdataloader
from MyNetwork import DiscriNet
from MyLoss import FocalLoss


class DiscriTrainer(nn.Module):
    def __init__(self, data_path, model_save_path, dataset_path, screen_model_path, load_screen='current',
                 device='cuda', fold=0, bbox=(24, 24, 20), batch_size=128, loss='ce',
                 optimizer='sgd', init_lr=1e-4, all_epoch=50, decay_exponent=0.9, config=None, if_test=False,
                 random_negatives=200000, aug_num=10, add_fp=False, resample_num=(10000, 10000, 10000),
                 modality=('T1', 'T2', 'T2S')):
        """
        Trainer of the Discrimination Network.
        """
        super(DiscriTrainer, self).__init__()

        self.bbox = bbox
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.decay_exponent = decay_exponent
        self.all_epoch = all_epoch
        self.config = config
        self.resample_num = resample_num
        self.modality = modality
        self.fold = fold
        self.random_negatives = random_negatives
        if load_screen:
            self.screen_trainer = ScreenTrainer(
                data_path=data_path,
                model_save_path=screen_model_path,
                dataset_path=dataset_path,
                device=device,
                fold=fold,
                modality=modality,
                if_test=True)
            self.screen_trainer.load_model(load_screen)
        else:
            self.screen_trainer = None

        # path define
        self.data_path = data_path
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
        self.model = DiscriNet(in_channel=len(modality), num_class=2)
        self.model.to(self.device)

        # loss function
        if loss == 'ce':
            self.loss_fc = nn.CrossEntropyLoss()
        elif loss == 'weighted ce':
            self.loss_fc = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 0.75], device=device))
        elif loss == 'focal loss':
            self.loss_fc = FocalLoss()
        else:
            raise ValueError('No such optimizer')

        # optimizer
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=init_lr, momentum=0.99, nesterov=True)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=init_lr)
        else:
            raise ValueError('No such optimizer')

        self.epoch = 1
        self.lr = init_lr
        self.train_metric = [0] * 3
        self.test_metric = [0] * 4

    def train_epoch(self):
        self.model.train()
        train_accum = [0] * 6
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
            modality=self.modality,
            num_workers=2,
        )
        dataloader = tqdm(dataloader)
        for img_batch, label_batch in dataloader:
            img_batch = img_batch.to(self.device).float()
            label_batch = label_batch.to(self.device)

            pred_batch = self.model(img_batch)
            loss = self.loss_fc(pred_batch, label_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            y_hat = pred_batch.argmax(axis=1).detach().cpu().numpy()
            y = label_batch.detach().cpu().numpy()

            train_accum[0] += img_batch.shape[0]
            train_accum[1] += loss.detach().cpu().numpy() * img_batch.shape[0]
            train_accum[2] += np.sum(y_hat == y)  # acc
            train_accum[3] += np.sum((y_hat == 1) & (y == 1))  # tp
            train_accum[4] += np.sum((y_hat == 1) & (y != 1))  # fp
            train_accum[5] += np.sum((y_hat != 1) & (y == 1))  # fn

            self.train_metric[0] = train_accum[1] / train_accum[0]  # loss
            self.train_metric[1] = train_accum[2] / train_accum[0]  # acc
            self.train_metric[2] = 2 * train_accum[3] / np.clip(2 * train_accum[3] + train_accum[4] + train_accum[5],
                                                                a_min=1e-5, a_max=1e10)  # f1
            dataloader.set_description('Epoch: %d, ' % self.epoch +
                                       'train loss %.4f, ' % self.train_metric[0] +
                                       'train acc %.4f, ' % self.train_metric[1] +
                                       'train f1 %.4f, ' % self.train_metric[2])

        return self.train_metric

    def val_epoch(self):
        self.screen_trainer.model.eval()
        self.model.eval()
        test_accum = [0] * 6
        for pat in self.val_list_sub:
            data_list = []
            for mod in self.modality:
                data_list.append(np.load(self.data_path + '%s/%s_space-T2S_%s.npy' % (pat, pat, mod)))
            cmb, h = mio.load(self.data_path + '%s/%s_space-T2S_CMB.nii.gz' % (pat, pat))
            img = np.stack(data_list, axis=0)
            _, _, n_obj, pred_init_space, candidates_list, score_init_space = \
                self.screen_trainer.inference(img, patch_size=(160, 160, 80), if_nms=True, thresh=0.1, size=2)
            pred_fp_reduced, reduc_candidates_list, num = self.inference(img, candidates_list, thresh=0.5, size=2)
            pe_dis = PairwiseMeasures(ref_img=cmb, seg_img=pred_fp_reduced, analysis='microbleeds',
                                      measures=('f1_score', 'tp', 'fn', 'fp'),
                                      connectivity=3, pixdim=h.get_voxel_spacing(), empty=True,
                                      threshold=0.5, thresh_assign=3)
            tp_dis, fn_dis, fp_dis, f1_dis = pe_dis.m_dict['tp'][0](), pe_dis.m_dict['fn'][0](), \
                                             pe_dis.m_dict['fp'][0](), pe_dis.m_dict['f1_score'][0]()
            test_accum[0] += 1
            test_accum[1] += 1 if np.sum(cmb) else 0
            test_accum[2] += tp_dis
            test_accum[3] += fn_dis
            test_accum[4] += fp_dis
            test_accum[5] += f1_dis if np.sum(cmb) else 0
            print('%s: reduc TP %d, reduc FN %d, reduc FP %d, reduc F1 %.4f' % (pat, tp_dis, fn_dis, fp_dis, f1_dis))

        self.test_metric[0] = test_accum[2]
        self.test_metric[1] = test_accum[3]
        self.test_metric[2] = test_accum[4] / test_accum[0]
        self.test_metric[3] = test_accum[5] / test_accum[1]
        print('Epoch: %d, reduc TP %d, reduc FN %d, reduc avg FP %.2f, reduc F1 %.4f' %
              (self.epoch, self.test_metric[0], self.test_metric[1], self.test_metric[2], self.test_metric[3]))

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
            torch.save(state, self.model_save_path + 'epoch_%d_%d_%d_%.2f_%.2f.pth.tar' %
                       (self.epoch, self.test_metric[0], self.test_metric[1], self.test_metric[2], self.test_metric[3]))

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
            print(matched_model)
            raise ValueError('Too many matched models!')
        if not silent:
            print('Discrimination model: %s, device: %s, epoch: %d'
                  % (self.model_save_path + model_name, self.device, self.epoch))

    def inference(self, data: np.ndarray, candidates_list, thresh=0.5, size=2):
        shape = data.shape[1:]
        tp_list = []
        num = 0
        pred_fp_reduced = np.zeros(shape)
        for position in candidates_list:
            position = np.array(position, dtype=int)
            x, y, z = position
            # Slightly move the candidates to deal with disturbance
            position_enlarged = [[i, j, k] for i in [x-1, x, x+1] for j in [y-1, y, y+1] for k in [z-1, z, z+1]]
            regions = np.zeros((len(position_enlarged), len(self.modality), self.bbox[0], self.bbox[1], self.bbox[2]))
            for i, pos in enumerate(position_enlarged):
                neighbour = self.get_neighbour(data, pos)
                regions[i] = neighbour
            regions = torch.tensor(regions, dtype=torch.float32, device=self.device)
            out_enlarged = self.model(regions).detach()
            out_enlarged = nnf.softmax(out_enlarged, dim=1)[:, 1].cpu().numpy()
            if np.max(out_enlarged) > thresh:
                pos_new = position_enlarged[np.argmax(out_enlarged)]
                tp_list.append(pos_new)
                pred_fp_reduced[pos_new[0]-size//2:pos_new[0]+size//2,
                                pos_new[1]-size//2:pos_new[1]+size//2,
                                pos_new[2]-size//2:pos_new[2]+size//2] = 1
                num += 1
        return pred_fp_reduced, tp_list, num

    def get_neighbour(self, data: np.ndarray, position):
        shape = data.shape[1:]
        if self.bbox[0] // 2 <= position[0] <= shape[0] - self.bbox[0] // 2 and \
                self.bbox[1] // 2 <= position[1] <= shape[1] - self.bbox[1] // 2 and \
                self.bbox[2] // 2 <= position[2] <= shape[2] - self.bbox[2] // 2:
            return data[:, position[0] - self.bbox[0] // 2:position[0] + self.bbox[0] // 2,
                           position[1] - self.bbox[1] // 2:position[1] + self.bbox[1] // 2,
                           position[2] - self.bbox[2] // 2:position[2] + self.bbox[2] // 2]
        else:
            data = np.pad(data, ((0, 0), (self.bbox[0] // 2, self.bbox[0] // 2),
                                 (self.bbox[1] // 2, self.bbox[1] // 2), (self.bbox[2] // 2, self.bbox[2] // 2)),
                          mode='constant', constant_values=0)
            return data[:, position[0]:position[0] + self.bbox[0],
                           position[1]:position[1] + self.bbox[1],
                           position[2]:position[2] + self.bbox[2]]


