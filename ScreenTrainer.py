import torch.nn as nn
import os
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as nnf
import SimpleITK as sitk
import json
from scipy import ndimage
import medpy.io as mio

from Utils import find_binary_object
from MyDataloader import get_train_cases, get_cmbdataloader
from MyNetwork import ScreenNet
from MyLoss import FocalLoss
from PairwiseMeasures_modified import PairwiseMeasures


class ScreenTrainer(nn.Module):
    def __init__(self, data_path, model_save_path, dataset_path, device='cuda', all_epoch=50,
                 fold=0, bbox=(20, 20, 16), batch_size=32, loss='ce',
                 optimizer='sgd', init_lr=1e-3, decay_exponent=0.9, config=None, if_test=False,
                 random_negatives=1e5, aug_num=10, add_fp=False,
                 resample_num=(10000, 10000, 10000), modality=('T1', 'T2', 'T2S')):
        """
        Trainer of the Screening Network.
        """
        super(ScreenTrainer, self).__init__()

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

        # path define
        self.data_path = data_path
        self.dataset_path = dataset_path
        self.model_name = model_save_path.split('/')[-2]
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
                                (fold, self.bbox[0], self.bbox[1], self.bbox[2], random_negatives, aug_num)
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
                if os.path.exists(dataset_path + 'fold_%d/fp_%s_current.json' % (self.fold, self.model_name)):
                    with open(dataset_path + 'fold_%d/fp_%s_current.json' % (self.fold, self.model_name), mode='r') as f:
                        self.train_cases_fp = json.load(f)
            print('Dataset: pos %d, neg %d, fp %d' %
                  (len(self.train_cases_pos), len(self.train_cases_neg), len(self.train_cases_fp)))
        else:
            self.train_cases_fp = []
            self.train_cases_pos = []
            self.train_cases_neg = []

        # model
        self.model = ScreenNet(is_fc=False, in_channel=len(modality), num_class=2)
        self.model.to(self.device)

        # loss function
        if loss == 'ce':
            self.loss_fc = nn.CrossEntropyLoss()
        elif loss == 'weighted ce':
            self.loss_fc = nn.CrossEntropyLoss(weight=torch.tensor([0.25, 0.75], device=device))
        elif loss == 'focal loss':
            self.loss_fc = FocalLoss(alpha=0.25, gamma=2, num_classes=2)
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
            num_workers=2,
            modality=self.modality
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
        self.model.eval()
        test_accum = [0] * 6
        for pat in self.val_list_sub:
            data_list = []
            for mod in self.modality:
                data_list.append(np.load(self.data_path + '%s/%s_space-T2S_%s.npy' % (pat, pat, mod)))
            img = np.stack(data_list, axis=0)
            cmb, h = mio.load(self.data_path + '%s/%s_space-T2S_CMB.nii.gz' % (pat, pat))
            pred, pred_post, n_obj, pred_init_space, candidates_list, score_init_space = \
                self.inference(img, patch_size=(160, 160, 80), thresh=0.1, size=2, if_nms=True)
            pe = PairwiseMeasures(ref_img=cmb, seg_img=pred_init_space, analysis='microbleeds',
                                  measures=('f1_score', 'tp', 'fn', 'fp'),
                                  connectivity=3, pixdim=h.get_voxel_spacing(), empty=True,
                                  threshold=0.5, thresh_assign=3)
            tp, fn, fp = pe.m_dict['tp'][0](), pe.m_dict['fn'][0](), pe.m_dict['fp'][0]()
            f1 = pe.m_dict['f1_score'][0]()
            test_accum[0] += 1
            test_accum[1] += tp
            test_accum[2] += fn
            test_accum[3] += fp
            test_accum[4] += f1 if np.sum(cmb) else 0
            test_accum[5] += 1 if np.sum(cmb) else 0
            print('%s: TP %d, FN %d, FP %d, F1 %.4f' % (pat, tp, fn, fp, f1))
        self.test_metric[0] = test_accum[1]  # TP
        self.test_metric[1] = test_accum[2]  # FN
        self.test_metric[2] = test_accum[3] / test_accum[0]  # avg FP
        self.test_metric[3] = test_accum[4] / test_accum[5]  # avg F1
        print('Epoch: %d, TP %d, FN %d, avg FP %.4f, avg F1 %.4f' %
              (self.epoch, self.test_metric[0], self.test_metric[1], self.test_metric[2], self.test_metric[3]))

        return self.test_metric

    def get_fp(self, thresh=0.1, if_aug=False):
        """Obtain false positives by applying initial model on training data"""
        print(' --- Obtaining FP --- ')
        self.model.eval()
        if if_aug:
            if os.path.exists(self.dataset_path + 'fold_%d/fp_%s_current_aug.json' % (self.fold, self.model_name)):
                with open(self.dataset_path + 'fold_%d/fp_%s_current_aug.json' % (self.fold, self.model_name), mode='r') as f:
                    fp = json.load(f)
                with open(self.dataset_path + 'fold_%d/fp_%s_epoch-%d_aug.json' % (self.fold, self.model_name, self.epoch), mode='w') as f:
                    json.dump(fp, f)
        else:
            if os.path.exists(self.dataset_path + 'fold_%d/fp_%s_current.json' % (self.fold, self.model_name)):
                with open(self.dataset_path + 'fold_%d/fp_%s_current.json' % (self.fold, self.model_name), mode='r') as f:
                    fp = json.load(f)
                with open(self.dataset_path + 'fold_%d/fp_%s_epoch-%d.json' % (self.fold, self.model_name, self.epoch), mode='w') as f:
                    json.dump(fp, f)

        aug_list = []
        if if_aug:
            for pat in self.train_list_sub:
                for i in range(self.aug_num):
                    aug_list.append(pat + '_aug%d' % i)
        fp_list = self.train_cases_fp if len(self.train_cases_fp) else []
        loader = tqdm(self.train_list_sub + aug_list)
        for pat in loader:
            data_list = []
            for mod in self.modality:
                data_list.append(np.load(self.data_path + '%s/%s_space-T2S_%s.npy' % (pat, pat, mod)))
            cmb = np.load(self.data_path + '%s/%s_space-T2S_CMB.npy' % (pat, pat), mmap_mode='r')
            shape = cmb.shape
            img = np.stack(data_list, axis=0)
            pred, pred_post, n_obj, pred_init_space, candidates_list, score_init_space = \
                self.inference(img, patch_size=(160, 160, 80), thresh=thresh, size=4)
            for (x, y, z) in candidates_list:
                if x > shape[0] - self.bbox[0] // 2 or x < self.bbox[0] // 2 or \
                        y > shape[1] - self.bbox[1] // 2 or y < self.bbox[1] // 2 or \
                        z > shape[2] - self.bbox[2] // 2 or z < self.bbox[2] // 2:
                    continue
                if np.sum(cmb[x - 1:x + 1, y - 1:y + 1, z - 1:z + 1]):
                    sample = {'pat': pat, 'start': (x, y, z), 'have cmb': 1}
                else:
                    sample = {'pat': pat, 'start': (x, y, z), 'have cmb': 0}
                    fp_list.append(sample)
            loader.set_description('FP num: %d' % len(fp_list))

        self.train_cases_fp = fp_list
        with open(self.dataset_path + 'fold_%d/fp_%s_current.json' % (self.fold, self.model_name), mode='w') as f:
            json.dump(fp_list, f)
        print(' --- Finish, FP num: %d ---' % len(fp_list))

        return fp_list

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
            torch.save(state, self.model_save_path + 'epoch_%d_%d_%d_%.4f_%.4f.pth.tar' %
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
            raise ValueError('Too many matched models!')
        if not silent:
            print('Screen model: %s, device: %s, epoch: %d'
                  % (self.model_save_path + model_name, self.device, self.epoch))

    def inference(self, data: np.ndarray, patch_size=None, thresh=0.5, size=2, if_nms=True):
        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=0)
        shape = [data.shape[1], data.shape[2], data.shape[3]]

        # compute the output size and the patches location, exactly corresponding to the architecture of ScreenNet
        if patch_size is None:
            patch_size = shape
        out_size = [(shape[0] - 8) // 2 - 5, (shape[1] - 8) // 2 - 5, (shape[2] - 4) // 2 - 5]
        out_patch_size = [(patch_size[0] - 8) // 2 - 5, (patch_size[1] - 8) // 2 - 5, (patch_size[2] - 4) // 2 - 5]
        # print(data.shape, out_size, out_patch_size)
        num_xyz = [out_size[i] // out_patch_size[i] for i in range(3)]
        overlap_xyz = [((num_xyz[i] + 1) * out_patch_size[i] - out_size[i]) // num_xyz[i] for i in range(3)]
        x_starts = [(out_patch_size[0] - overlap_xyz[0]) * n for n in range(num_xyz[0])]
        x_starts.append(out_size[0] - out_patch_size[0])
        y_starts = [(out_patch_size[1] - overlap_xyz[1]) * n for n in range(num_xyz[1])]
        y_starts.append(out_size[1] - out_patch_size[1])
        z_starts = [(out_patch_size[2] - overlap_xyz[2]) * n for n in range(num_xyz[2])]
        z_starts.append(out_size[2] - out_patch_size[2])
        out_starts = [(x, y, z) for z in z_starts for y in y_starts for x in x_starts]
        starts = [(2*x, 2*y, 2*z) for (x, y, z) in out_starts]

        # inference by sliding window strategy
        pred = np.zeros(out_size)
        overlap = np.zeros(out_size)
        data = torch.tensor(data).float()
        for st, out_st in zip(starts, out_starts):
            data_patch = data[:, st[0]:st[0] + patch_size[0], st[1]:st[1] + patch_size[1], st[2]:st[2] + patch_size[2]]
            data_patch = data_patch.to(self.device).unsqueeze(0)
            pred_patch = self.model(data_patch).detach()
            pred_patch = nnf.softmax(pred_patch, dim=1).squeeze()[1].detach().cpu().numpy()
            pred[out_st[0]:out_st[0] + out_patch_size[0],
                 out_st[1]:out_st[1] + out_patch_size[1],
                 out_st[2]:out_st[2] + out_patch_size[2]] += pred_patch
            overlap[out_st[0]:out_st[0] + out_patch_size[0],
                    out_st[1]:out_st[1] + out_patch_size[1],
                    out_st[2]:out_st[2] + out_patch_size[2]] += 1

        pred /= overlap
        pred_th = pred.copy()
        pred_th[pred_th < thresh] = 0
        if if_nms:
            pred_itk = sitk.GetImageFromArray(pred_th)
            pred_itk = sitk.RegionalMaxima(pred_itk)
            pred_post = sitk.GetArrayFromImage(pred_itk)
            labeled, n_obj = find_binary_object(pred_post)
            maxima_list = ndimage.center_of_mass(labeled, labeled, range(1, n_obj+1))
        else:
            pred_post = pred_th.copy()
            pred_post[pred_post >= thresh] = 1
            labeled, n_obj = find_binary_object(pred_post)
            maxima_list = ndimage.center_of_mass(labeled, labeled, range(1, n_obj + 1))

        # find candidates
        score_init_space = np.zeros(shape)
        score_init_space[9:pred.shape[0] * 2 + 9, 9:pred.shape[1] * 2 + 9, 7:pred.shape[2] * 2 + 7] = \
            nnf.interpolate(torch.tensor(pred, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                            scale_factor=2, mode='trilinear', align_corners=False).squeeze().numpy()

        # map the results back to input volume space
        pred_init_space = np.zeros(shape)
        candidates_list = []
        for (x, y, z) in maxima_list:
            x = int(2 * x + 9)
            y = int(2 * y + 9)
            z = int(2 * z + 7)
            if x < 0 or x >= shape[0] \
                    or y < 0 or y >= shape[1] \
                    or z < 0 or z >= shape[2]:
                continue
            pred_init_space[max(x-size//2, 0):min(x+size//2, shape[0]),
                            max(y-size//2, 0):min(y+size//2, shape[1]),
                            max(z-size//2, 0):min(z+size//2, shape[1])] = 1
            candidates_list.append((x, y, z))
        return pred, pred_post, n_obj, pred_init_space, candidates_list, score_init_space



