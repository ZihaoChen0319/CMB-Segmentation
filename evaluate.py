import os
import time
import numpy as np
import medpy.io as mio
import torch
import argparse
import medpy.metric.binary as mmb

from SegTrainer import UNetTrainer
from PairwiseMeasures_modified import PairwiseMeasures

torch.backends.cudnn.benchmark = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    # cmd parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fold', type=int, default=0, help='choose the fold(0,1,2,3,4) you want to train')
    parser.add_argument('-dt', '--discri_threshold', type=float, default=0.5, help='discrimination model threshold')
    parser.add_argument('-st', '--screen_threshold', type=float, default=0.1, help='screen model threshold')
    parser.add_argument('-dm', '--discri_model', type=str, default=None, help='name of discrimination model')
    parser.add_argument('-sm', '--screen_model', type=str, default=None, help='name of screen model')
    parser.add_argument('-u', '--unet', type=str, default=None, help='name of unet model')
    args = parser.parse_args()
    args.discri_model = 'discrimination'
    args.screen_model = 'screening'
    args.unet = 'unet'

    # path define
    data_path = './data/Task2_processed/Preprocessed_v2/'
    dataset_path = './dataset/'
    screen_model_path = './models/%s/' % args.screen_model
    discri_model_path = './models/%s/' % args.discri_model
    unet_model_path = './models/%s/' % args.unet
    pred_save_path = './results/'
    if not os.path.exists(pred_save_path):
        os.makedirs(pred_save_path)

    # some config
    config = {
        'fold': args.fold,
        'bbox': [32, 32, 24],
        'random_negatives': 200000,
        'aug_num': 10,
        'modality': ['T2S'],
    }

    # define device
    gpu = 0
    device = 'cuda:%d' % gpu if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    # device = 'cpu'

    # trainer
    trainer = UNetTrainer(
        data_path=data_path,
        model_save_path=unet_model_path,
        dataset_path=dataset_path,
        screen_model_path=screen_model_path,
        discri_model_path=discri_model_path,
        load_screen='current',
        load_discri='current',
        device=device,
        fold=config['fold'],
        modality=config['modality'],
        if_test=True
    )
    trainer.load_model('current')
    trainer.screen_trainer.model.eval()
    trainer.discri_trainer.model.eval()
    trainer.model.eval()

    test_accum = [0] * 18
    for pat in trainer.val_list_sub:
        start = time.time()
        data_list = []
        for mod in trainer.modality:
            data_list.append(np.load(trainer.data_path + '%s/%s_space-T2S_%s.npy' % (pat, pat, mod)))
        cmb, h = mio.load(data_path + '%s/%s_space-T2S_CMB.nii.gz' % (pat, pat))
        img = np.stack(data_list, axis=0)
        pred, pred_post, n_obj, pred_init_space, candidates_list, score_init_space = \
            trainer.screen_trainer.inference(img, patch_size=(160, 160, 80), thresh=args.screen_threshold,
                                             size=2, if_nms=True)
        pred_fp_reduced, reduc_candidates_list, num = \
            trainer.discri_trainer.inference(img, candidates_list, size=2, thresh=args.discri_threshold)
        seg_pred = trainer.inference(img, reduc_candidates_list, if_translate=False)
        pe_scr = PairwiseMeasures(ref_img=cmb, seg_img=pred_init_space, analysis='microbleeds',
                                  measures=('f1_score', 'tp', 'fn', 'fp'),
                                  connectivity=3, pixdim=h.get_voxel_spacing(), empty=True,
                                  threshold=0.5, thresh_assign=3)
        pe_dis = PairwiseMeasures(ref_img=cmb, seg_img=pred_fp_reduced, analysis='microbleeds',
                                  measures=('f1_score', 'tp', 'fn', 'fp'),
                                  connectivity=3, pixdim=h.get_voxel_spacing(), empty=True,
                                  threshold=0.5, thresh_assign=3)
        pe_seg = PairwiseMeasures(ref_img=cmb, seg_img=seg_pred, analysis='microbleeds',
                                  measures=('f1_score', 'tp', 'fn', 'fp', 'mean_diceover',
                                            'absolute_count_difference', 'absolute_volume_difference'),
                                  connectivity=3, pixdim=h.get_voxel_spacing(), empty=True,
                                  threshold=0.5, thresh_assign=1)
        tp_scr, fn_scr, fp_scr, f1_scr = pe_scr.m_dict['tp'][0](), pe_scr.m_dict['fn'][0](), \
                                         pe_scr.m_dict['fp'][0](), pe_scr.m_dict['f1_score'][0]()
        tp_dis, fn_dis, fp_dis, f1_dis = pe_dis.m_dict['tp'][0](), pe_dis.m_dict['fn'][0](), \
                                         pe_dis.m_dict['fp'][0](), pe_dis.m_dict['f1_score'][0]()
        tp_seg, fn_seg, fp_seg, f1_seg = pe_seg.m_dict['tp'][0](), pe_seg.m_dict['fn'][0](), \
                                         pe_seg.m_dict['fp'][0](), pe_seg.m_dict['f1_score'][0]()
        dice = pe_seg.m_dict['mean_diceover'][0]()
        vol_diff = pe_seg.m_dict['absolute_volume_difference'][0]()
        count_diff = pe_seg.m_dict['absolute_count_difference'][0]()
        test_accum[0] += 1  # number of cases
        test_accum[1] += 1 if np.sum(cmb) else 0  # number of cases with CMB
        test_accum[2] += 1 if np.sum(cmb) and dice else 0  # number of cases with CMB and F1 is not 0
        test_accum[3] += tp_scr
        test_accum[4] += fn_scr
        test_accum[5] += fp_scr
        test_accum[6] += f1_scr if np.sum(cmb) else 0
        test_accum[7] += tp_dis
        test_accum[8] += fn_dis
        test_accum[9] += fp_dis
        test_accum[10] += f1_dis if np.sum(cmb) else 0
        test_accum[11] += tp_seg
        test_accum[12] += fn_seg
        test_accum[13] += fp_seg
        test_accum[14] += f1_seg if np.sum(cmb) else 0
        test_accum[15] += dice if np.sum(cmb) else 0
        test_accum[16] += vol_diff
        test_accum[17] += count_diff
        stop = time.time()
        print('%s: scr TP %d, scr FN %d, scr FP %d, scr F1 %.2f, dis TP %d, dis FN %d, dis FP %d, dis F1 %.2f, '
              '\n\t seg TP %d, seg FN %d, seg FP %d, count diff %.2f, F1 %.2f, Dice %.2f, volume diff %.2f, time %.1f'
              % (pat, tp_scr, fn_scr, fp_scr, f1_scr, tp_dis, fn_dis, fp_dis, f1_dis, tp_seg, fn_seg, fp_seg,
                 count_diff, f1_seg, dice, vol_diff, stop-start))

        new_img = img[0]
        new_img[new_img == 0] = np.min(new_img)
        mio.save(new_img, pred_save_path + '%s_volume.nii.gz' % pat, hdr=h)
        # mio.save(pred, pred_save_path + '%s_pred.nii.gz' % pat, hdr=h)
        # mio.save(pred_post, pred_save_path + '%s_pred_post.nii.gz' % pat, hdr=h)
        # mio.save(pred_init_space, pred_save_path + '%s_pred_init_space.nii.gz' % pat, hdr=h)
        # mio.save(score_init_space, pred_save_path + '%s_score_init_space.nii.gz' % pat, hdr=h)
        # mio.save(pred_fp_reduced, pred_save_path + '%s_pred_fp_reduced.nii.gz' % pat, hdr=h)
        mio.save(seg_pred, pred_save_path + '%s_seg.nii.gz' % pat, hdr=h)

    print('Overall: scr TP %d, scr FN %d, scr avg FP %.2f, scr F1 %.2f, dis TP %d, dis FN %d, dis avg FP %.2f, dis F1 %.2f, '
          '\n\t seg TP %d, seg FN %d, seg avg FP %.2f, count diff %.2f, F1 %.2f, Dice %.2f, volume diff %.2f'
          % (test_accum[3], test_accum[4], test_accum[5]/test_accum[0], test_accum[6]/test_accum[1],
             test_accum[7], test_accum[8], test_accum[9]/test_accum[0], test_accum[10]/test_accum[1],
             test_accum[11], test_accum[12], test_accum[13]/test_accum[0], test_accum[17]/test_accum[0],
             test_accum[14]/test_accum[1], test_accum[15]/test_accum[1], test_accum[16]/test_accum[0]))
