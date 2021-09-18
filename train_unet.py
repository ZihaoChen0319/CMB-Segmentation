import torch
import os
import argparse

from SegTrainer import UNetTrainer

torch.backends.cudnn.benchmark = False
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':
    # cmd parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fold', type=int, default=0, help='choose the fold(0,1,2,3,4) you want to train')
    parser.add_argument('-sn', '--screen_name', type=str, default='screening', help='name of screen model')
    parser.add_argument('-dn', '--discri_name', type=str, default='discrimination', help='name of discrimination model')
    parser.add_argument('-un', '--unet_name', type=str, default='unet', help='name of unet model')
    args = parser.parse_args()
    print('Training fold: %d' % args.fold)

    # path define
    data_path = '../data/Task2_processed/Preprocessed_v2/'
    model_save_path = './models/%s/' % args.unet_name
    screen_model_path = './models/%s/' % args.screen_name
    discri_model_path = './models/%s/' % args.discri_name
    dataset_path = './dataset/'

    # some config
    config = {
        'fold': args.fold,
        'bbox': [32, 32, 24],
        'batch_size': 64,
        'all_epochs': 100,
        'loss': 'soft dice',
        'init_lr': 1e-2,
        'optimizer': 'sgd',
        'weight_decay': 'poly',
        'decay_exponent': 0.9,
        'random_negatives': 200000,
        'aug_num': 0,
        'resample_num': [8000, 0, 8000],
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
        model_save_path=model_save_path,
        dataset_path=dataset_path,
        screen_model_path=screen_model_path,
        load_screen='current',
        discri_model_path=discri_model_path,
        load_discri='current',
        device=device,
        all_epoch=config['all_epochs'],
        fold=config['fold'],
        bbox=config['bbox'],
        batch_size=config['batch_size'],
        loss=config['loss'],
        optimizer=config['optimizer'],
        init_lr=config['init_lr'],
        decay_exponent=config['decay_exponent'],
        random_negatives=config['random_negatives'],
        aug_num=config['aug_num'],
        add_fp=True,
        resample_num=config['resample_num'],
        modality=config['modality'],
        config=config
    )
    trainer.load_model('current')

    # training and validation
    while trainer.epoch <= config['all_epochs']:
        trainer.adjust_lr()
        train_loss, train_dice = trainer.train_epoch()
        if trainer.epoch % 5 == 0:
            trainer.val_epoch()
        trainer.save_model()
        trainer.epoch += 1





