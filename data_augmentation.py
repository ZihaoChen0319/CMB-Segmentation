import torchio as tio
import numpy as np
import os


if __name__ == '__main__':
    data_path = './data/Task2_processed/Preprocessed_v2/'
    files = os.listdir(data_path)
    files = [x for x in files if len(x.split('_')) == 1]
    aug_num = 10

    for file in files:
        # load image
        img_tio = tio.Subject(
            t1=tio.ScalarImage(data_path + '%s/%s_space-T2S_T1.nii.gz' % (file, file)),
            t2=tio.ScalarImage(data_path + '%s/%s_space-T2S_T2.nii.gz' % (file, file)),
            t2s=tio.ScalarImage(data_path + '%s/%s_space-T2S_T2S.nii.gz' % (file, file)),
            cmb=tio.LabelMap(data_path + '%s/%s_space-T2S_CMB.nii.gz' % (file, file))
        )

        # determine the transformation
        flip = tio.RandomFlip(axes=('lr',), flip_probability=0.5)
        rotation = tio.RandomAffine(scales=0, degrees=10, translation=0, center='image',
                                    image_interpolation='bspline', default_pad_value=0)
        transform = tio.Compose([flip, rotation])

        # random transform the image
        for i in range(aug_num):
            file_name = '%s_aug%d' % (file, i)
            if not os.path.exists(data_path + file_name + '/'):
                os.makedirs(data_path + file_name + '/')
            print(file_name)
            img_aug = transform(img_tio)
            # img_aug['t1'].save(data_path + '%s/%s_space-T2S_T1.nii.gz' % (file_name, file_name))
            # img_aug['t2'].save(data_path + '%s/%s_space-T2S_T2.nii.gz' % (file_name, file_name))
            # img_aug['t2s'].save(data_path + '%s/%s_space-T2S_T2S.nii.gz' % (file_name, file_name))
            # img_aug['cmb'].save(data_path + '%s/%s_space-T2S_CMB.nii.gz' % (file_name, file_name))
            t1 = img_aug['t1'].data.squeeze().numpy().astype(np.float32)
            t2 = img_aug['t2'].data.squeeze().numpy().astype(np.float32)
            t2s = img_aug['t2s'].data.squeeze().numpy().astype(np.float32)
            cmb = img_aug['cmb'].data.squeeze().numpy().astype(np.float32)
            t1[np.abs(t1) < 1e-4] = 0
            t2[np.abs(t2) < 1e-4] = 0
            t2s[np.abs(t2s) < 1e-4] = 0
            np.save(data_path + '%s/%s_space-T2S_T1.npy' % (file_name, file_name), t1)
            np.save(data_path + '%s/%s_space-T2S_T2.npy' % (file_name, file_name), t2)
            np.save(data_path + '%s/%s_space-T2S_T2S.npy' % (file_name, file_name), t2s)
            np.save(data_path + '%s/%s_space-T2S_CMB.npy' % (file_name, file_name), cmb)






