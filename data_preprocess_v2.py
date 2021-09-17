import os
import numpy as np
import medpy.io as mio

from nnunet.preprocessing.preprocessing import resample_data_or_seg


def img_normalize(array):
    if not type(array) is np.ndarray:
        array = np.array(array).squeeze()
    bg_value = np.median(array[:10, :10, :5])
    array = array - bg_value  # set the background to 0
    content = array[np.abs(array) > 1e-4]
    content = (content - np.mean(content)) / np.std(content)  # z-score normalization
    array[np.abs(array) > 1e-4] = content
    return array


def get_bbox(array_list):
    array = np.stack(array_list, axis=0)
    cc, xx, yy, zz = np.nonzero(array)
    x_min = np.min(xx)
    x_max = np.max(xx)
    y_min = np.min(yy)
    y_max = np.max(yy)
    z_min = np.min(zz)
    z_max = np.max(zz)
    return x_min, x_max, y_min, y_max, z_min, z_max


def crop_bbox(array, bounding_box, margin):
    if not type(array) is np.ndarray:
        array = np.array(array).squeeze()
    shape = array.shape
    if len(shape) == 3:
        return array[max(0, bounding_box[0] - margin[0]):min(bounding_box[1] + margin[1], shape[0]-1),
                     max(0, bounding_box[2] - margin[2]):min(bounding_box[3] + margin[3], shape[1]-1),
                     max(0, bounding_box[4] - margin[4]):min(bounding_box[5] + margin[5], shape[2]-1)]
    else:
        return array[:, max(0, bounding_box[0] - margin[0]):min(bounding_box[1] + margin[1], shape[1] - 1),
                        max(0, bounding_box[2] - margin[2]):min(bounding_box[3] + margin[3], shape[2] - 1),
                        max(0, bounding_box[4] - margin[4]):min(bounding_box[5] + margin[5], shape[3] - 1)]


if __name__ == '__main__':
    data_path = './data/Task2/'
    save_path = './data/Task2_processed/Preprocessed_v2/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files = os.listdir(data_path)
    files = [file for file in files if file[0] != '.']
    # files = [file for file in files if file[4] == '1']

    target_spacing = np.array((0.48828101, 0.48828101, 0.80000001))

    for file in files:
        if not os.path.exists(save_path + file + '/'):
            os.makedirs(save_path + file + '/')
        print('\n', file)
        # load data
        cmb, h = mio.load(data_path + '%s/%s_space-T2S_CMB.nii.gz' % (file, file))
        t1, h = mio.load(data_path + '%s/%s_space-T2S_desc-masked_T1.nii.gz' % (file, file))
        t2, h = mio.load(data_path + '%s/%s_space-T2S_desc-masked_T2.nii.gz' % (file, file))
        t2s, h = mio.load(data_path + '%s/%s_space-T2S_desc-masked_T2S.nii.gz' % (file, file))
        t1 = img_normalize(t1)
        t2 = img_normalize(t2)
        t2s = img_normalize(t2s)
        img = np.stack([t1, t2, t2s], axis=0)
        origin_spacing = np.array(h.get_voxel_spacing())
        origin_shape = np.array(t1.shape)
        print('original spacing:', origin_spacing, 'original shape:', origin_shape)

        # resample
        do_separate_z = True if np.max(origin_spacing) / np.min(origin_spacing) > 2.99 else False
        new_shape = np.floor(list(origin_shape * origin_spacing / target_spacing)).astype(int)
        img_resampled = resample_data_or_seg(img, new_shape, is_seg=False, do_separate_z=do_separate_z, axis=[2])
        cmb = np.expand_dims(cmb, axis=0)
        cmb_resampled = resample_data_or_seg(cmb, new_shape, is_seg=True, do_separate_z=do_separate_z, axis=[2])
        print('new spacing:', target_spacing, 'new shape:', img_resampled.shape)

        # cropping
        bbox = get_bbox(img_resampled)
        img_new = crop_bbox(img_resampled, bounding_box=bbox, margin=(10, 10, 10, 10, 10, 10))
        img_new = np.array(img_new, dtype=np.float32)
        cmb_new = crop_bbox(cmb_resampled, bounding_box=bbox, margin=(10, 10, 10, 10, 10, 10))
        cmb_new = np.array(cmb_new, dtype=np.float32)
        print('shape after crop:', img_new.shape)

        # save processed data
        np.save(save_path + '%s/%s_space-T2S_T1.npy' % (file, file), img_new[0])
        np.save(save_path + '%s/%s_space-T2S_T2.npy' % (file, file), img_new[1])
        np.save(save_path + '%s/%s_space-T2S_T2S.npy' % (file, file), img_new[2])
        np.save(save_path + '%s/%s_space-T2S_CMB.npy' % (file, file), cmb_new[0])
        h.set_voxel_spacing(target_spacing)
        h.set_offset((0, 0, 0))
        mio.save(img_new[0], save_path + '%s/%s_space-T2S_T1.nii.gz' % (file, file), hdr=h)
        mio.save(img_new[1], save_path + '%s/%s_space-T2S_T2.nii.gz' % (file, file), hdr=h)
        mio.save(img_new[2], save_path + '%s/%s_space-T2S_T2S.nii.gz' % (file, file), hdr=h)
        mio.save(cmb_new[0], save_path + '%s/%s_space-T2S_CMB.nii.gz' % (file, file), hdr=h)






