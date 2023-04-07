import cv2
import h5py
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import data_position_noise, data_segmented_noise, visualise_data, visualise_label

bh, bw = (192, 112)
h, w = (240, 320)

def normalize_labels(data):
    data = np.array(data).astype(np.float32).reshape((-1, 45))
    data_mean = np.expand_dims(np.mean(data, axis=-1), axis=-1)
    data_std  = np.expand_dims(np.std(data, axis=-1), axis=-1)

    return (data - data_mean) / data_std

def normalize_data(id, data):
    norm = cv2.normalize(np.float32(data), None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return dict(zip(id, norm))

def crop_data(data):
    cw = 56
    ch = 48

    cdata = []
    for img in data:
        cimg = np.asarray(img[ch:, int((w/2)-cw):int((w/2)+cw)])
        cdata.append(cv2.resize(np.float32(cimg), (64, 96)))

    return cdata


def prepare_model_data(type):
    assert type == 'train' or type == 'test'

    l = h5py.File('./data/ITOP_side_' + type + '_labels.h5', 'r')
    d = h5py.File('./data/ITOP_side_' + type + '_depth_map.h5', 'r')

    l_id, pos, valid = np.asarray(l.get('id')), np.asarray(l.get('real_world_coordinates')), np.asarray(l.get('is_valid'))
    d_id, data = np.asarray(d.get('id')), np.asarray(d.get('data'))

    is_valid = np.where(valid == 1)

    l_id, pos = l_id[is_valid], pos[is_valid]
    d_id, data = d_id[is_valid], data[is_valid]

    norm_pos = normalize_labels(pos)
    labels = dict(zip(l_id, norm_pos))

    data = crop_data(data)
    images = normalize_data(d_id, data)

    x, y = [], []
    for key in images:
        x.append(images[key])
        y.append(labels[key])

    return np.asarray(y, dtype=np.float32), np.asarray(x, dtype=np.float32)

def save_modified_data():
    train_label, train_data = prepare_model_data('train')
    test_label, test_data = prepare_model_data('test')

    with h5py.File('./data/resize_dataset_itop.hdf5', 'w') as hf:
        hf.create_dataset('x_train', data=train_data, shape=train_data.shape, compression='gzip', chunks=True)
        hf.create_dataset('y_train', data=train_label, shape=train_label.shape, compression='gzip', chunks=True)
        hf.create_dataset('x_test', data=test_data, shape=test_data.shape, compression='gzip', chunks=True)
        hf.create_dataset('y_test', data=test_label, shape=test_label.shape, compression='gzip', chunks=True)
    

def main():
    """
    python3 read_data.py -disp True

    Inputs:
        -disp (bool): True if you want to display the labels and depth map, else False.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-disp", type=str, required=True)
    parser.add_argument("-export", type=str, required=True)
    args = parser.parse_args()

    if args.disp not in ("True", "False"): raise ValueError("-disp must be \"True\" or \"False\".")
    else: _disp = args.disp == "True"

    if args.export not in ("True", "False"): raise ValueError("-export must be \"True\" or \"False\".")
    else: _export = args.export == "True"

    f = h5py.File('./data/ITOP_side_test_labels.h5', 'r')
    id, pos, valid = np.asarray(f.get('id')), np.asarray(f.get('real_world_coordinates')), np.asarray(f.get('is_valid'))

    unique, counts = np.unique(valid, return_counts=True)
    print(dict(zip(unique, counts)))

    is_valid = np.where(valid == 1)
    norm_pos = normalize_labels(pos[is_valid])
    labels = dict(zip(id[is_valid], norm_pos))

    idx = random.randint(0, len(labels))
    if _disp: visualise_label(norm_pos[idx])

    f = h5py.File('./data/ITOP_side_test_depth_map.h5', 'r')
    id, data = np.asarray(f.get('id')), np.asarray(f.get('data'))

    data = crop_data(data[is_valid])
    dimg = normalize_data(id[is_valid], data)

    if _disp: visualise_data(data[idx])
    if _export: save_modified_data()

    # cntr = f.get('segmentation')
    # img_pos = f.get('image_coordinates')
    # data_position_noise(data[idx], img_pos[idx])
    # data_segmented_noise(data[idx], cntr[idx])



# execute main function
if __name__ == "__main__":
    main()