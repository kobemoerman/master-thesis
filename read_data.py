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
    # data = np.array(data).astype(np.float32).reshape((-1, 45))
    # data_mean = np.expand_dims(np.mean(data, axis=-1), axis=-1)
    # data_std  = np.expand_dims(np.std(data, axis=-1), axis=-1)

    # return (data - data_mean) / data_std
    label = []
    for d in data:
        norm = (d / np.linalg.norm(d)).flatten().astype(np.float32)
        label.append(norm)
    return np.asarray(label, dtype=np.float32)

def normalize_data(id, data):
    # cw = 80
    # ch = 10
    cw = 56
    ch = 48

    cdict = {}
    for key, img in zip(id, data):
        cimg = np.asarray(img[ch:, int((w/2)-cw):int((w/2)+cw)])
        #cimg = cv2.resize(np.float32(cimg), (80, 128))
        
        norm = (cimg - np.min(cimg)) / (np.max(cimg) - np.min(cimg))

        # row, col = np.where(cimg >= 3.4)
        # norm[row,col] = 1.0
        
        # noise_fact = 0.3
        # row, col = np.where(cimg < 3.4)
        # noise_img = norm + noise_fact * np.random.normal(loc=0.0, scale=1.0, size=norm.shape)
        # norm[row,col] = noise_img[row,col]
        # norm = np.clip(norm, 0., 1.)

        cdict[key] = norm

    return cdict


def prepare_model_data(type):
    assert type == 'train' or type == 'test'

    l = h5py.File('./data/ITOP_side_' + type + '_labels.h5', 'r')
    d = h5py.File('./data/ITOP_side_' + type + '_depth_map.h5', 'r')

    l_id, pos, valid = np.asarray(l.get('id')), np.asarray(l.get('real_world_coordinates')), np.asarray(l.get('is_valid'))
    d_id, data = np.asarray(d.get('id')), np.asarray(d.get('data'))

    is_valid = np.where(valid == 1)

    l_id, pos = l_id[is_valid], pos[is_valid]
    d_id, data = d_id[is_valid], data[is_valid]

    # norm_pos = normalize_labels(pos)
    norm_pos = pos
    labels = dict(zip(l_id, norm_pos))
    print(norm_pos.shape)

    images = normalize_data(d_id, data)

    x, y = [], []
    for key in images:
        x.append(images[key])
        y.append(labels[key])

    return np.asarray(y, dtype=np.float32), np.asarray(x, dtype=np.float32)

def save_modified_data():
    train_label, train_data = prepare_model_data('train')
    test_label, test_data = prepare_model_data('test')

    with h5py.File('./data/v_dataset_itop.hdf5', 'w') as hf:
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
    img_pos = np.asarray(f.get('image_coordinates'), dtype=np.int32)

    unique, counts = np.unique(valid, return_counts=True)
    print(dict(zip(unique, counts)))

    is_valid = np.where(valid == 1)
    norm_pos = normalize_labels(pos[is_valid])
    labels = dict(zip(id[is_valid], norm_pos))

    idx = random.randint(0, len(labels))
    if _disp: visualise_label(norm_pos[idx])

    f = h5py.File('./data/ITOP_side_test_depth_map.h5', 'r')
    id, data = np.asarray(f.get('id')), np.asarray(f.get('data'))

    data = data[is_valid]
    # dimg = normalize_data(id[is_valid], data)

    if _disp: visualise_data(data[idx])
    if _export: save_modified_data()

    # data_segmented_noise(data)
    # data_position_noise(data, img_pos[is_valid])



# execute main function
if __name__ == "__main__":
    main()