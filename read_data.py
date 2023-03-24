import os
import cv2
import h5py
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

# (192, 112)

bh, bw = (192, 112)
h, w = (240, 320)
data_h5 = os.getcwd() + "/data/" 

def setup_labels(data_list, is_valid):
    cdata = []

    for i, label in enumerate(data_list):
        if is_valid[i] == 0: continue
        ca = label[1]  # neck
        cb = label[8]  # torso
        cjnt = (ca + cb) * 0.5

        pose = label - cjnt  # centered
        pose = 2 * (pose - np.min(pose)) / (np.max(pose) - np.min(pose)) - 1  # scale

        cdata.append(pose)

    return np.asarray(cdata)

def setup_depth_map(data_list, is_valid):
    cdata = []

    cw = 56
    ch = 48
    for i, img in enumerate(data_list):
        if is_valid[i] == 0: continue
        cimg = np.asarray(img[ch:, int((w/2)-cw):int((w/2)+cw)])
        norm = (cimg - np.min(cimg)) / (np.max(cimg) - np.min(cimg))
        cdata.append(norm)
    
    return np.asarray(cdata)

def prepare_model_data(type):
    assert type == 'train' or type == 'test'

    f = h5py.File(data_h5 + 'ITOP_side_' + type + '_labels.h5', 'r')
    is_valid = np.asarray(f.get('is_valid'))
    label = np.asarray(f.get('real_world_coordinates'))
    label = setup_labels(label, is_valid)

    f = h5py.File(data_h5 + 'ITOP_side_' + type + '_depth_map.h5', 'r')
    depth_map = np.asarray(f.get('data'))
    depth_map = setup_depth_map(depth_map, is_valid)

    return label, depth_map

def save_modified_data():
    train_label, train_data = prepare_model_data('train')
    test_label, test_data = prepare_model_data('test')

    with h5py.File(data_h5 + 'norm_dataset_itop.hdf5', 'w') as hf:
        hf.create_dataset('x_train', data=train_data, shape=train_data.shape, compression='gzip', chunks=True)
        hf.create_dataset('y_train', data=train_label, shape=train_label.shape, compression='gzip', chunks=True)
        hf.create_dataset('x_test', data=test_data, shape=test_data.shape, compression='gzip', chunks=True)
        hf.create_dataset('y_test', data=test_label, shape=test_label.shape, compression='gzip', chunks=True)
    

def visualise_data(data):
    cw = 56
    ch = 48

    data_reshape = data[ch:, int((w/2)-cw):int((w/2)+cw)]

    print(data_reshape.shape)

    plt.matshow(data_reshape, cmap=plt.cm.viridis, interpolation='bicubic')
    plt.colorbar()
    
    plt.grid(visible=None)
    plt.axis('off')
    plt.show()

def visualise_label(data, norm=False):
    joint_id_to_name = {
        0: 'Head',        8: 'Torso',
        1: 'Neck',        9: 'R Hip',
        2: 'R Shoulder',  10: 'L Hip',
        3: 'L Shoulder',  11: 'R Knee',
        4: 'R Elbow',     12: 'L Knee',
        5: 'L Elbow',     13: 'R Foot',
        6: 'R Hand',      14: 'L Foot',
        7: 'L Hand',
    }

    joint_map = [[0,1],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7],[1,8],[8,9],[8,10],[9,10],[9,11],[10,12],[11,13],[12,14]]

    if norm:
        dx, dy, dz = zip(*data)

        ca = data[1]  # neck
        cb = data[8]  # torso
        cjnt = (ca + cb) * 0.5
        pose = data - cjnt  # centered

        pose = 2 * (pose - np.min(pose)) / (np.max(pose) - np.min(pose)) - 1  # scale

        dx, dy, dz = zip(*pose)
        
    else:
        dx, dy, dz = zip(*data)
        

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for j in range(len(joint_id_to_name)):
        ax.text(dx[j]+.025, dy[j]+.025, dz[j], '%s' % str(joint_id_to_name.get(j)), size=7)

    for parent, child in joint_map:
        ax.plot([dx[parent], dx[child]], 
                [dy[parent], dy[child]], 
                [dz[parent], dz[child]], 'k-')

    p = ax.scatter3D(dx, dy, dz, c=dz)
    fig.colorbar(p)

    ax.view_init(elev=90, azim=-90)
    plt.grid(visible=None)
    plt.axis('off')
    plt.show()

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

    # 500 test is bad
    idx = random.randint(0, 10000)
    print(idx)

    f = h5py.File(data_h5 + 'ITOP_side_test_labels.h5', 'r')
    pos, valid = f.get('real_world_coordinates'), f.get('is_valid')
    
    unique, counts = np.unique(valid, return_counts=True)
    print(dict(zip(unique, counts)))

    if _disp: 
        visualise_label(pos[idx], False)
        visualise_label(pos[idx], True)

    f = h5py.File(data_h5 + 'ITOP_side_test_depth_map.h5', 'r')
    data = f.get('data')

    if _disp: visualise_data(data[idx])

    if _export: save_modified_data()



# execute main function
if __name__ == "__main__":
    main()