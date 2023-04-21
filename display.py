import h5py
import random
import cv2
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from utils import visualise_label

idx = 5455

def image_label(data, image):
    ax1 = plt.axes()
    ax1.imshow(data)

    colors = ['c', 'r', 'b', 'orange', 'b', 'orange', 'b', 'orange', 'm', 'purple', 'm', 'purple', 'm', 'purple', 'm']

    for i in range(0, image.shape[0]):
        ax1.scatter(x=image[i][0], y=image[i][1], c=colors[i], s=20)

    dx, dy= zip(*image)
    joint_map = [[0,1],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7],[1,8],[8,9],[8,10],[9,11],[10,12],[11,13],[12,14]]
    colors = ['c', 'r', 'r', 'b', 'orange', 'b', 'orange', 'r', 'purple', 'm', 'purple', 'm', 'purple', 'm', 'purple']
    for i, joint in enumerate(joint_map):
        parent, child = joint
        ax1.plot([dx[parent], dx[child]], 
                [dy[parent], dy[child]], '-', color=colors[i])
    plt.grid(visible=None)
    plt.axis('off')
    plt.show()

def load_pcd():
    f = h5py.File('./data/ITOP_side_test_point_cloud.h5', 'r')
    data = np.asarray(f.get('data'), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, np.pi))
    pcd = pcd.rotate(R, center=(0,0,0))

    o3d.io.write_point_cloud("./data/pcd.ply", pcd)
    o3d.visualization.draw_geometries([pcd])

def save_pcd():
    pcd = o3d.io.read_point_cloud('./data/pcd.ply')

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image('./data/pcd.png')
    vis.destroy_window()

def display_preprocessing():
    f = h5py.File('./data/ITOP_side_test_labels.h5', 'r')
    joints = np.asarray(f.get('real_world_coordinates'), dtype=np.float32)
    image = np.asarray(f.get('image_coordinates'), dtype=np.float32)

    f = h5py.File('./data/ITOP_side_test_depth_map.h5', 'r')
    data = np.asarray(f.get('data'))
    data = np.asarray(data[idx], dtype=np.float32)
    joints = np.asarray(joints[idx], dtype=np.float32)
    image = np.asarray(image[idx], dtype=np.float32)

    image_label(data, image)
        
    h, w = (240, 320)
    cw = 56
    ch = 48
    data = np.asarray(data[ch:, int((w/2)-cw):int((w/2)+cw)], dtype=np.float32)
    ax2 = plt.axes()
    ax2.imshow(data)
    plt.grid(visible=None)
    plt.axis('off')
    plt.show()

    data = cv2.resize(np.float32(data), (64, 96))
    ax3 = plt.axes()
    ax3.imshow(data)
    plt.grid(visible=None)
    plt.axis('off')
    plt.show()

def data_validity():
    f = h5py.File('./data/ITOP_side_test_labels.h5', 'r')
    joints = np.asarray(f.get('real_world_coordinates'), dtype=np.float32)
    image = np.asarray(f.get('image_coordinates'), dtype=np.float32)
    valid = np.asarray(f.get('is_valid'), dtype=np.float32)
    seg = np.asarray(f.get('segmentation'), dtype=np.float32)

    f = h5py.File('./data/ITOP_side_test_depth_map.h5', 'r')
    data = np.asarray(f.get('data'))

    is_valid = np.where(valid == 0)

    data = np.asarray(data[is_valid], dtype=np.float32)
    joints = np.asarray(joints[is_valid], dtype=np.float32)
    image = np.asarray(image[is_valid], dtype=np.float32)

    idx = random.randint(0, data.shape[0])

    image_label(data[idx], image[idx])

    is_valid = np.where(valid == 1)
    idx = random.randint(0, data.shape[0])

    seg = np.asarray(seg[is_valid], dtype=np.float32)

    ax = plt.axes()
    ax.imshow(seg[idx])
    plt.grid(visible=None)
    plt.axis('off')
    plt.show()

def normalized_coords():
    f = h5py.File('./data/ITOP_side_test_labels.h5', 'r')
    joints = np.asarray(f.get('real_world_coordinates'), dtype=np.float32)
    valid = np.asarray(f.get('is_valid'), dtype=np.float32)

    is_valid = np.where(valid == 1)
    joints = joints[is_valid]
    joints = joints[random.randint(0, len(joints))]

    norm = np.linalg.norm(joints)

    visualise_label(joints)
    print(joints)

    data = joints.flatten().astype(np.float32)
    data_mean = np.expand_dims(np.mean(data, axis=-1), axis=-1)
    data_std  = np.expand_dims(np.std(data, axis=-1), axis=-1)

    joints = (data - data_mean) / data_std
    
    # joints = joints / norm
    visualise_label(joints)
    print(joints)

    joints = (joints * data_std) + data_mean

    # joints = joints * norm
    visualise_label(joints)
    print(joints.reshape((15, 3)))

def plot_joints():
    gt = np.array([[ 0.01599121,  0.505661  , -0.0078125 ],
       [-0.00598145,  0.23808289, -0.0078125 ],
       [-0.16766357,  0.23442078,  0.00195312],
       [ 0.15563965,  0.241745  , -0.015625  ],
       [-0.23944092,  0.00189209, -0.12304688],
       [ 0.32806396,  0.01628113, -0.11914062],
       [-0.14764404,  0.05162048, -0.3359375 ],
       [ 0.23614502,  0.04740906, -0.36914062],
       [ 0.        ,  0.        ,  0.        ],
       [-0.10076904, -0.24055481,  0.01367188],
       [ 0.11268616, -0.23554993,  0.00195312],
       [-0.1395874 , -0.77119446,  0.0703125 ],
       [ 0.13876343, -0.7575226 ,  0.01953125],
       [-0.15301514, -1.275589  ,  0.17382812],
       [ 0.16497803, -1.2521515 ,  0.13476562]], dtype=np.float32)
    
    pred = np.array([[ 0.00664376,  0.48448923, -0.05896349],
       [ 0.00498535,  0.24518248, -0.0240878 ],
       [-0.16845858,  0.24859257, -0.06168836],
       [ 0.17342599,  0.24555534,  0.00942011],
       [-0.20037709,  0.05174495, -0.23873678],
       [ 0.28130907,  0.07980462, -0.14905208],
       [-0.08865879,  0.03022723, -0.44614077],
       [ 0.24449362,  0.1527352 , -0.38215512],
       [ 0.00000376,  0.00004768,  0.0000248 ],
       [-0.10899299, -0.24538228,  0.00460257],
       [ 0.10203844, -0.2465317 ,  0.04539451],
       [-0.11590815, -0.7444695 ,  0.04194095],
       [ 0.12265465, -0.7372111 ,  0.05815578],
       [-0.14744669, -1.2013534 ,  0.17352068],
       [ 0.1337322 , -1.2034496 ,  0.16930684]], dtype=np.float32)
    

    
    print(np.linalg.norm(gt[6]-gt[7]))
    print(np.linalg.norm(pred[6]-pred[7]))
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    joint_map = [[0,1],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7],[1,8],[8,9],[8,10],[9,10],[9,11],[10,12],[11,13],[12,14]]

    dx, dy, dz = zip(*gt)
    for parent, child in joint_map:
        ax.plot([dx[parent], dx[child]], 
                [dy[parent], dy[child]], 
                [dz[parent], dz[child]], 'r-')

    ax.scatter3D(dx, dy, dz, c=dz)

    dx, dy, dz = zip(*pred)
    for parent, child in joint_map:
        ax.plot([dx[parent], dx[child]], 
                [dy[parent], dy[child]], 
                [dz[parent], dz[child]], 'b-')

    ax.scatter3D(dx, dy, dz, c=dz)

    ax.view_init(elev=90, azim=-90)
    # plt.grid(visible=None)
    # plt.axis('off')
    plt.show()

plot_joints()