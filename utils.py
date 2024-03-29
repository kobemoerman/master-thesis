import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

bh, bw = (192, 112)
h, w = (240, 320)

def visualise_data(data):
    cw = 56
    ch = 48
    data = np.asarray(data[ch:, int((w/2)-cw):int((w/2)+cw)], dtype=np.float32)
    data = cv2.resize(data, (56, 96))

    print(data.shape)
    plt.matshow(data, cmap=plt.cm.viridis, interpolation='bicubic')
    plt.colorbar()
    
    plt.grid(visible=None)
    plt.axis('off')
    plt.show()

def visualise_label(data):
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

    data = np.reshape(data, (15, 3))
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

def data_position_noise(data, joints):
    # get 4 random indices
    sample = random.sample(range(data.shape[0]), 4)

    # joint colors
    colors = ['c', 'r', 'b', 'orange', 'b', 'orange', 'b', 'orange', 'm', 'purple', 'm', 'purple', 'm', 'purple', 'm']

    fig = plt.figure(figsize=(8, 12))
    fig.tight_layout()
    for i in range(len(sample)):
        # resize data
        joint = joints[sample[i]]
        img   = data[sample[i]]

        # normalise data
        img  = (img - np.min(img)) / (np.max(img) - np.min(img))

        # plot image
        ax = fig.add_subplot(2, 4, i+1)
        ax.imshow(img)
        # plot joints
        for c, j in enumerate(joint):
            ax.scatter(x=j[0], y=j[1], c=colors[c], s=5)
        ax.axis('off')

        cimg = np.copy(img)

        # set region around joint locations
        for j in joint:
            for dx in range(int(j[0])-10, int(j[0])+10):
                for dy in range(int(j[1])-10, int(j[1])+10):
                    if dx < 0 or dy < 0 or dx >= w or dy >= h: continue
                    img[dy][dx] = -1
        row, col = np.where(img == -1)

        noise_fact = 0.3
        noise_img = cimg + noise_fact * np.random.normal(loc=0.0, scale=1.0, size=cimg.shape)
        cimg[row, col] = noise_img[row,col]
        cimg = np.clip(cimg, 0., 1.)

        # plot noise
        ax = fig.add_subplot(1, 4, i+1)
        ax.imshow(cimg)
        ax.axis('off')

    plt.show()

def data_segmented_noise(data):
    # get 4 random indices
    sample = random.sample(range(data.shape[0]), 4)

    fig = plt.figure(figsize=(8, 12))
    fig.tight_layout()
    for i in range(len(sample)):
        # resize data
        cw = 56
        ch = 48
        img = data[sample[i]]
        img = np.asarray(img[ch:, int((w/2)-cw):int((w/2)+cw)])

        # segment the body
        row, col = np.where(img < 3.4)

        # normalise data
        img  = (img - np.min(img)) / (np.max(img) - np.min(img))
        cimg = np.copy(img)

        # add noise to data
        noise_fact = 0.3
        noise_img  = img + noise_fact * np.random.normal(loc=0.0, scale=1.0, size=img.shape)
        cimg[row,col] = noise_img[row,col]
        cimg = np.clip(cimg, 0., 1.)

        # plot data
        ax = fig.add_subplot(2, 4, i+1)
        ax.imshow(img)
        ax.axis('off')
        ax = fig.add_subplot(1, 4, i+1)
        ax.imshow(cimg)
        ax.axis('off')

    
    plt.savefig('noise')
    plt.show()