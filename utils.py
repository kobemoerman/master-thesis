import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    fig = plt.figure()

    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(data)

    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(data)

    for j in joints:
        ax2.scatter(x=j[0], y=j[1], c='r', s=40)

    ax3 = fig.add_subplot(1,3,3)
    cdata = np.copy(data)
    for j in joints:
        for dx in range(j[0]-10, j[0]+10):
            for dy in range(j[1]-10, j[1]+10):
                if dx < 0 or dy < 0 or dx >= w or dy >= h: continue
                data[dy][dx] = -1

    row, col = np.where(data == -1)
    noise_fact = 0.4

    noise_img = cdata + noise_fact * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    cdata[row,col] = noise_img[row,col]

    ax3.imshow(cdata)

    plt.show()

def data_segmented_noise(data, cntr):
    cw = 56
    ch = 48
    data = np.asarray(data[ch:, int((w/2)-cw):int((w/2)+cw)])
    cntr = np.asarray(cntr[ch:, int((w/2)-cw):int((w/2)+cw)])
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,4,1)

    ax1.imshow(data)

    row, col = np.where(data < 3.4)
    
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    cdata = np.copy(data)
    cdata[row, col] = -1
    
    ax2 = fig.add_subplot(1,4,2)
    ax2.imshow(cdata)
    ax3 = fig.add_subplot(1,4,3)
    ax3.imshow(cntr)

    noise_fact = 0.4
    noise_img  = data + noise_fact * np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    cdata[row,col] = noise_img[row,col]

    cdata = np.clip(cdata, 0., 1.)

    ax4 = fig.add_subplot(1,4,4)
    ax4.imshow(cdata)

    plt.show()