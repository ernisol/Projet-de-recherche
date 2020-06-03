import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
#%% Figure 1 : fonctionnement de la méthode directe

ROOT = 'C:\\Users\\Erwan\\Desktop\\Erwan backup\\École\\4A\\Projet\\Données d\'entrainement\\Données maison 2'
os.chdir(ROOT)

def Figure1(interesting_frame = 920):
    cap = cv2.VideoCapture('WIN_20200602_21_35_03_Pro.mp4')
    bound = 5
    
    for frame_count in range(1, interesting_frame-5):
        ret, frame = cap.read()
    
    frames_list = []
    for i in range(bound):
        ret, frame = cap.read()
        frames_list.append(frame)
    
    frame = frames_list[-1]
    mean  = np.mean(np.array(frames_list), axis = 0)
    mean = mean.astype(np.uint8)
    
    frame_sub = cv2.absdiff(frame, mean)
    
    grey = cv2.cvtColor(frame_sub, cv2.COLOR_RGB2GRAY)
    grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    
    plt.figure()
    
    for i in range(5):
        ax = plt.subplot(2,5,i+1)
        ax.set_title('Image '+str(i+1))
        plt.imshow(frames_list[i])
        plt.axis('off')
    
    ax = plt.subplot(2,5,7)
    plt.imshow(frames_list[-1])
    ax.set_title('Image courante')
    plt.axis('off')
    
    ax = plt.subplot(2,5,8)
    plt.imshow(mean)
    ax.set_title('Image moyenne')
    plt.axis('off')
    
    ax = plt.subplot(2,5,9)
    plt.imshow(grey_3_channel)
    ax.set_title('Résultat de la soustraction')
    plt.axis('off')
    
    plt.show()
    
    cap.release()

#%% Figure 2 : fonctionnement de la méthode clustering

def get_cluster(frame, mean_images, centers):
    r, g, b = get_mean_colors(frame)
    feature = np.array([get_contrast(frame), r, g, b])/norming_param
    dist = 1e8
    cluster = -1
    for i in range(np.shape(centers)[0]):
        dist2 = np.linalg.norm(feature - centers[i,:])
        if dist2<dist:
            cluster = i
            dist = dist2
    return mean_images[cluster]

def Figure2(interesting_frame = 1140):
    cap = cv2.VideoCapture('WIN_20200602_21_35_03_Pro.mp4')
    
    for frame_count in range(1, interesting_frame):
        ret, frame = cap.read()
    
    ret, frame = cap.read()
    
    mean2 = get_cluster(frame, imgs, model.cluster_centers_)
    mean2 = mean2.astype(np.uint8)
    
    frame_sub = cv2.absdiff(frame, mean2)
    
    grey = cv2.cvtColor(frame_sub, cv2.COLOR_RGB2GRAY)
    grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    
    plt.figure()
    
    
    ax = plt.subplot(1,3,1)
    plt.imshow(frame)
    ax.set_title('Image courante')
    plt.axis('off')
    
    ax = plt.subplot(1,3,2)
    plt.imshow(mean2)
    ax.set_title('Image moyenne')
    plt.axis('off')
    
    ax = plt.subplot(1,3,3)
    plt.imshow(grey_3_channel)
    ax.set_title('Résultat de la soustraction')
    plt.axis('off')
    
    plt.show()
    
    cap.release()
