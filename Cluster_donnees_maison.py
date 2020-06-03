import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
import cv2 # pip install opencv-python

#%%

ROOT = 'C:\\Users\\Erwan\\Desktop\\Erwan backup\\École\\4A\\Projet' + '\\Données d\'entrainement\\Données maison 2\\Frames'
def getpaths():
#    pathlist = []
#    path = ROOT
#    branchs = ['Ringvej-1','Ringvej-2','Ringvej-3']
#    for b in branchs:
#        path =  ROOT + '\\' + b
#        liste = os.listdir(path)
#        for subt in liste:
#            if subt[:5] == 'cam1-':
#                path =  ROOT + '\\' + b + '\\'  + subt
#                pathlist.append(path)
    pathlist1 = os.listdir(ROOT)
    pathlist2 = [ROOT+'\\'+p for p in pathlist1]
    return pathlist2


pathlist = getpaths()

#%%

def show_im(path):
    plt.imread(path)
    im = Image.open(path)
    plt.figure()
    A = np.array(im)
    A = cv2.resize(A, dsize=(238,158), interpolation=cv2.INTER_CUBIC)
    plt.imshow(A, cmap = 'gray')

show_im(pathlist[0])
show_im(pathlist[-1])
show_im(pathlist[150])
show_im(pathlist[250])


#%%
errlist = []

def get_array(path):
    im = Image.open(path)
    return np.array(im)

def get_contrast(A):
    B = np.mean(A, axis = 2)
    m = np.min(B)
    M = np.max(B)
    if m==M:
        return 0
    contrast = (M-m)/(m+M)
    return contrast

def get_mean_colors(A):
    r = np.mean(A[:,:,0])
    g = np.mean(A[:,:,1])
    b = np.mean(A[:,:,2])
    return r, g, b

def get_luminosity(A):
    return np.mean(A)

def get_all_features():
    features = []
    for path in tqdm(pathlist):
        A = get_array(path)
        try:
            r, g, b = get_mean_colors(A)
            feature = [get_contrast(A), r, g, b]
        except:
            errlist.append(path)
            features.append([0,0])
        features.append(feature)
    return features


#%%

features = np.array(get_all_features())

#%%
noerror_features = []
noerror_pathlist = []
for i, p in enumerate(pathlist):
    if features[i,0] >= 0.01 and features[i,1] >= 0.01 and features[i,0]<=1 and features[i,1]<=255:
        noerror_pathlist.append(p)
        noerror_features.append(features[i])

noerror_features = np.array(noerror_features)
norming_param = noerror_features.max(axis=0)
features_normed = noerror_features / norming_param

#%%

def plot_features(f):
    plt.figure()
    plt.scatter(f[:,2], f[:,3])
    plt.xlabel('contraste')
    plt.ylabel('luminosité')
    plt.title('luminosité moyenne et contraste pour chaque image')
    plt.show()

plot_features(features_normed)

#%%

def clusters(f, r):
    inertias = []
    for k in tqdm(range(1, r)):
        model = KMeans(n_clusters = k)
        model.fit(f)
        inertias.append(model.inertia_)
    return(inertias)


def find_best_k(f, r):
    inertias = clusters(f, r)
    suggested_k = 1
    bestangle = np.pi
    angles = []
    for k in range(2, r-1):
        y1 =inertias[k-2]
        y2 =inertias[k-1]
        y3 = inertias[k]
        angle = np.pi - np.arctan(y1-y2) + np.arctan(y2-y3)
        print(180*angle/np.pi)
        angles.append(angle)
        if angle < bestangle:
            print('better')
            bestangle = angle
            suggested_k = k
    plt.figure()
    plt.plot(range(1,r), inertias)
    plt.xlabel('k')
    plt.ylabel('inertie')
    for i,a in enumerate(angles):
        plt.text(i+2,inertias[i+1], str(int(10*180*a/np.pi)/10))
    plt.title('inertie en fonction du nombre de clusters k')
    plt.show()
    print('suggested k = '+ str(suggested_k))
    return suggested_k


bestk = find_best_k(features_normed, 15)

#%%
def cluster(f, k):
    model = KMeans(n_clusters = k)
    model.fit(f)
    labels = model.labels_
    colorlist = np.array(['red', 'blue', 'green', 'yellow', 'black', 'brown'])
    try:
        plt.figure()
        plt.scatter(f[:,0], f[:,1], color = colorlist[labels])
        plt.xlabel('contraste')
        plt.ylabel('luminosité rouge')
        plt.title('luminosité moyenne et contraste pour chaque image')
        plt.show()
    except:
        print(np.array(colorlist)[labels])


cluster(features_normed, bestk)

#%%

    
def get_mean_images(f,k):
    model = KMeans(n_clusters = k)
    model.fit(f)
    labels = model.labels_
    images = []
    for c in tqdm(range(k)):
        mean_img = np.zeros_like(get_array(pathlist[0]))
        img_list = []
        for i,p in enumerate(noerror_pathlist):
            if labels[i]==c:
                A = get_array(p)
#                try: 
#                    mean_img += A
#                except:
#                    print(np.shape(A))
#                    print(np.shape(mean_img))
                img_list.append(A)
                
        #images.append(mean_img/count)
        images.append(np.mean(img_list, axis = 0))
    return images, labels, model

imgs,labels, model = get_mean_images(features_normed, bestk)
#%%
plt.figure()
nrow = int(np.sqrt(bestk))
for i, im in enumerate(imgs):
    ax = plt.subplot(nrow, np.ceil(bestk/nrow), i+1)
    ax.set_title('Cluster '+ str(i+1))
    plt.imshow(im.astype(int), cmap = 'gray')

#%% test soustraction arrière-plan
def test_soustraction(nrow = 3):
    n = len(noerror_pathlist)
    plt.figure()
    A = None
    for i in range(nrow):
        index = np.random.randint(n)
        path = noerror_pathlist[index]
        A = get_array(path)
        plt.subplot(nrow,3,3*i+1)
        plt.imshow(A, cmap = 'gray')
        A = A.astype(float)
        mean_image = imgs[labels[index]].astype(int)
        A -= mean_image
        A = np.sqrt(A*A)
#        A = substract(A, mean_image)
        plt.subplot(nrow,3,3*i+2)
        plt.imshow(mean_image, cmap = 'gray')
        plt.subplot(nrow,3,3*i+3)
        plt.imshow(np.mean(A, axis = 2), cmap = 'gray')
#        plt.imshow(A, cmap = 'gray')
    plt.show()
    return(A)
    
#def substract(imA, imB):
#    dim = np.shape(imA)
#    mask = abs(np.mean(imA-imB, axis = 2))
#    result = imA[:,:,:]
#    for i in range(dim[0]):
#        for j in range(dim[1]):
#            if mask[i,j] <=50:
#                result[i,j,0]=0
#                result[i,j,1]=0
#                result[i,j,2]=0
#    print(np.shape(result))
#    return result.astype(int)

A = test_soustraction()