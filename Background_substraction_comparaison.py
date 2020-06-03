import numpy as np
import cv2
import os



ROOT = 'C:\\Users\\Erwan\\Desktop\\Erwan backup\\École\\4A\\Projet\\Données d\'entrainement\\Données maison 2'
os.chdir(ROOT)
cap = cv2.VideoCapture('WIN_20200602_21_36_16_Pro.mp4')
#cap = cv2.VideoCapture('WIN_20200602_21_35_03_Pro.mp4')

#cap = cv2.VideoCapture(0)

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
        

dim = (int(cap.get(3)), int(cap.get(4)))
images_history = []
bound = 5
mean = np.zeros(dim, dtype = np.uint8)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter('donnees_maisons_comparatif_soustraction2='+str(bound)+'frames.avi', fourcc, 20.0, (3*dim[0], dim[1]))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        break
    
    images_history.append(frame)
    if len(images_history) > bound:
        images_history.pop(0)
        
    if len(images_history)>0:
        mean  = np.mean(np.array(images_history), axis = 0)
        mean = mean.astype(np.uint8)
    
    mean2 = get_cluster(frame, imgs, model.cluster_centers_)
    mean2 = mean2.astype(np.uint8)
    
    framesub1 = cv2.absdiff(frame, mean)
    framesub2 = cv2.absdiff(frame, mean2)
        
    # Our operations on the frame come here
    
    grey = cv2.cvtColor(framesub1, cv2.COLOR_RGB2GRAY)
    # Make the grey scale image have three channels
    grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    
    grey2 = cv2.cvtColor(framesub2, cv2.COLOR_RGB2GRAY)
    # Make the grey scale image have three channels
    grey_3_channel2 = cv2.cvtColor(grey2, cv2.COLOR_GRAY2BGR)
    
    img = np.hstack((frame, grey_3_channel, grey_3_channel2))
    out.write(img)
    
    # Display the resulting frame
    cv2.imshow('substraction', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()