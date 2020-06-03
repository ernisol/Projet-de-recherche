import numpy as np
import cv2
import os

ROOT = 'C:\\Users\\Erwan\\Desktop\\Erwan backup\\École\\4A\\Projet\\Données d\'entrainement\\AAU RainSnow Traffic Surveillance Dataset'
os.chdir(ROOT)
#cap = cv2.VideoCapture('Ringvej\\Ringvej-1\\cam1.mkv')
cap = cv2.VideoCapture(0)

dim = (int(cap.get(3)), int(cap.get(4)))
images_history = []
bound = 5
mean = np.zeros(dim, dtype = np.uint8)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter('substraction_pedestrian2_history='+str(bound)+'frames.avi', fourcc, 20.0, (2*dim[0], dim[1]))

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
    
    frame = cv2.absdiff(frame, mean)
        
    # Our operations on the frame come here
    
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Make the grey scale image have three channels
    grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    
    img = np.hstack((images_history[-1], grey_3_channel))
    out.write(img)
    
    # Display the resulting frame
    cv2.imshow('substraction', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()