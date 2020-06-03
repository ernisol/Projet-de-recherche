import numpy as np
import cv2
import os


ROOT = 'C:\\Users\\Erwan\\Desktop\\Erwan backup\\École\\4A\\Projet\\Données d\'entrainement\\Données maison 2'
os.chdir(ROOT)
videos = os.listdir()


for i,vid in enumerate(videos):
    cap = cv2.VideoCapture(vid)
    success,image = cap.read()
    count = 0
    success = True
    print('Traitement de la vidéo ' + str(i+1))
    
    while success:
        success,image = cap.read()
        if image is None:
            break
        if count%20 == 0:
            print('Saving frame ' + str(count))
            a = cv2.imwrite('Frames\\vid'+str(i+1)+'-frame-'+str(count)+'.png', image)
            print(a)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1
        
    cap.release()
