#from deepface import DeepFace 
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

class PotraitFace:
    def __init__(self, weights_path):
        self.model=YOLO(weights_path)

    def get_embeddings(self, img):
        img_array=np.array(img) 
        
        try:
            results = self.model.predict(img_array, verbose=False,show=False,conf=0.25,device='cpu')[0]
            return results
        except:
            return None
    @staticmethod
    def magnified_coordinates(x,y,w,h):
        X=x-0.25*w if x-0.25*w>=0 else 0
        Y=y-0.25*h if y-0.25*h>=0 else 0
        W=w+0.5*w
        H=h+0.5*h
        
        return(X,Y,W,H)
    
    def get_face_coordinates(self, results):
        #print("getting face coordinates")
        crop_coordinates=[]
        if results==None:
            return crop_coordinates
        
        for result in results:
            if result.boxes is None:
                continue
            x1, y1, x2, y2 = result.boxes.xyxy.tolist()[0]
            #confidence = result.boxes.conf.tolist()[0]

            X1,Y1,W,H= self.magnified_coordinates(x1,y1,x2-x1,y2-y1)
            X2=X1+W
            Y2=Y1+H
            crop_coordinates.append((int(X1),int(Y1),int(X2),int(Y2)))
        print("Done getting face coordinates...")    
        return crop_coordinates
        
    def get_faces(self,img, coordinates):
        potraitfaces_list=[]

        #print("getting faces from coordinates")
        if coordinates==[]:
            return potraitfaces_list

        for i in range(len(coordinates)):
            x1=coordinates[i][0]
            x2=coordinates[i][2]
            y1=coordinates[i][1]
            y2=coordinates[i][3]
            
            image=np.array(img)[y1:y2,x1:x2]
            image=cv2.resize(image,(360,360),interpolation=cv2.INTER_LANCZOS4)
            potraitfaces_list.append(image)    

        #print(potraitfaces_list)
        print("Done getting faces from coordinates...")
        return potraitfaces_list
    
    
    
    
