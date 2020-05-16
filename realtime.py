import cv2
import sys
from keras.models import load_model
import time
import numpy as np
from decimal import Decimal
from cnn_model import define_model, model_weights

#first we'll the resie the image
def resize_img(image_path):
    img = cv2.imread(image_path,1)
    img = cv2.resize(img,(48,48))
    return True 

def realtime_emotions():
    model = define_model()
    model = model_weights(model)
    print('Loaded')
    #saving loacation for the image
    saving_location='C:/Users/KIIT/Desktop/emotion_detect/saving_location/img1.jpg'
    
    #matrix for the predictions 
    result = np.array((1,7))
    #flag for knowing if the prediction has started or not 
    start = False
    #loading the haar cascade
    faceCascade = 'C:/Users/KIIT/Desktop/emotion_detect/haarcascades/haarcascade_frontalface_default.xml'
    #list of emotions
    EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']
    
    #now assigning the emojis 
    
    emoji = []
    for index,emotion in enumerate(EMOTIONS):
       emoji.append(cv2.imread('C:/Users/KIIT/Desktop/emotion_detect/emojis/' + emotion.lower()  + '.png', -1))
    

    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 640)
    video_capture.set(4, 480) 
    
    prev_time = time.time()
    
    #start the feed
    while True:
        
        #capturing farme by frame
        ret, frame = video_capture.read()
        # mirror the frame
        frame = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #finding faces using haar cascades
        faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        #the box aroung the face
        for (x, y, w, h) in faces:
                # required region for the face
            roi_color = frame[y-90:y+h+70, x-50:x+w+50]

            # save the detected face
            cv2.imwrite(saving_location, roi_color)
            # draw a rectangle bounding the face
            cv2.rectangle(frame, (x-10, y-70),
                            (x+w+20, y+h+40), (15, 175, 61), 4)
            
            curr_time = time.time()
            
            if curr_time-prev_time >=1:
                #reading the image
                img = cv2.imread(saving_location,0)
                
                if img is not None:
                    start = True
                    
                    #resizing the image
                    img = cv2.resize(img,(48,48))
                    img = np.reshape(img,(1,48,48,1))
                    
                    #doing the predictions
                    result = model.predict(img)
                    print(EMOTIONS[np.argmax(result[0])])
                
                prev_time = time.time()
                
            if start==True:
                total_sum = np.sum(result[0])
                #select the emoji
                emoji = emoji[np.argmax(result[0])]
                for index,emotion in enumerate(EMOTIONS):
                    text = str(
                        round(Decimal(result[0][index]/total_sum*100), 2) ) + "%"
                    # for drawing progress bar
                    cv2.rectangle(frame, (100, index * 20 + 10), (100 +int(result[0][index] * 100), (index + 1) * 20 + 4),
                                    (255, 0, 0), -1)
                    # for putting emotion labels
                    cv2.putText(frame, emotion, (10, index * 20 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (7, 109, 16), 2)
                    # for putting percentage confidence
                    cv2.putText(frame, text, (105 + int(result[0][index] * 100), index * 20 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                for c in range(0, 3):
                    # for doing overlay we need to assign weights to both foreground and background
                    foreground = emoji[:, :, c] * (emoji[:, :, 3] / 255.0)
                    background = frame[350:470, 10:130, c] * (1.0 - emoji[:, :, 3] / 255.0)
                    frame[350:470, 10:130, c] = foreground + background
                    
            break
        
        cv2.imshow('Video',frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video_capture.release()
    cv2.destroyAllWindows()