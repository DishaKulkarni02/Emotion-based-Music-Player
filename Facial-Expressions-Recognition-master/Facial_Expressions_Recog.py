import cv2
from PIL import Image
from PIL import ImageTk
import threading
import tkinter as tk
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np
import webbrowser
from gtts import gTTS
import os
from tkinter import *

def button1_clicked(videoloop_stop):
    threading.Thread(target=videoLoop, args=(videoloop_stop,)).start()
    
def videoLoop(mirror=False):
    No = 0
    cap = cv2.VideoCapture(No)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier =load_model('Emotion_little_vgg.h5')
    class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

    while True:
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            cv2.imwrite("img_name22.png", frame)

            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                preds = classifier.predict(roi)[0]
                label=class_labels[preds.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)

                if label == ("Sad"):
                    print(label)
                    cv2.imwrite("Sad_emo.png", frame)

                    mytext = 'wake up wake up'
                    language = 'en'
                    myobj = gTTS(text=mytext, lang=language, slow=False)
                    myobj.save("output.mp3")
                    os.system("start output.mp3")
                    exit()
                    break;
             
            else:
                cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        
        cv2.imshow('Emotion Detector',frame)
        mood = labels
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(label)
    cv2.imwrite("emotion.png", frame) 

    url = "https://open.spotify.com/search/{} songs".format(label)

    webbrowser.open(url)
    
# videoloop_stop is a simple switcher between ON and OFF modes
videoloop_stop = [False]

root = tk.Tk()
root.geometry("1920x1080+0+0")

button1 = tk.Button(root, text=" Start Camera ", bg="blue",fg="white", font=("", 35), command=lambda: button1_clicked(videoloop_stop))
button1.place(x=550, y=300, width=400, height=100)

root.mainloop()