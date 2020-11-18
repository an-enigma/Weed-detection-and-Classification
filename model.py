import h5py
import sys,os
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
FIXED_ANCHOR=['Black-grass',
 'Charlock',
 'Cleavers',
 'Common Chickweed',
 'Common wheat',
 'Fat Hen',
 'Loose Silky-bent',
 'Maize',
 'Scentless Mayweed',
 'Shepherd’s Purse',
 'Small-flowered Cranesbill',
 'Sugar beet']

def closest(x):
    x=list(x[0])
    mx=max(x)
    for i in range(len(x)):
        if(x[i]==mx):
            return i

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")

img_counter = 0
upper_left = (320, 30)
bottom_right = (900,650)
while True:
    ret, frame = cam.read()
    cv2.rectangle(frame, upper_left, bottom_right, (0, 0, 0), 5)
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
    # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        #img = "opencv_frame_{}.png".format(img_counter)
        #cv2.imwrite(img, frame)
        # print("{} written!".format(img))
        roi = frame[30:650, 320:900]
        tup=(128,128)
        imgr = cv2.resize(roi, tup, interpolation = cv2.INTER_AREA)
        #print(frame)
        
        #imgr=im.resize(tup)
        img_counter += 1
        pred=[]
        pred.append(imgr)
        pre=np.asarray(pred)
        model=tf.keras.models.load_model('weeds.h5')
        p=model.predict(pre)
        print(p)
        Prediction=closest(p)
        janstr=FIXED_ANCHOR[Prediction]
        print("Answer:",janstr)
        # if(p[0][0]<0.4):
        #     print("it's Benign skin lesion")
        #     janstr='This is imperfect'
        # elif(p[0][0]>0.6):
        #     print("it's a Malign lesion")
        #     janstr='This is perfect'
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,500)
        fontScale              = 3
        fontColor              = (1,1,1)
        lineType               = 2

        cv2.putText(frame,janstr,bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
        cv2.imshow("Result of the input",frame)

cam.release()

cv2.destroyAllWindows()


