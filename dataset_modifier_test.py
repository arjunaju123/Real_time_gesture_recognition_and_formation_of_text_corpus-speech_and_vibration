import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "crop_full_test"
counter = 0

#img = r"C:\Users\54721\OneDrive\Desktop\ASL_realtime\new_img_dir\A\Image_1680769197.0531795.jpg"
#img = cv2.imread(r"C:\Users\54721\OneDrive\Desktop\ASL_realtime\asl_alphabet_train\asl_alphabet_train\B\B16.jpg")
# print("image type is",type(img))
# hands, img = detector.findHands(img)
lst=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y','Z']

count=0

# initializing character
ch = 'A'
 
for sample_img in os.listdir(f"C:\\Users\\54721\\OneDrive\\Desktop\\ASL_realtime\\asl_alphabet_test\\asl_alphabet_test"):
    print("sample image is",sample_img)
    count=count+1
    print("count is",count)
    #print("character is",lst[ele])
    # if(count>1000):
    #     break
    img = cv2.imread(f"C:\\Users\\54721\\OneDrive\\Desktop\\ASL_realtime\\asl_alphabet_test\\asl_alphabet_test\\{sample_img}")
    hands, img = detector.findHands(img)
    print("status of hand is",hands)
    if not hands:
        print(ch, "not detected")
        ch = chr(ord(ch) + 1)
        print("character incremented hand not detected :",ch)
    print("image type of img is",type(img))
    if hands:
        #print("insideeee!!!!!!!!!!!!!!!!!!!!!!!!!!")
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        h, w, c = imgCrop.shape
        if(h==0 or w==0):
            print("Character is",ch)
            continue #Other wise error cv2.error: OpenCV(4.7.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\resize.cpp:4062: error: (-215:Assertion failed) !ssize.empty() in function 'cv::resize'

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)  
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        # print("character here is",lst[ele])
        # print("ele is",ele)
        # if not os.path.exists(f"C:\\Users\\54721\\OneDrive\\Desktop\\ASL_realtime\\crop_full_test\\{lst[ele]}"):
    
        #     os.makedirs(f"C:\\Users\\54721\\OneDrive\\Desktop\\ASL_realtime\\crop_full_test\\{lst[ele]}")

        cv2.imwrite(f'{folder}/{ch}_test.jpg',imgWhite)
        
        print("######Before character incremented hand detected######## :",ch)
        ch = chr(ord(ch) + 1)
        print("######character incremented hand detected######## :",ch)
