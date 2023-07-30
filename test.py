import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import keyboard
from datetime import datetime
import pyttsx3  
import turtle

# import serial
# import time

# # Connect to Arduino Uno
# ser = serial.Serial('COM3', 9600)  # Replace 'COMX' with the appropriate serial port

# # Wait for Arduino to initialize
# time.sleep(2)

# # Send a command to Arduino
# command = "vibrate"  # Customize this command as per your needs
# ser.write(command.encode())

# # Close the serial connection
# ser.close()
#import streamlit as st
#import tkinter as tk

#D:\Anaconda_folder\envs\ASL_realtime\Lib\site-packages\cvzone
#Change the input shape from classificationModule inside the cvzone library according o the input size of image specified when creating the model
#####################################################################
# st.title("Web App using opencv cvzone and streamlit")
# st.title("FAce Detection")
# run = st.checkbox("Run")
# frame_window = st.image([])
#####################################################################

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
#classifier = Classifier("Model/keras_model_yes_no.h5", "Model/labels_yes_no.txt")
#classifier = Classifier("Model/vgg_16_custom_data_model.h5", "Model/labels.txt")

# classifier1 = Classifier("Model/vgg_16_updated_new_data.h5", "Model/labels3.txt")
classifier1 = Classifier("Model/vgg_16_updated_new_data.h5", "Model/labels3.txt")

#classifier2 = Classifier("Model/vgg_16_custom_data_model.h5", "Model/labels1.txt")

classifier_newdata = Classifier("Model/vgg_16_new_data.h5", "Model/new_data_labels.txt")
#default model is classifier 1
classifier=classifier1

engine = pyttsx3.init()  
engine.setProperty('rate', 50)

offset = 20
imgSize = 300

counter = 0

#labels1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'LEFT', 'M', 'N', 'NO', 'O', 'P', 'Q', 'R', 'RIGHT', 'S', 'SPACE', 'START', 'STOP', 'T', 'U', 'V', 'W', 'X', 'Y', 'YES', 'Z']
#labels=['yes','no']
#labels2 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

labels3 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S','SPACE' ,'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
#{'LEFT': 0, 'NO': 1, 'RIGHT': 2, 'START': 3, 'STOP': 4, 'YES': 5}
labels_new_data=['LEFT','NO','RIGHT','START','STOP','YES']
#default labels are labels1
labels=labels3

word=[]
temp=[]
new_text=''
append_text=''
finalBuffer=[]
counts=0
flag=True

while True:
    success, img = cap.read()
    # if success==False:
    #     print("IMage is not read properly")
    imgOutput = img.copy()
    if flag:
        img_dis=r"C:\Users\Sanora Teressa Raju\Desktop\ASL_REALTIME\white.png"
        img_dis=cv2.imread(img_dis)
        flag=False
    hands, img = detector.findHands(img)
    #print("hands is",hands)

    if keyboard.is_pressed('1'):
         classifier=classifier1
         labels=labels3
         print("Sign model selected......")
    elif keyboard.is_pressed('2'):
         classifier=classifier_newdata
         labels=labels_new_data
         print("gesture model is selected.....")
         
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        h, w, c = imgCrop.shape
        if(h==0 or w==0):
            continue

        imgCropShape = imgCrop.shape
        print("IMage crop shape is",imgCropShape)
        print("IMage white shape is",imgWhite.shape)
        aspectRatio = h / w

        # k = cv2.waitKey(33)
        # if k==27:    # Esc key to stop
        #     break
        # elif k==-1:  # normally -1 returned,so don't print it
        #     continue
        # else:
        #     print (k) # else print its value
        #     cv2.putText(imgOutput, chr(k), (i, 50), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        #     i+=15

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            # print("IMage crop 1 is",imgCrop)
            # print("Length is",len(imgCrop))
            # print("IMage crop shape 1 is",imgCrop.shape)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            print("Image resize shape is",imgResizeShape)
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            #imgWhite = cv2.resize(imgWhite, (32, 32))
            print("IMage white shape is in second",imgWhite.shape)
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print("prediction and its index are",prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            # print("IMage crop 2 is",imgCrop)
            # print("Length is",len(imgCrop))
            # print("IMage crop shape 2 is",imgCrop.shape)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            print("Image resize shape is",imgResizeShape)
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            #imgWhite = cv2.resize(imgWhite, (32, 32))
            #print("IMage white shape is in second",imgWhite.shape)
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print("prediction and its index are",prediction, index)
        
        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)

        #frame_window.image(img)
##################################################################################3
        # font = cv2.FONT_HERSHEY_SIMPLEX
        
        # i = 10
        # while(1):
        #     #cv2.imshow('img',img)

        #     k = cv2.waitKey(33)
        #     if k==27:    # Esc key to stop
        #         break
        #     elif k==-1:  # normally -1 returned,so don't print it
        #         continue
        #     else:
        #         #print (k) # else print its value
        #         cv2.putText(img,labels[index], (i, 50), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        #         i+=15

        #################################################################################################
        print("Labels are",labels[index])
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # i = 10
        # while(1):
        #     cv2.imshow('img',imgOutput)

        #     k = cv2.waitKey(33)
        #     if k==27:    # Esc key to stop
        #         break
        #     elif k==-1:  # normally -1 returned,so don't print it
        #         continue
        #     else:
        #         print (k) # else print its value
        #         cv2.putText(imgOutput, chr(k), (i, 50), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        #         i+=15
        # i=10
        # print("Words are",word)
        # for j in range(len(temp)):
        #     if keyboard.is_pressed("c"):
        #         word.append(labels[index])
        #         cv2.putText(imgOutput, word[j], (i, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        #     i+=15


        # try:
		# 	self.textBrowser.setText("\n         "+str(img_text))
	    # except:
		# 	pass
        
        if cv2.waitKey(1) == ord('c'):
            try:
                counts+=1
                append_text+=labels[index]
                if(append_text=='SPACE'):
                     labels[index]=' '
                new_text+=labels[index]
                print("Append text is",append_text)
                print("New text is",new_text)
                # if not os.path.exists('./TempGest'):
                #     os.mkdir('./TempGest')
                # img_names = "./TempGest/"+"{}{}.png".format(str(counts),str(img_text))
                # save_imgs = cv2.resize(mask1, (image_x, image_y))
                # cv2.imwrite(img_names, save_imgs)
                # self.textBrowser_4.setText(new_text)
                cv2.putText(imgOutput, new_text, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                img_copy=img_dis.copy()
                cv2.putText(img_dis, new_text, (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

                #turtle.write(new_text)
                #turtle.done()
                #root = tk.Tk()
                #root.geometry("300x300")
                # T = tk.Text(root, height=2, width=30)
                # T.pack()
                # T.insert(tk.END, new_text)
                # tk.mainloop()

            except:
                    append_text+=''
                
            if(len(append_text)>1):
                    finalBuffer.append(append_text)
                    append_text=''
            else:
                    finalBuffer.append(append_text)
                    append_text=''

            print("final buffer is",finalBuffer)
        
        
        # if keyboard.is_pressed('Backspace'):
        #     if((len(new_text))!=0): 
        #         new_text = new_text[:-1] 
        #         print("new text is",new_text)
        #         finalBuffer = finalBuffer[:-1]
        #         print("Final buffer is",finalBuffer)
        #         cv2.putText(img_dis, new_text, (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)          

        print("check11111111111111111111111111")
        if keyboard.is_pressed('q'):
            break
        
        print("check222222222222222222222222222")
        if keyboard.is_pressed('s'):
            print("saving....")
            print("length of final buffer is",len(finalBuffer))
            if(len(finalBuffer)>=1):
                # f=open("temp.txt","w")
                # for i in finalBuffer:
                #     f.write(i)
                # f.close()
                # at specified directory
                print("inside saving")
                x = datetime.now()
                f = r"C:\Users\Sanora Teressa Raju\Desktop\ASL_REALTIME\logs\\" + x.strftime('%d-%m-%Y-%H-%M-%S.txt')
                f = r"C:\Users\Sanora Teressa Raju\Desktop\ASL_REALTIME\logs\\" + "sam.txt"
                with open(f, 'w') as fp:
                     for i in finalBuffer:
                        fp.write(i)
                     fp.close()
                     print('created', f)
            #break


        if keyboard.is_pressed('v'):
        # convert this text to speech    
            engine.say(new_text)
            print("inside voice converter................")
            x = datetime.now()
            v_file = r"C:\Users\Sanora Teressa Raju\Desktop\ASL_REALTIME\voice_logs\\" + x.strftime('%d-%m-%Y-%H-%M-%S.mp3')
            engine.save_to_file(new_text, v_file)  
            # play the speech  
            engine.runAndWait() 

    if keyboard.is_pressed('q'):
        break

    if keyboard.is_pressed('v'):
    # convert this text to speech    
        engine.say(new_text)
        print("inside voice converter................")
        x = datetime.now()
        v_file = r"C:\Users\Sanora Teressa Raju\Desktop\ASL_REALTIME\voice_logs\\" + x.strftime('%d-%m-%Y-%H-%M-%S.mp3')
        engine.save_to_file(new_text, v_file)  
        # play the speech  
        engine.runAndWait() 

    if keyboard.is_pressed('s'):
            print("saving....")
            if(len(finalBuffer)>=1):
                # f=open("temp.txt","w")
                # for i in finalBuffer:
                #     f.write(i)
                # f.close()
                # at specified directory
                x = datetime.now()
                f = r"C:\Users\Sanora Teressa Raju\Desktop\ASL_REALTIME\logs\\" + x.strftime('%d-%m-%Y-%H-%M-%S.txt')
                with open(f, 'w') as fp:
                     for i in finalBuffer:
                        fp.write(i)
                     fp.close()
                     print('created', f)
            #break

    cv2.imshow("Image", imgOutput)
    cv2.imshow("Image_text", img_dis)
    cv2.waitKey(1) 