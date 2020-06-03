import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np 
import os
import matplotlib.pyplot as plt 
import imutils
from darkflow.net.build import TFNet
import tensorflow as tf

characterRecognition = tf.keras.models.load_model('character_recognition.h5')
options = {"pbLoad": "yolo-character.pb", "metaLoad": "yolo-character.meta", "gpu":0.9}
yoloCharacter = TFNet(options)

number_cascade = cv2.CascadeClassifier('russian_plate_numbers.xml')

  
def adjusted_detect_number(img):
	number_img = img.copy()

	num_rect = number_cascade.detectMultiScale(number_img, 
                                              scaleFactor = 1.2, 
                                              minNeighbors = 5)
	for (x, y, w, h) in num_rect:
		cv2.rectangle(number_img, (x, y), 
					(x + w, y + h), (255, 0, 255), 1)
		new_img = number_img[y:y + h, x:x + w]
	return new_img

img_auto = '../static/data/user_image.jpeg'

img = cv2.imread(img_auto)
img_copy = img.copy()
img_cropped = adjusted_detect_number(img_copy)

plt.imshow(img_cropped)

def secondCrop(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,0)
    contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if(len(areas)!=0):
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        secondCrop = img[y:y+h,x:x+w]
    else: 
        secondCrop = img
    return secondCrop

def cnnCharRecognition(img):
    dictionary = {0:'0', 1:'1', 2 :'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
    11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K',
    21:'L', 22:'M', 23:'N', 24:'P', 25:'Q', 26:'R', 27:'S', 28:'T', 29:'U',
    30:'V', 31:'W', 32:'X', 33:'Y', 34:'Z'}

    blackAndWhiteChar=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackAndWhiteChar = cv2.resize(blackAndWhiteChar,(75,100))
    image = blackAndWhiteChar.reshape((1, 100,75, 1))
    image = image / 255.0
    new_predictions = characterRecognition.predict(image)
    char = np.argmax(new_predictions)
    return dictionary[char]

def firstCrop(img, predictions):
    predictions.sort(key=lambda x: x.get('confidence'))
    xtop = predictions[-1].get('topleft').get('x')
    ytop = predictions[-1].get('topleft').get('y')
    xbottom = predictions[-1].get('bottomright').get('x')
    ybottom = predictions[-1].get('bottomright').get('y')
    firstCrop = img[ytop:ybottom, xtop:xbottom]
    cv2.rectangle(img,(xtop,ytop),(xbottom,ybottom),(0,255,0),1)
    return firstCrop

def secondCrop(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,127,255,0)
    contours,_ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if(len(areas)!=0):
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        secondCrop = img[y:y+h,x:x+w]
    else: 
        secondCrop = img
    return secondCrop

def yoloCharDetection(predictions,img):
	charList = []
	positions = []
	for i in predictions:
		if i.get("confidence")>0.10:
			xtop = i.get('topleft').get('x')
			positions.append(xtop)
			ytop = i.get('topleft').get('y')
			xbottom = i.get('bottomright').get('x')
			ybottom = i.get('bottomright').get('y')
			char = img[ytop:ybottom, xtop:xbottom]
			cv2.rectangle(img,(xtop,ytop),( xbottom, ybottom ),(255,0,0),1)
			charList.append(cnnCharRecognition(char))

	# cv2.imshow(img)
	sortedList = [x for _,x in sorted(zip(positions,charList))]
	licensePlate="".join(sortedList)
	return licensePlate

cor2 =  img_cropped.copy()
predictions = yoloCharacter.return_predict(cor2)
cor2 =  img_cropped.copy()
number_predict = yoloCharDetection(predictions,cor2)
print(number_predict) #list number predict