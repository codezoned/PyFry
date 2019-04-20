import cv2
from PIL import Image, ImageOps, ImageEnhance
import os
from utils.utils import Colors
from imutils import face_utils
import dlib
'''
TODO: -> Compressing (Crushing) and back (to increase noise) :: DONE
      -> Applying Red and Orange hue filters for classic deep fry look :: DONE
      -> Detecting eye coordinates :: DONE
      -> applying the deepfry eye flare in the center (irisCoords is returning those coordinates)
'''
def irisCoords(eye):
    #Finding the center point of th eye using the average outer extremes average of the eyes
    mid = (eye[0] +eye[3])/2
    mid = (int(mid[0]), int(mid[1]))
    return mid

def generateHue(img):
    #Generating and increasing prominency of red band of the image
    img = img.convert('RGB')
    red = img.split()[0] #(R,G,B)
    red = ImageEnhance.Contrast(red).enhance(2.0)
    red = ImageEnhance.Brightness(red).enhance(1.5)
    red = ImageOps.colorize(red, Colors.RED, Colors.YELLOW)
    img = Image.blend(img, red, 0.77)
    #Keeping a 100% sharpness value for now, But would probably be better with a higher sharpness value
    img = ImageEnhance.Sharpness(img).enhance(150)
    return img

def crushAndBack(img):
    img = img.convert('RGB')
    w,h = img.width, img.height
    img = img.resize((int(w ** .95), int(h ** .95)), resample=Image.LANCZOS)
    img = img.resize((int(w ** .90), int(h ** .90)), resample = Image.BILINEAR)
    img = img.resize((int(w ** .90), int(h ** .90)), resample = Image.BICUBIC)
    img = img.resize((w,h), resample = Image.BICUBIC)
    return img

def main():
    # Initialising dlib for frontal facial features
    flare = Image.open('flare.png')
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("assets\shape_predictor_68_face_landmarks.dat")

    (lS, lE) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rS, rE) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    
    imgCV = cv2.imread('meme1.jpg')
    img = Image.open('meme1.jpg')

    gray = cv2.cvtColor(imgCV, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lS:lE]
        rightEye = shape[rS:rE]
        #print(leftEye)
        #print(irisCoords(leftEye))
        #leftEyeHull = cv2.convexHull(leftEye)
        #rightEyeHull = cv2.convexHull(rightEye)
        #cv2.drawContours(imgCV, [leftEyeHull], -1, (0, 255, 0), 1)
        #cv2.drawContours(imgCV, [rightEyeHull], -1, (0, 255, 0), 1)
        #cv2.circle(imgCV, irisCoords(leftEye),5, (0.255,255), 1) 
        #cv2.circle(imgCV, irisCoords(rightEye),5, (0.255,255), 1) 
    #rightEyeArray = []
    #rightEyeArray.append(irisCoords(leftEye))
    eyeLeft = (leftEye[0],(leftEye[1] +leftEye[2])/2,leftEye[3],(leftEye[4]+leftEye[5])/2)
    eyeLeft = (leftEye[0], leftEye[1])
    
    img = img.convert('RGB')
    img = crushAndBack(img)
    img = generateHue(img)
    rec0=(leftEye[1][0]-100,leftEye[1][1]-100)
    rec1=(leftEye[4][0]+100,leftEye[4][1]+100)
    rec2=(rightEye[1][0]-100,rightEye[1][1]-100)
    rec3=(rightEye[4][0]+100,rightEye[4][1]+100)
    print("Area for left eye",rec0,rec1)
    print("Area for right eye",rec2,rec3)
    area=(rec0[0],rec0[1],rec1[0],rec1[1])
    area1=(rec2[0],rec2[1],rec3[0],rec3[1])
    flare1=flare.resize((rec1[0]-rec0[0],rec1[1]-rec0[1]))
    flare2=flare.resize((rec3[0]-rec2[0],rec3[1]-rec2[1]))
    img.paste(flare1,area,flare1)
    img.paste(flare2,area1,flare2)
      
    
    #img.paste(flare,eyeLeft,flare)

    img.show()
    img.save('output.jpg')

    #img.save('sample.jpg','jpeg')

if __name__ == '__main__':
    main()
