import cv2
from PIL import Image, ImageOps, ImageEnhance
import os
from utils.utils import Colors
from imutils import face_utils
import dlib
'''
TODO: -> Compressing (Crushing) and back (to increase noise) :: DONE
      -> Applying Red and Orange hue filters for classic deep fry look
      -> Detecting eye coordinates and applying the deepfry eye flare in the center::DONE
'''
def irisCoords(eye):
    #Finding the center point of the eye using the average outer extremes average of the eyes
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
def addFlare(img):
    ''' Initialising dlib for frontal facial features '''
    flare = Image.open('flare.png')
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("assets\shape_predictor_68_face_landmarks.dat")

    (lS, lE) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rS, rE) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    
    imgCV = cv2.imread('test.jpg')
    #imgCV = cv2.imread('test2.jpg')

    gray = cv2.cvtColor(imgCV, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lS:lE]
        rightEye = shape[rS:rE]
    '''
        Assigning an area to paste the flare png Using the coordinates given by the Dlib module
        ln,rn is the distance between the top left and bottom right of the iris multiplied by 4.
        This is used to find the basic coordinates of the area in which the flare image will be pasted
    '''

    rn=(rightEye[4][0]-rightEye[0][0])*3
    ln=(leftEye[4][0]-leftEye[0][0])*3

    rec0=(leftEye[1][0]-ln,leftEye[1][1]-ln)
    rec1=(leftEye[4][0]+ln,leftEye[4][1]+ln)
     
    rec2=(rightEye[1][0]-rn,rightEye[1][1]-rn)
    rec3=(rightEye[4][0]+rn,rightEye[4][1]+rn)
    
    print("Area for left eye",rec0,rec1)
    print("Area for right eye",rec2,rec3)

    """ Area Assignment for left eye and right eye"""
    areaLeft=(rec0[0],rec0[1],rec1[0],rec1[1])
    areaRight=(rec2[0],rec2[1],rec3[0],rec3[1])
    
    """ Resizing the flare image to fit the area"""
    flareLeft=flare.resize((rec1[0]-rec0[0],rec1[1]-rec0[1]))
    flareRight=flare.resize((rec3[0]-rec2[0],rec3[1]-rec2[1]))
    
    """Pasting the flare image on the area.
       Third parameter is an alpha channel that provides transparency for the png"""
    img.paste(flareLeft,areaLeft,flareLeft)
    img.paste(flareRight,areaRight,flareRight)
    return img


def main():
    img = Image.open('test.jpg')
    #img = Image.opne('test2.jpg')
    img = img.convert('RGB')
    img = crushAndBack(img)
    img = generateHue(img)
    img = addFlare(img)
       
    img.show()
    #img.save('output2.jpg')
    img.save('output.jpg')

   

if __name__ == '__main__':
    main()
