import cv2
from PIL import Image
import os
from imutils import face_utils
import dlib
'''
TODO: -> Compressing (Crushing) and back (to increase noise) :: DONE
      -> Applying Red and Orange hue filters for classic deep fry look
      -> Detecting eye coordinates and applying the deepfry eye flare in the center
'''
def crushAndBack(img):
    img = img.convert('RGB')
    w,h = img.width, img.height
    img = img.resize((int(w ** .95), int(h ** .95)), resample=Image.LANCZOS)
    img = img.resize((int(w ** .85), int(h ** .85)), resample = Image.BILINEAR)
    img = img.resize((int(w ** .70), int(h ** .70)), resample = Image.BICUBIC)
    img = img.resize((w,h), resample = Image.BICUBIC)
    return img

def main():
    # Initialising dlib for frontal facial features
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("assets\shape_predictor_68_face_landmarks.dat")

    (lS, lE) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rS, rE) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    
    imgCV = cv2.imread('test.jpg')

    gray = cv2.cvtColor(imgCV, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lS:lE]
        rightEye = shape[rS:rE]
        print(leftEye)
    cv2.imshow("Frame", imgCV)
    cv2.waitKey(0)
    #img.save('sample.jpg','jpeg')

if __name__ == '__main__':
    main()
