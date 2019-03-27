import cv2
from PIL import Image
import os
import imutils
'''
TODO: -> Compressing (Crushing) and back (to increase noise)
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
    img = Image.open('test.jpg')
    img = img.convert('RGB')
    img = crushAndBack(img)
    img.show()
    #img.save('sample.jpg','jpeg')

if __name__ == '__main__':
    main()
