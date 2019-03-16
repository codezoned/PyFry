import cv2
from PIL import Image
import os
'''
TODO: -> Compressing (Crushing) and back (to increase noise)
      -> Applying Red and Orange hue filters for classic deep fry look
      -> Detecting eye coordinates and applying the deepfry eye flare in the center
'''
def crushAndBack(img):
    img = img.convert('RGB')
    w,h = img.width, img.height
    img = img.resize(((int(w ** 0.7), int(h ** 0.7)), resample = Image.LANCZOS))
    img = img.resize(((int(w ** 0.6), int(h ** 0.6)), resample = Image.BILINEAR))
    img = img.resize(((int(w ** 0.5), int(h ** 0.5)), resample = Image.BICUBIC))
    img = img.resize((w,h), resample = Image.BICUBIC)
    return img

def main():
    img = Image.open('test.jpeg')
    img = crushAndBack(img)    
    img.save('sample.jpg','jpeg')

if __name__ == '__main__':
    main()
