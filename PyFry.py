import cv2
from PIL import Image, ImageOps, ImageEnhance
from PIL import ImageTk
import os
from utils.utils import Colors
from imutils import face_utils
import dlib
import tkinter as tk
from tkinter import filedialog
'''
TODO: -> Compressing (Crushing) and back (to increase noise) :: DONE
      -> Applying Red and Orange hue filters for classic deep fry look :: DONE
      -> Detecting eye coordinates and applying the deepfry eye flare in the center::DONE

'''
# def userInput():
#     #Allowing user to choose the image that has to be deepfried
#     root = tk.Tk()
#     root.withdraw()
#     global filepath
#     filepath = list(root.tk.splitlist(filedialog.askopenfilenames(title="PyFry - Choose Image")))
#     print("picture location = ",filepath)


def irisCoords(eye):
    # Finding the center point of the eye using the average outer extremes average of the eyes
    mid = (eye[0] + eye[3])/2
    mid = (int(mid[0]), int(mid[1]))
    return mid


def generateHue(img):
    # Generating and increasing prominency of red band of the image
    img = img.convert('RGB')
    red = img.split()[0]  # (R,G,B)
    red = ImageEnhance.Contrast(red).enhance(2.0)
    red = ImageEnhance.Brightness(red).enhance(1.5)
    red = ImageOps.colorize(red, Colors.RED, Colors.YELLOW)
    img = Image.blend(img, red, 0.77)
    # Keeping a 100% sharpness value for now, But would probably be better with a higher sharpness value
    img = ImageEnhance.Sharpness(img).enhance(150)
    return img


def crushAndBack(img):
    img = img.convert('RGB')
    w, h = img.width, img.height
    img = img.resize((int(w ** .95), int(h ** .95)), resample=Image.LANCZOS)
    img = img.resize((int(w ** .90), int(h ** .90)), resample=Image.BILINEAR)
    img = img.resize((int(w ** .90), int(h ** .90)), resample=Image.BICUBIC)
    img = img.resize((w, h), resample=Image.BICUBIC)
    return img


def addFlare(img):
    ''' Initialising dlib for frontal facial features '''
    flare = Image.open('flare.png')
    detect = dlib.get_frontal_face_detector()
    face_landmarks_file = "assets/shape_predictor_68_face_landmarks.dat"
    face_landmarks_file = os.path.abspath(face_landmarks_file)
    predict = dlib.shape_predictor(face_landmarks_file)

    (lS, lE) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rS, rE) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

    imgCV = cv2.imread('temp.jpg')
    # imgCV = cv2.imread('test2.jpg')

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

    rn = (rightEye[4][0]-rightEye[0][0])*3
    ln = (leftEye[4][0]-leftEye[0][0])*3

    rec0 = (leftEye[1][0]-ln, leftEye[1][1]-ln)
    rec1 = (leftEye[4][0]+ln, leftEye[4][1]+ln)

    rec2 = (rightEye[1][0]-rn, rightEye[1][1]-rn)
    rec3 = (rightEye[4][0]+rn, rightEye[4][1]+rn)

    print("Area for left eye", rec0, rec1)
    print("Area for right eye", rec2, rec3)

    """ Area Assignment for left eye and right eye"""
    areaLeft = (rec0[0], rec0[1], rec1[0], rec1[1])
    areaRight = (rec2[0], rec2[1], rec3[0], rec3[1])

    """ Resizing the flare image to fit the area"""
    flareLeft = flare.resize((rec1[0]-rec0[0], rec1[1]-rec0[1]))
    flareRight = flare.resize((rec3[0]-rec2[0], rec3[1]-rec2[1]))

    """Pasting the flare image on the area.
       Third parameter is an alpha channel that provides transparency for the png"""
    img.paste(flareLeft, areaLeft, flareLeft)
    img.paste(flareRight, areaRight, flareRight)
    return img


def pyfry():
    # grab a reference to the image panels
    global panelA, panelB
    # open a file chooser dialog and allow the user to select an input
    # image
    img_path = filedialog.askopenfilename()
    print(img_path)

    # ensure a file path was selected
    if len(img_path) > 0:
        img = Image.open(img_path)
        img = crushAndBack(img)
        img = generateHue(img)
        img.save('temp.jpg')
        img = addFlare(img)

        # img.show()
        # img.save('output2.jpg')
        filename = os.path.splitext(os.path.basename(img_path))[0]
        img.save('%s_output.jpg' % filename)
        print("output saved as %s_output.jpg" % filename)

        original_image = cv2.imread(img_path)
        deep_fried_img_path = '%s_output.jpg' % filename
        deep_fried_img = cv2.imread(deep_fried_img_path)
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        deep_fried_img = cv2.cvtColor(deep_fried_img, cv2.COLOR_BGR2RGB)

        # convert the images to PIL format...
        original_image = Image.fromarray(original_image)
        deep_fried_img = Image.fromarray(deep_fried_img)

        # ...and then to ImageTk format
        original_image = ImageTk.PhotoImage(original_image)
        deep_fried_img = ImageTk.PhotoImage(deep_fried_img)

        # if the panels are None, initialize them
        if panelA is None or panelB is None:
            # the first panel will store our original image
            panelA = tk.Label(image=original_image)
            panelA.image = original_image
            panelA.pack(side="left", padx=10, pady=10)

            # while the second panel will store the edge map
            panelB = tk.Label(image=deep_fried_img)
            panelB.image = deep_fried_img
            panelB.pack(side="right", padx=10, pady=10)

        # otherwise, update the image panels
        else:
            # update the pannels
            panelA.configure(image=original_image)
            panelB.configure(image=deep_fried_img)
            panelA.image = original_image
            panelB.image = deep_fried_img


# userInput()
# initialize the window toolkit along with the two image panels
root = tk.Tk()
root.title("Welcome to PyFry")

# root.geometry("600x100")
panelA = None
panelB = None

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn = tk.Button(root, text="click here to select new image to deep fry!!!", command=pyfry)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# kick off the GUI
root.mainloop()
        
   