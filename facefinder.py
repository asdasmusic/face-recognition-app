#import required libraries
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
#import OpenCV library
import cv2
#import matplotlib library
import matplotlib.pyplot as plt
#importing time library for speed comparisons of both classifiers
import time

#Write the UI
ui = Tk()
ui.title("Face Finder v1.0")
ui.wm_iconbitmap('icon.ico')
ui.configure(bg='#770933')
def openfile():
    global open
    open = filedialog.askopenfilename(filetypes=(("JPG Images", "*.jpg"),
                                                 ("JPEG Images", "*.jpeg"),
                                                 ("PNG Images", "*.png"),
                                                 ("BMP Images", "*.bmp"),
                                                 ("All Files", "*.*")),
                                      title="Open An Image")
    if len(open)>0:
      ui.destroy()

sty = ttk.Style()
sty.configure("Bold.TButton", font=('System', 20), background='lime')
icon = PhotoImage(file="img.png")
txt = Label(text="©2020 by Anshuman",fg='white', bg='#b34700')
txt.pack(side="bottom", anchor=E)
btn = ttk.Button(ui, text="Select Your Photo",
                 width=20,
                 style = "Bold.TButton",
                 image=icon,
                 compound = TOP,
                 command=openfile)
btn.pack(side="bottom", fill="both", expand="yes", padx="15", pady="15")
# kick off the UI
ui.mainloop()

# convert BGR to RGB
def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_faces(f_cascade, colored_img, scaleFactor=1.1):
    # just making a copy of image passed, so that passed image is not changed
    img_copy = colored_img.copy()

    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);

    # go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_copy

def process(input):
    #load cascade classifier training file for haarcascade
    haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    #load cascade classifier training file for lbpcascade
    lbp_face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
    #load image
    image = cv2.imread(input)

    #------------HAAR-----------
    #note time before detection
    t1 = time.time()
    #call our function to detect faces
    haar_detected_img = detect_faces(haar_face_cascade, image)
    #note time after detection
    t2 = time.time()
    #calculate time difference
    dt1 = t2 - t1
    #print the time differene

    #------------LBP-----------
    #note time before detection
    t1 = time.time()
    lbp_detected_img = detect_faces(lbp_face_cascade, image)
    #note time after detection
    t2 = time.time()
    #calculate time difference
    dt2 = t2 - t1
    #print the time differene

    #----------Let's do some fancy drawing-------------
    #create a figure of 2 plots (one for Haar and one for LBP)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    f.canvas.set_window_title('Face Finder v1.0')
    f.patch.set_facecolor('#273746')
    plt.rcParams["font.family"] = 'Arial'
    f.suptitle('Face Recognition Results', fontsize=18, color='red', weight='bold')
    #show Haar image
    ax1.set_title('Haar Detection time: ' + str(round(dt1, 3)) + ' secs',fontsize=12,color='white')
    ax1.tick_params(axis='x', colors='#00e6e6')
    ax1.tick_params(axis='y', colors='#00e6e6')
    ax1.imshow(convertToRGB(haar_detected_img))
    #show LBP image
    ax2.set_title('LBP Detection time: ' + str(round(dt2, 3)) + ' secs',fontsize=12,color='white')
    ax2.tick_params(axis='x', colors='#00e6e6')
    ax2.tick_params(axis='y', colors='#00e6e6')
    ax2.imshow(convertToRGB(lbp_detected_img))
    #show output
    plt.show()

#call load function
process(open)
