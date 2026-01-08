import cv2
import os
import tkinter as tk
from tkinter import *
from PIL import Image,ImageTk
from datetime import datetime
from tkinter import messagebox, filedialog
from face_detection import detect


# Defining CreateWidgets() function to create necessary tkinter widgets
# Defining CreateWidgets() function to create necessary tkinter widgets
def createwidgets():
    root.feedlabel = Label(root, bg="steelblue", fg="white", text="WEBCAM FEED", font=('Comic Sans MS',20))
    root.feedlabel.grid(row=1, column=1, padx=10, pady=10, columnspan=2)

    root.cameraLabel = Label(root, bg="steelblue", borderwidth=3, relief="groove")
    root.cameraLabel.grid(row=2, column=1, padx=10, pady=10, columnspan=2)

    # Auto-save info label (removed browse functionality)
    auto_save_info = Label(root, text="Images auto-saved to: imgs/", bg="steelblue", fg="white", font=('Comic Sans MS', 10))
    auto_save_info.grid(row=3, column=1, padx=10, pady=10, columnspan=2)

    root.captureBTN = Button(root, text="CAPTURE", command=Capture, bg="LIGHTBLUE", font=('Comic Sans MS',15), width=20)
    root.captureBTN.grid(row=4, column=1, padx=10, pady=10)

    root.CAMBTN = Button(root, text="STOP CAMERA", command=StopCAM, bg="LIGHTBLUE", font=('Comic Sans MS',15), width=13)
    root.CAMBTN.grid(row=4, column=2)

    root.previewlabel = Label(root, bg="steelblue", fg="white", text="IMAGE PREVIEW", font=('Comic Sans MS',20))
    root.previewlabel.grid(row=1, column=4, padx=10, pady=10, columnspan=2)

    root.imageLabel = Label(root, bg="steelblue", borderwidth=3, relief="groove")
    root.imageLabel.grid(row=2, column=4, padx=10, pady=10, columnspan=2)

    root.openImageEntry = Entry(root, width=55, textvariable=imagePath)
    root.openImageEntry.grid(row=3, column=4, padx=10, pady=10)

    root.openImageButton = Button(root, width=10, text="BROWSE", command=imageBrowse)
    root.openImageButton.grid(row=3, column=5, padx=10, pady=10)
    
    root.startPredict = Button(root, text="START PREDICT", command=StartPredict, bg="LIGHTBLUE", font=('Comic Sans MS',15), width=20)
    root.startPredict.grid(row=4, column=5, padx=10, pady=10)

    # Calling ShowFeed() function
    ShowFeed()

# Defining ShowFeed() function to display webcam feed in the cameraLabel
def ShowFeed():
    # Initialize FPS tracking (only once)
    if not hasattr(root, 'fps_start_time'):
        root.fps_start_time = datetime.now()
        root.fps_frame_count = 0
        root.fps_display = 0.0
    
    # Capturing frame by frame
    ret, frame = root.cap.read()

    if ret:
        # FPS Calculation
        root.fps_frame_count += 1
        elapsed_time = (datetime.now() - root.fps_start_time).total_seconds()
        
        if elapsed_time > 1.0:  # Update FPS every second
            root.fps_display = root.fps_frame_count / elapsed_time
            root.fps_frame_count = 0
            root.fps_start_time = datetime.now()
        
        # Flipping the frame vertically
        frame = cv2.flip(frame, 1)

        # Display FPS at top-left corner
        fps_text = f"FPS: {root.fps_display:.1f}"
        cv2.putText(frame, fps_text, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
        
        # Displaying date and time on the feed
        cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (10, 55), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1)

        # Changing the frame color from BGR to RGB
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        
        canvas = detect(cv2image, frame_copy)
        # Creating an image memory from the above frame exporting array interface
        videoImg = Image.fromarray(canvas)

        # Creating object of PhotoImage() class to display the frame
        imgtk = ImageTk.PhotoImage(image = videoImg)

        # Configuring the label to display the frame
        root.cameraLabel.configure(image=imgtk)

        # Keeping a reference
        root.cameraLabel.imgtk = imgtk

        # Calling the function after 10 milliseconds
        root.cameraLabel.after(10, ShowFeed)
    else:
        # Configuring the label to display the frame
        root.cameraLabel.configure(image='')
    

# Auto-create imgs folder if not exists
imgs_folder = os.path.join(os.path.dirname(__file__), 'imgs')
if not os.path.exists(imgs_folder):
    os.makedirs(imgs_folder)

def imageBrowse():
    
    # Presenting user with a pop-up for directory selection. initialdir argument is optional
    # Retrieving the user-input destination directory and storing it in destinationDirectory
    # Setting the initialdir argument is optional. SET IT TO YOUR DIRECTORY PATH
    root.openDirectory = filedialog.askopenfilename(initialdir="YOUR DIRECTORY PATH")
    
    # Displaying the directory in the directory textbox
    imagePath.set(root.openDirectory)
    print('test')
    print(imagePath)
    # Opening the saved image using the open() of Image class which takes the saved image as the argument
    imageView = Image.open(root.openDirectory)
    
    # Resizing the image using Image.resize()
    imageResize = imageView.resize((640, 480), Image.Resampling.LANCZOS)

    # Creating object of PhotoImage() class to display the frame
    imageDisplay = ImageTk.PhotoImage(imageResize)

    # Configuring the label to display the frame
    root.imageLabel.config(image=imageDisplay)

    # Keeping a reference
    root.imageLabel.photo = imageDisplay

# Defining Capture() to capture and save the image and display the image in the imageLabel
def Capture():
    # Storing the date in the mentioned format in the image_name variable
    image_name = datetime.now().strftime('%d-%m-%Y %H-%M-%S')

    # Auto-save to imgs/ folder
    image_path = os.path.join(os.path.dirname(__file__), 'imgs')
    
    # Ensure imgs folder exists
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # Concatenating the image_path with image_name and with .jpg extension and saving it in imgName variable
    imgName = os.path.join(image_path, image_name + ".jpg")

    # Capturing the frame
    ret, frame = root.cap.read()

    # Displaying date and time on the frame
    cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (430,460), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255))

    # Writing the image with the captured frame. Function returns a Boolean Value which is stored in success variable
    success = cv2.imwrite(imgName, frame)

    # Opening the saved image using the open() of Image class which takes the saved image as the argument
    saved_image = Image.open(imgName)

    # Creating object of PhotoImage() class to display the frame
    saved_image = ImageTk.PhotoImage(saved_image)

    # Configuring the label to display the frame
    root.imageLabel.config(image=saved_image)

    # Keeping a reference
    root.imageLabel.photo = saved_image

    # Displaying messagebox
    if success :
        messagebox.showinfo("SUCCESS", "IMAGE CAPTURED AND SAVED IN " + imgName)


# Defining StopCAM() to stop WEBCAM Preview
def StopCAM():
    # Stopping the camera using release() method of cv2.VideoCapture()
    root.cap.release()

    # Configuring the CAMBTN to display accordingly
    root.CAMBTN.config(text="START CAMERA", command=StartCAM)

    # Displaying text message in the camera label
    root.cameraLabel.config(text="OFF CAM", font=('Comic Sans MS',70))

def StartCAM():
    # Creating object of class VideoCapture with webcam index
    root.cap = cv2.VideoCapture(0)

    # Setting width and height
    width_1, height_1 = 640, 480
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_1)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_1)

    # Configuring the CAMBTN to display accordingly
    root.CAMBTN.config(text="STOP CAMERA", command=StopCAM)

    # Removing text message from the camera label
    root.cameraLabel.config(text="")

    # Calling the ShowFeed() Function
    ShowFeed()

def StartPredict():
    # Check if an image has been selected
    if not hasattr(root, 'openDirectory') or not root.openDirectory:
        messagebox.showerror("ERROR", "NO IMAGE SELECTED! Please browse and select an image first.")
        return
    
    image = cv2.imread(root.openDirectory)
    if image is None:
        messagebox.showerror("ERROR", "Failed to load image. Please select a valid image file.")
        return
        
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    canvas = detect(rgb, gray)

    predict_image = Image.fromarray(canvas)
    predict_image = predict_image.resize((250, 250), Image.Resampling.LANCZOS)
    # Creating object of PhotoImage() class to display the frame
    predict_image = ImageTk.PhotoImage(predict_image)

    # Configuring the label to display the frame
    root.imageLabel.config(image=predict_image)

    # Keeping a reference
    root.imageLabel.photo = predict_image








# Creating object of tk class
root = tk.Tk()

# Creating object of class VideoCapture with webcam index
root.cap = cv2.VideoCapture(0)

# Setting width and height
width, height = 640, 480
root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Setting the title, window size, background color and disabling the resizing property
root.title("Pycam")
root.geometry("1340x700")
root.resizable(True, True)
root.configure(background = "sky blue")

# Creating tkinter variables
imagePath = StringVar()

createwidgets()
root.mainloop()
