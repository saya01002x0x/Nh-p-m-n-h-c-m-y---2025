import cv2
import os
import tkinter as tk
from tkinter import *
from PIL import Image,ImageTk
from datetime import datetime
from tkinter import messagebox, filedialog
from face_detection import detect


# Defining CreateWidgets() function to create necessary tkinter widgets
def createwidgets():
    # Title Bar with modern design
    title_frame = Frame(root, bg="#2c5f8d", height=100)
    title_frame.grid(row=0, column=0, columnspan=4, sticky="ew", padx=0, pady=0)
    
    # Icon and title container
    title_container = Frame(title_frame, bg="#2c5f8d")
    title_container.pack(expand=True)
    
    title_label = Label(title_container, text="AGE DETECTION SYSTEM", 
                        bg="#2c5f8d", fg="white", 
                        font=('Segoe UI', 32, 'bold'))
    title_label.pack(pady=(15, 5))
    
    subtitle_label = Label(title_container, text="Real-time Face & Age Recognition", 
                          bg="#2c5f8d", fg="#c8e0f4", 
                          font=('Segoe UI', 12))
    subtitle_label.pack(pady=(0, 15))
    
    # Left Panel - Camera Feed with rounded corners
    left_panel = Frame(root, bg="#4a7ba7", relief="raised", borderwidth=3)
    left_panel.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
    
    camera_header = Frame(left_panel, bg="#4a7ba7", height=50)
    camera_header.pack(fill="x", padx=15, pady=(15, 10))
    
    root.feedlabel = Label(camera_header, bg="#4a7ba7", fg="white", 
                           text="───  CAMERA FEED  ───", 
                           font=('Segoe UI', 14, 'bold'))
    root.feedlabel.pack(anchor="center")

    root.cameraLabel = Label(left_panel, bg="#2c3e50", borderwidth=0)
    root.cameraLabel.pack(padx=15, pady=10)
    
    # Right Panel - Image Preview with rounded corners
    right_panel = Frame(root, bg="#5a8fb7", relief="raised", borderwidth=3)
    right_panel.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")
    
    preview_header = Frame(right_panel, bg="#5a8fb7", height=50)
    preview_header.pack(fill="x", padx=15, pady=(15, 10))
    
    root.previewlabel = Label(preview_header, bg="#5a8fb7", fg="white", 
                             text="───  IMAGE PREVIEW  ───", 
                             font=('Segoe UI', 14, 'bold'))
    root.previewlabel.pack(anchor="center")

    # Create a container for the image with dashed border
    image_container = Frame(right_panel, bg="#c8ddef", width=660, height=500)
    image_container.pack(padx=15, pady=10)
    image_container.pack_propagate(False)
    
    # Dashed border frame
    dashed_frame = Frame(image_container, bg="#c8ddef", 
                        highlightbackground="#8ab4d4", 
                        highlightthickness=2, 
                        highlightcolor="#8ab4d4",
                        relief="flat")
    dashed_frame.place(relx=0.5, rely=0.5, anchor="center", 
                      width=640, height=480)
    
    root.imageLabel = Label(dashed_frame, bg="#c8ddef", 
                           text="No image selected", 
                           fg="#7a9db8", 
                           font=('Segoe UI', 16, 'italic'),
                           borderwidth=0)
    root.imageLabel.place(relx=0.5, rely=0.5, anchor="center")

    # Browse button centered below preview
    root.openImageButton = Button(right_panel, width=18, text="BROWSE", 
                                 command=imageBrowse,
                                 bg="#7db3da", fg="white",
                                 font=('Segoe UI', 10, 'bold'),
                                 relief="raised", borderwidth=1,
                                 activebackground="#6ba3d0",
                                 cursor="hand2", pady=8)
    root.openImageButton.pack(pady=(5, 15))
    
    # Bottom buttons frame
    button_frame = Frame(root, bg="#d4e8f7")
    button_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=40, pady=20)
    
    # Configure grid for button frame
    button_frame.grid_columnconfigure(0, weight=1)
    button_frame.grid_columnconfigure(1, weight=1)
    button_frame.grid_columnconfigure(2, weight=1)
    button_frame.grid_columnconfigure(3, weight=1)
    
    root.captureBTN = Button(button_frame, text="CAPTURE", command=Capture, 
                            bg="#5a8fb7", fg="white", 
                            font=('Segoe UI', 14, 'bold'), 
                            width=18, height=2,
                            relief="raised", borderwidth=3,
                            activebackground="#4a7ba7", 
                            cursor="hand2")
    root.captureBTN.grid(row=0, column=0, padx=8, pady=10)

    root.CAMBTN = Button(button_frame, text="STOP CAMERA", command=StopCAM, 
                        bg="#c85a5a", fg="white", 
                        font=('Segoe UI', 14, 'bold'), 
                        width=18, height=2,
                        relief="raised", borderwidth=3,
                        activebackground="#b04545", 
                        cursor="hand2")
    root.CAMBTN.grid(row=0, column=1, padx=8, pady=10)
    
    browse_btn = Button(button_frame, text="BROWSE", command=imageBrowse,
                       bg="#5a8fb7", fg="white",
                       font=('Segoe UI', 14, 'bold'),
                       width=18, height=2,
                       relief="raised", borderwidth=3,
                       activebackground="#4a7ba7",
                       cursor="hand2")
    browse_btn.grid(row=0, column=2, padx=8, pady=10)
    
    root.startPredict = Button(button_frame, text="START PREDICT", 
                              command=StartPredict, 
                              bg="#4db89a", fg="white", 
                              font=('Segoe UI', 14, 'bold'), 
                              width=18, height=2,
                              relief="raised", borderwidth=3,
                              activebackground="#3ea085",
                              cursor="hand2")
    root.startPredict.grid(row=0, column=3, padx=8, pady=10)

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

        # Display FPS at top-left corner with background
        fps_text = f"FPS: {root.fps_display:.1f}"
        # Add black background for better visibility
        cv2.rectangle(frame, (5, 5), (140, 45), (0, 0, 0), -1)
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Displaying date and time on the feed with background
        time_text = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        cv2.rectangle(frame, (5, 50), (240, 82), (0, 0, 0), -1)
        cv2.putText(frame, time_text, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

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
    
    if root.openDirectory:  # Check if user selected a file
        # Displaying the directory in the directory textbox
        imagePath.set(root.openDirectory)
        
        # Opening the saved image using the open() of Image class which takes the saved image as the argument
        imageView = Image.open(root.openDirectory)
        
        # Resizing the image using Image.resize()
        imageResize = imageView.resize((640, 480), Image.Resampling.LANCZOS)

        # Creating object of PhotoImage() class to display the frame
        imageDisplay = ImageTk.PhotoImage(imageResize)

        # Configuring the label to display the frame
        root.imageLabel.config(image=imageDisplay, text="", bg="#c8ddef")

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
    root.imageLabel.config(image=saved_image, text="", bg="#c8ddef")

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
    root.CAMBTN.config(text="START CAMERA", command=StartCAM,
                      bg="#4db89a", activebackground="#3ea085")

    # Displaying text message in the camera label
    root.cameraLabel.config(text="CAMERA OFF", 
                           font=('Segoe UI', 32, 'bold'),
                           fg="#6ba3d0", bg="#2c3e50")

def StartCAM():
    # Creating object of class VideoCapture with webcam index
    root.cap = cv2.VideoCapture(0)

    # Setting width and height
    width_1, height_1 = 640, 480
    root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_1)
    root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_1)

    # Configuring the CAMBTN to display accordingly
    root.CAMBTN.config(text="STOP CAMERA", command=StopCAM,
                      bg="#c85a5a", activebackground="#b04545")

    # Removing text message from the camera label
    root.cameraLabel.config(text="", bg="#2c3e50")

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
    # Giữ nguyên kích thước như khi browse ảnh (640x480)
    predict_image = predict_image.resize((640, 480), Image.Resampling.LANCZOS)
    # Creating object of PhotoImage() class to display the frame
    predict_image = ImageTk.PhotoImage(predict_image)

    # Configuring the label to display the frame
    root.imageLabel.config(image=predict_image, text="", bg="#c8ddef")

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
root.title("AGE DETECTION SYSTEM")
root.geometry("1400x850")
root.resizable(True, True)
root.configure(background="#d4e8f7")

# Configure grid weights for responsive design
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Creating tkinter variables
imagePath = StringVar()

createwidgets()
root.mainloop()
