#Importing cv2
import cv2
import numpy as np
from tensorflow import keras

# Load Model
import os
model_path = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.exists(model_path):
    print(f"Warning: Model not found at {model_path}. Please check your directories.")

# Keras 3 Loading (TFSMLayer for SavedModel)
try:
    # Try loading as a standard Keras model first (for .keras or .h5)
    model = keras.models.load_model(model_path)
    is_layer = False
except Exception:
    # Fallback to TFSMLayer for legacy SavedModel folders in Keras 3
    print("Loading as TFSMLayer...")
    model = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    is_layer = True

# Create ImageDataGenerator
test_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)
print('OK')

#Loading cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray, frame): 
    # We create a function that takes as input the image in black and white (gray) 
    # and the original image (frame), and that will return the same image with the detector rectangles. 
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    # scaleFactor -- specifying how much the image size is reduced at each image scale
    # minNeighbors -- specifying how many neighbors each candidate rectangle should have
    
    for (x, y, w, h) in faces:  # For each detected face: (faces is the tuple of x,y--point of upper left corner, w-width, h-height)
        # Extract face region
        img_cat = frame[y:y+h, x:x+w]  # Fixed: was x:y, should be y:y+h
        
        if img_cat.size == 0:  # Skip if face region is invalid
            continue
            
        # Resize for model input
        img_age = cv2.resize(img_cat, (120, 120))
        img_age = np.expand_dims(img_age, axis=0)  # Add batch dimension
        img_age = img_age.astype('float32')
        
        # Predict age
        img_predict = test_generator.flow(img_age, batch_size=1, shuffle=False)
        
        if is_layer:
            # TFSMLayer prediction
            batch_data = next(img_predict)
            outputs = model(batch_data)
            # Outputs is a dict, get the first value
            prediction_tensor = list(outputs.values())[0]
            output_predict = int(np.squeeze(prediction_tensor))
        else:
            # Standard Keras model prediction
            output_predict = int(np.squeeze(model.predict(img_predict, verbose=0)))
        
        # Age-based color logic: Green if <30, Red if >=30
        if output_predict < 30:
            color = (0, 255, 0)  # Green (BGR format)
        else:
            color = (0, 0, 255)  # Red (BGR format)
        
        # Draw rectangle with age-based color
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Display age with improved font (HERSHEY_DUPLEX)
        age_text = f"Age: {output_predict}"
        font_scale = max(0.6, w / 200)  # Dynamic font size based on face width
        cv2.putText(frame, age_text, (x, y-10), 
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, color, 2)
        
    return frame  # We return the image with the detector rectangles.  

# def age_predict()