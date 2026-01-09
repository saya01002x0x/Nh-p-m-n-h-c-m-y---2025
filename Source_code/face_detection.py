#Importing cv2
import cv2
import numpy as np
from tensorflow import keras

# Load Model 
import os
model_path = os.path.join(os.path.dirname(__file__), 'models', 'age_model.keras')
if not os.path.exists(model_path):
    print(f"Warning: Model not found at {model_path}. Please train the model first using train.py")

# Load Keras model (.keras format)
print(f"Loading model from: {model_path}")
model = keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

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
    
    for (x, y, w, h) in faces:
        # Extract face region
        img_cat = frame[y:y+h, x:x+w]
        
        if img_cat.size == 0:
            continue
            
        # Resize for model input
        img_age = cv2.resize(img_cat, (120, 120))
        
        # Ensure 3 channels (RGB)
        if len(img_age.shape) == 2:  # Grayscale
            img_age = cv2.cvtColor(img_age, cv2.COLOR_GRAY2RGB)
        elif img_age.shape[2] == 4:  # RGBA
            img_age = cv2.cvtColor(img_age, cv2.COLOR_RGBA2RGB)
        elif img_age.shape[2] == 3:  # BGR -> RGB
            img_age = cv2.cvtColor(img_age, cv2.COLOR_BGR2RGB)
            
        img_age = np.expand_dims(img_age, axis=0)
        img_age = img_age.astype('float32')
        
        # Predict age
        img_predict = test_generator.flow(img_age, batch_size=1, shuffle=False)
        output_predict = int(np.squeeze(model.predict(img_predict, verbose=0)))
        
        # Age-based color: Green if <30, Red if >=30
        if output_predict < 30:
            color = (0, 255, 0)  # Green
        else:
            color = (0, 0, 255)  # Red
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Display age
        age_text = f"Age: {output_predict}"
        font_scale = max(0.6, w / 200)
        cv2.putText(frame, age_text, (x, y-10), 
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, color, 2)
        
    return frame