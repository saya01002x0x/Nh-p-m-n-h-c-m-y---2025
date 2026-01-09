#Importing cv2
import cv2
import numpy as np
from tensorflow import keras
from collections import deque

# Load Model (New trained model with val_loss: 168)
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

# Dictionary để lưu lịch sử dự đoán cho mỗi khuôn mặt (smoothing)
# Key: vị trí khuôn mặt (x, y), Value: deque chứa các dự đoán gần nhất
age_history = {}
SMOOTHING_FRAMES = 5  # Số frame để tính trung bình (có thể điều chỉnh: 3-10)

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
        current_prediction = int(np.squeeze(model.predict(img_predict, verbose=0)))
        
        # Tạo key cho khuôn mặt dựa trên vị trí (làm tròn để tránh key khác nhau cho cùng 1 mặt)
        face_key = (round(x / 50) * 50, round(y / 50) * 50)
        
        # Khởi tạo hoặc cập nhật lịch sử dự đoán
        if face_key not in age_history:
            age_history[face_key] = deque(maxlen=SMOOTHING_FRAMES)
        
        age_history[face_key].append(current_prediction)
        
        # Tính tuổi trung bình từ lịch sử
        output_predict = int(np.mean(age_history[face_key]))
        
        # Màu cho khung và chữ (RGBA format vì frame đã được convert sang RGBA)
        rectangle_color = (255, 255, 255, 255)  # Trắng cho khung
        text_color = (255, 255, 0, 255)  # Vàng cho chữ (R=255, G=255, B=0)
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_color, 3)
        
        # Display age
        age_text = f"Age: {output_predict}"
        font_scale = max(0.6, w / 200)
        cv2.putText(frame, age_text, (x, y-10), 
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, 2)
        
    # Dọn dẹp các face_key cũ không còn xuất hiện (optional)
    if len(age_history) > 10:  # Giới hạn số lượng face được lưu
        # Xóa key cũ nhất nếu có quá nhiều
        oldest_key = next(iter(age_history))
        age_history.pop(oldest_key)
        
    return frame