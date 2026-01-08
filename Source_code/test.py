import numpy as np
import pandas as pd
from pathlib import Path
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow import keras
from sklearn.metrics import r2_score, mean_squared_error

# Setup Tkinter for file dialog
root = tk.Tk()
root.withdraw()  # Hide the main window

# 1. Load Model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'age_model.keras')
if not os.path.exists(model_path):
    print(f"❌ Model not found at: {model_path}")
    print("Please train the model first or place 'age_model.keras' in the 'models' folder.")
    exit()

print(f"🔄 Loading model from: {model_path}...")
try:
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# 2. Select Test Data Directory
print("\n👉 Please select a folder containing images to test (e.g., a folder named '25' containing images of 25-year-olds).")
test_dir = filedialog.askdirectory(title="Select Folder with Test Images")

if not test_dir:
    print("❌ No folder selected. Exiting.")
    exit()

print(f"📂 Selected folder: {test_dir}")

# 3. Load Images
image_dir = Path(test_dir)
filepaths = pd.Series(list(image_dir.glob(r'*.jpg')) + list(image_dir.glob(r'*.png')) + list(image_dir.glob(r'*.jpeg')), name='Filepath').astype(str)

if filepaths.empty:
    print("❌ No images found in selected folder (looking for .jpg, .png, .jpeg).")
    exit()

# Try to extract true age from folder name (assuming folder name is the age, e.g., '25')
try:
    folder_name = os.path.basename(test_dir)
    true_age_val = int(folder_name)
    print(f"ℹ️ Assuming true age is {true_age_val} based on folder name.")
    ages = pd.Series([true_age_val] * len(filepaths), name='Age')
except ValueError:
    print("⚠️ Folder name is not an integer. Cannot determine true age from folder name.")
    print("   Setting true age to 0 (metrics like RMSE/R2 will be invalid, but predictions will work).")
    ages = pd.Series([0] * len(filepaths), name='Age')

images_df = pd.concat([filepaths, ages], axis=1)

# 4. Create Data Generator
test_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_images = test_generator.flow_from_dataframe(
    dataframe=images_df,
    x_col='Filepath',
    y_col='Age',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=False 
)

# 5. Predict
print("\n🔮 Predicting...")
predictions = model.predict(test_images)
predicted_ages = np.squeeze(predictions)

# 6. Show Results
print("\n" + "="*40)
print("             RESULTS             ")
print("="*40)
print(f"Input Folder: {test_dir}")
print(f"Number of Images: {len(predicted_ages)}")
print("-" * 40)
print(f"Average Predicted Age: {np.mean(predicted_ages):.2f}")
print(f"Min Predicted Age:     {np.min(predicted_ages):.2f}")
print(f"Max Predicted Age:     {np.max(predicted_ages):.2f}")
print("-" * 40)

# 7. Calculate Metrics (if true age is valid)
if true_age_val > 0:
    true_ages = images_df['Age'].values
    
    mse = mean_squared_error(true_ages, predicted_ages)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_ages - predicted_ages))
    
    print(f"True Age (from folder): {true_age_val}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"MAE (Mean Absolute Error):      {mae:.2f}")
    
    # R2 might not be meaningful if variance of true_ages is 0 (all same age), but printing anyway
    # r2 = r2_score(true_ages, predicted_ages) 
    # print(f"R2 Score: {r2:.5f}")

print("="*40)

# Optional: List individual predictions
# print("\nIndividual Predictions:")
# for i, age in enumerate(predicted_ages):
#     print(f"Image {i+1}: {age:.1f}")