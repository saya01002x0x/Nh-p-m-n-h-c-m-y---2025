import numpy as np
import pandas as pd
from pathlib import Path
import os
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. Load Model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'age_model.keras')
if not os.path.exists(model_path):
    print(f"❌ Model not found at: {model_path}")
    print("Please train the model first or place 'age_model.keras' in the 'models' folder.")
    exit()

print(f"🔄 Loading model from: {model_path}...")
try:
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!\n")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# 2. Set Test Data Directory (relative path from Source_code)
test_base_dir = Path(os.path.dirname(__file__)) / 'dataset' / 'age_prediction_up' / 'test'

if not test_base_dir.exists():
    print(f"❌ Test directory not found: {test_base_dir}")
    print("Please ensure the dataset is in the correct location.")
    exit()

print(f"📂 Test directory: {test_base_dir}\n")

# 3. Scan all age folders (001, 002, 003, ...)
age_folders = sorted([f for f in test_base_dir.iterdir() if f.is_dir()])

if not age_folders:
    print("❌ No age folders found in test directory.")
    exit()

print(f"Found {len(age_folders)} age folders to test.\n")

# 4. Prepare data structures
all_predictions = []
all_true_ages = []
all_filepaths = []

# Create ImageDataGenerator
test_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# 5. Process each age folder
print("="*60)
print("TESTING EACH AGE GROUP")
print("="*60)

for age_folder in age_folders:
    folder_name = age_folder.name
    
    # Extract true age from folder name (e.g., "025" -> 25)
    try:
        true_age = int(folder_name)
    except ValueError:
        print(f"⚠️ Skipping folder '{folder_name}' (not a valid age number)")
        continue
    
    # Get all images in this folder
    image_files = list(age_folder.glob('*.jpg')) + list(age_folder.glob('*.png')) + list(age_folder.glob('*.jpeg'))
    
    if not image_files:
        print(f"⚠️ Age {true_age:3d}: No images found, skipping.")
        continue
    
    # Create dataframe for this age group
    filepaths = pd.Series([str(f) for f in image_files], name='Filepath')
    ages = pd.Series([true_age] * len(filepaths), name='Age')
    df = pd.concat([filepaths, ages], axis=1)
    
    # Create data generator
    test_images = test_generator.flow_from_dataframe(
        dataframe=df,
        x_col='Filepath',
        y_col='Age',
        target_size=(120, 120),
        color_mode='rgb',
        class_mode='raw',
        batch_size=32,
        shuffle=False
    )
    
    # Predict
    predictions = model.predict(test_images, verbose=0)
    predicted_ages = np.atleast_1d(np.squeeze(predictions))  # Ensure it's always an array
    
    # Calculate metrics for this age group
    mae = mean_absolute_error([true_age] * len(predicted_ages), predicted_ages)
    avg_pred = np.mean(predicted_ages)
    
    print(f"Age {true_age:3d}: {len(image_files):4d} images | Avg Pred: {avg_pred:5.1f} | MAE: {mae:5.2f}")
    
    # Store results
    all_predictions.extend(predicted_ages)
    all_true_ages.extend([true_age] * len(predicted_ages))
    all_filepaths.extend([str(f) for f in image_files])

# 6. Overall Results
print("\n" + "="*60)
print("OVERALL RESULTS")
print("="*60)

all_predictions = np.array(all_predictions)
all_true_ages = np.array(all_true_ages)

overall_mae = mean_absolute_error(all_true_ages, all_predictions)
overall_rmse = np.sqrt(mean_squared_error(all_true_ages, all_predictions))
overall_r2 = r2_score(all_true_ages, all_predictions)

print(f"Total Images Tested: {len(all_predictions)}")
print(f"MAE (Mean Absolute Error):  {overall_mae:.2f} years")
print(f"RMSE (Root Mean Squared Error): {overall_rmse:.2f} years")
print(f"R² Score: {overall_r2:.4f}")
print("="*60)

# 7. Visualization: Aggregated Performance Plot (hides dataset size)
# Group predictions by age bins to show trend without revealing exact data points
age_bins = np.arange(0, 101, 5)  # Bins: 0-5, 5-10, ..., 95-100
bin_centers = []
bin_avg_predictions = []
bin_std_predictions = []

for i in range(len(age_bins) - 1):
    start, end = age_bins[i], age_bins[i+1]
    mask = (all_true_ages >= start) & (all_true_ages < end)
    
    if mask.sum() > 0:
        bin_centers.append((start + end) / 2)
        bin_avg_predictions.append(np.mean(all_predictions[mask]))
        bin_std_predictions.append(np.std(all_predictions[mask]))

bin_centers = np.array(bin_centers)
bin_avg_predictions = np.array(bin_avg_predictions)
bin_std_predictions = np.array(bin_std_predictions)

plt.figure(figsize=(10, 8))

# Plot average prediction with error bars
plt.errorbar(bin_centers, bin_avg_predictions, yerr=bin_std_predictions, 
             fmt='o-', linewidth=2, markersize=8, capsize=5, 
             color='steelblue', ecolor='lightblue', label='Model Prediction (Avg ± Std)')

# Perfect prediction line
plt.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')

plt.xlabel('True Age Group (Center)', fontsize=13, fontweight='bold')
plt.ylabel('Predicted Age', fontsize=13, fontweight='bold')
plt.title(f'Age Prediction Performance by Age Group\nMAE: {overall_mae:.2f} | RMSE: {overall_rmse:.2f}', 
          fontsize=15, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.tight_layout()

output_plot = os.path.join(os.path.dirname(__file__), 'test_results.png')
plt.savefig(output_plot, dpi=150, bbox_inches='tight')
print(f"\n📊 Results plot saved to: {output_plot}")
plt.close()

print("\n✅ Testing completed!")