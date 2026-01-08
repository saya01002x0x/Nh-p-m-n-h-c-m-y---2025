# ============================================================================
# GOOGLE COLAB TRAINING SCRIPT - Age Prediction Model
# Copy toàn bộ code này vào 1 cell trong Colab
# ============================================================================

# 1. INSTALL & IMPORT
!pip install -q scikit-learn pandas matplotlib tensorflow

import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 2. MOUNT GOOGLE DRIVE
from google.colab import drive
drive.mount('/content/drive')

print("✅ Drive mounted successfully!")

# 3. LOAD DATASET FROM DRIVE
# IMPORTANT: Thay đổi đường dẫn này theo vị trí dataset của bạn trong Drive
image_dir = Path('/content/drive/MyDrive/age_prediction_dataset/train')

print(f"Loading dataset from: {image_dir}")
filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name='Filepath').astype(str)
ages = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='Age').astype(int)
images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)

print(f"✅ Loaded {len(images)} images")

# 4. DATA BALANCING
print("\n=== Starting Data Balancing ===")
age_ranges = [
    (1, 10, 8000), (11, 20, 8000), (21, 30, 8000), (31, 40, 8000), (41, 50, 8000),
    (51, 60, 8000), (61, 70, 8000), (71, 80, 8000), (81, 90, 8000), (91, 100, 8000),
]

balanced_dfs = []
for start_age, end_age, target_count in age_ranges:
    age_range_data = images[(images['Age'] >= start_age) & (images['Age'] <= end_age)]
    current_count = len(age_range_data)
    
    if current_count == 0:
        continue
    
    if current_count < target_count:
        n_repeats = target_count // current_count
        n_additional = target_count % current_count
        upsampled_parts = [age_range_data] * n_repeats
        if n_additional > 0:
            upsampled_parts.append(age_range_data.sample(n=n_additional, replace=True, random_state=42))
        balanced_data = pd.concat(upsampled_parts, ignore_index=True)
        print(f"Ages {start_age:3d}-{end_age:3d}: Upsampled {current_count:6,} → {len(balanced_data):6,}")
    else:
        balanced_data = age_range_data.sample(n=target_count, random_state=42)
        print(f"Ages {start_age:3d}-{end_age:3d}: Downsampled {current_count:6,} → {len(balanced_data):6,}")
    
    balanced_dfs.append(balanced_data)

images = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1.0, random_state=1).reset_index(drop=True)
print(f"\n✅ Balanced dataset: {len(images)} images")

# 5. TRAIN-TEST SPLIT
train_df, test_df = train_test_split(images, train_size=0.7, shuffle=True, random_state=1)
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

# 6. CREATE DATA GENERATORS
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# TĂNG BATCH SIZE cho GPU (256 thay vì 64)
BATCH_SIZE = 256

train_images = train_generator.flow_from_dataframe(
    dataframe=train_df, x_col='Filepath', y_col='Age',
    target_size=(120, 120), batch_size=BATCH_SIZE,
    class_mode='raw', subset='training', shuffle=True, seed=42
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df, x_col='Filepath', y_col='Age',
    target_size=(120, 120), batch_size=BATCH_SIZE,
    class_mode='raw', subset='validation', shuffle=True, seed=42
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df, x_col='Filepath', y_col='Age',
    target_size=(120, 120), batch_size=BATCH_SIZE,
    class_mode='raw', shuffle=False
)

# 7. BUILD MODEL (Improved with BatchNorm & Dropout)
inputs = tf.keras.Input(shape=(120, 120, 3))

# Conv Block 1
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D()(x)

# Conv Block 2
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D()(x)

# Flatten & Regularization
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

# Dense Layers
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# 8. COMPILE
model.compile(optimizer='adam', loss='mse')
print("\n✅ Model compiled")

# 9. TRAIN WITH EARLY STOPPING
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

print("\n🚀 Starting training...")
history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=20,
    callbacks=[early_stop],
    verbose=1
)

# 10. EVALUATE
print("\n📊 Evaluating on test set...")
predicted_ages = np.squeeze(model.predict(test_images))
true_ages = test_images.labels
r2 = r2_score(true_ages, predicted_ages)
final_val_loss = history.history['val_loss'][-1]

print(f"\n{'='*60}")
print(f"✅ TRAINING COMPLETED!")
print(f"{'='*60}")
print(f"Test R² Score: {r2:.5f}")
print(f"Final Val Loss: {final_val_loss:.5f}")
print(f"{'='*60}\n")

# 11. SAVE MODEL TO DRIVE
model_save_path = '/content/drive/MyDrive/age_prediction_model'
model.save(model_save_path)
print(f"✅ Model saved to: {model_save_path}")

# 12. PLOT TRAINING HISTORY
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(true_ages[:1000], predicted_ages[:1000], alpha=0.3)
plt.plot([0, 100], [0, 100], 'r--', label='Perfect Prediction')
plt.title(f'Predictions vs True Age (R²={r2:.3f})')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/content/drive/MyDrive/training_results.png', dpi=150)
plt.show()

print("\n✅ ALL DONE! Check your Google Drive for:")
print("  📁 Model: age_prediction_model/")
print("  📊 Results: training_results.png")
