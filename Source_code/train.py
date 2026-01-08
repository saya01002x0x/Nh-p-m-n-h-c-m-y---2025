import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load images
image_dir = Path(os.path.join(os.path.dirname(__file__), 'dataset', 'age_prediction_up', 'train'))
filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name='Filepath').astype(str)
ages = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='Age').astype(int)
images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)

# Save original distribution for comparison
original_distribution = images['Age'].value_counts().sort_index()
print(f"Original dataset size: {len(images)} images")
print(f"Age range: {images['Age'].min()} - {images['Age'].max()}")

# Data Balancing Strategy - Target: 8,000 images per 10-year age group (80k total)
print("\n=== Starting Comprehensive Data Balancing ===")
print("Target: 8,000 images per 10-year age range (80k total)\n")

# Define age ranges and their target counts
age_ranges = [
    (1, 10, 8000),    # Currently: 2,748
    (11, 20, 8000),   # Currently: 12,311
    (21, 30, 8000),   # Currently: 50,693
    (31, 40, 8000),   # Currently: 55,177
    (41, 50, 8000),   # Currently: 34,289
    (51, 60, 8000),   # Currently: 17,007
    (61, 70, 8000),   # Currently: 8,502
    (71, 80, 8000),   # Currently: 3,490
    (81, 90, 8000),   # Currently: 1,224
    (91, 100, 8000),  # Currently: 191
]

balanced_dfs = []

for start_age, end_age, target_count in age_ranges:
    # Get all data for this age range
    age_range_data = images[(images['Age'] >= start_age) & (images['Age'] <= end_age)]
    current_count = len(age_range_data)
    
    if current_count == 0:
        continue
    
    if current_count < target_count:
        # UPSAMPLE: Need more data
        n_repeats = target_count // current_count
        n_additional = target_count % current_count
        
        # Repeat the entire dataset n_repeats times
        upsampled_parts = [age_range_data] * n_repeats
        
        # Add random samples for the remaining
        if n_additional > 0:
            upsampled_parts.append(age_range_data.sample(n=n_additional, replace=True, random_state=42))
        
        balanced_data = pd.concat(upsampled_parts, ignore_index=True)
        print(f"Ages {start_age:3d}-{end_age:3d}: Upsampled   {current_count:6,} → {len(balanced_data):6,} ({len(balanced_data)/current_count:.2f}x)")
        
    else:
        # DOWNSAMPLE: Too much data
        sampling_fraction = target_count / current_count
        balanced_data = age_range_data.sample(n=target_count, random_state=42)
        print(f"Ages {start_age:3d}-{end_age:3d}: Downsampled {current_count:6,} → {len(balanced_data):6,} ({sampling_fraction:.2%})")
    
    balanced_dfs.append(balanced_data)

# Combine all balanced data
images = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1.0, random_state=1).reset_index(drop=True)
balanced_distribution = images['Age'].value_counts().sort_index()

print(f"\nBalanced dataset size: {len(images)} images")
print("=== Data Balancing Completed ===\n")

# Split into train and test sets
train_df, test_df = train_test_split(images, train_size=0.7, shuffle=True, random_state=1)

# Visualization: Before vs After Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Before balancing
ax1.bar(original_distribution.index, original_distribution.values, color='steelblue', alpha=0.7)
ax1.set_title('Phân bố tuổi TRƯỚC khi cân bằng dữ liệu', fontsize=14, fontweight='bold')
ax1.set_xlabel('Tuổi', fontsize=12)
ax1.set_ylabel('Số lượng ảnh', fontsize=12)
ax1.grid(axis='y', alpha=0.3)
ax1.text(0.98, 0.98, f'Tổng: {original_distribution.sum()} ảnh', 
         transform=ax1.transAxes, ha='right', va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# After balancing
ax2.bar(balanced_distribution.index, balanced_distribution.values, color='coral', alpha=0.7)
ax2.set_title('Phân bố tuổi SAU khi cân bằng dữ liệu', fontsize=14, fontweight='bold')
ax2.set_xlabel('Tuổi', fontsize=12)
ax2.set_ylabel('Số lượng ảnh', fontsize=12)
ax2.grid(axis='y', alpha=0.3)
ax2.text(0.98, 0.98, f'Tổng: {balanced_distribution.sum()} ảnh', 
         transform=ax2.transAxes, ha='right', va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('data_balancing_comparison.png', dpi=150, bbox_inches='tight')
print("Đã lưu biểu đồ so sánh tại: data_balancing_comparison.png\n")
plt.close()


# Create ImageDataGenerator
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

# Optimized batch size for 32GB RAM (256 instead of 64)
BATCH_SIZE = 256

# Change images to dataframe
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col='Filepath',
    y_col='Age',
    weight_col=None,
    target_size=(120, 120),
    color_mode='rgb',
    classes=None,
    class_mode='raw',
    batch_size=64,
    shuffle=True,
    seed=42,
    save_to_dir=None,
    save_prefix='',
    save_format='png',
    subset='training',
    interpolation='nearest',
    validate_filenames=True,
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col='Filepath',
    y_col='Age',
    weight_col=None,
    target_size=(120, 120),
    color_mode='rgb',
    classes=None,
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    save_to_dir=None,
    save_prefix='',
    save_format='png',
    subset='validation',
    interpolation='nearest',
    validate_filenames=True,
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col='Filepath',
    y_col='Age',
    weight_col=None,
    target_size=(120, 120),
    color_mode='rgb',
    classes=None,
    class_mode='raw',
    batch_size=BATCH_SIZE,
    shuffle=False,
    save_to_dir=None,
    save_prefix='',
    save_format='png',
    interpolation='nearest',
    validate_filenames=True,
)

# Create Model with BatchNormalization and Dropout
inputs = tf.keras.Input(shape=(120, 120, 3))

# Conv Block 1
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D()(x)

# Conv Block 2 - 64 filters
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D()(x)

# Conv Block 3 - 128 filters (NEW! Deeper model for better accuracy)
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D()(x)

# Flatten & Regularization
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)  # Dropout to reduce overfitting

# Dense Layers (increased capacity)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)

# Output
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='mse'
)

# Callbacks for better training
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Wait 5 epochs before stopping (increased for deeper model)
    restore_best_weights=True,
    verbose=1
)

# NEW! Reduce learning rate when loss plateaus
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,       # Reduce LR by half
    patience=2,       # Wait 2 epochs before reducing
    min_lr=0.00001,   # Don't go below this
    verbose=1
)

print("\n🚀 Starting training with optimizations:")
print(f"   - Batch size: {BATCH_SIZE}")
print(f"   - Max epochs: 20")
print(f"   - Model: 3 Conv layers (16→64→128 filters)")
print(f"   - Early stopping enabled (patience=5)")
print(f"   - ReduceLROnPlateau enabled\n")

history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=20,  # More epochs since we have LR reduction
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Save Model (Keras 3 requires .keras extension)
model_save_path = os.path.join(os.path.dirname(__file__), 'models', 'age_model.keras')
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"\n✅ Model saved to: {model_save_path}")

# Predict test images
predicted_ages = np.squeeze(model.predict(test_images))
true_ages = test_images.labels

r2 = r2_score(true_ages, predicted_ages)
print("Test R^2 Score: {:.5f}".format(r2))

# list all data in history
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()