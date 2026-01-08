import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json
from datetime import datetime

print("="*80)
print("HYPERPARAMETER TUNING - Age Prediction Model")
print("="*80)

# ============================================================================
# LOAD AND BALANCE DATA (Same as train.py)
# ============================================================================

print("\n[1/4] Loading and balancing dataset...")

image_dir = Path(os.path.join(os.path.dirname(__file__), 'dataset', 'age_prediction_up', 'train'))
filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name='Filepath').astype(str)
ages = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='Age').astype(np.int)
images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)

# Data Balancing
age_ranges = [
    (1, 10, 12000), (11, 20, 12000), (21, 30, 12000), (31, 40, 12000), (41, 50, 12000),
    (51, 60, 12000), (61, 70, 12000), (71, 80, 12000), (81, 90, 12000), (91, 100, 10000),
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
    else:
        balanced_data = age_range_data.sample(n=target_count, random_state=42)
    
    balanced_dfs.append(balanced_data)

images = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1.0, random_state=1).reset_index(drop=True)
print(f"Balanced dataset: {len(images)} images")

# Train-Test Split
train_df, test_df = train_test_split(images, train_size=0.7, shuffle=True, random_state=1)

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

experiments = {
    'scenario1_default': {
        'name': 'Scenario 1: Default (Adam, batch=64)',
        'optimizer': tf.keras.optimizers.Adam(),
        'batch_size': 64,
        'color': 'steelblue'
    },
    'scenario2_low_lr': {
        'name': 'Scenario 2: Low LR (0.0001, batch=64)',
        'optimizer': tf.keras.optimizers.Adam(learning_rate=0.0001),
        'batch_size': 64,
        'color': 'coral'
    },
    'scenario3_small_batch': {
        'name': 'Scenario 3: Small Batch (Adam, batch=32)',
        'optimizer': tf.keras.optimizers.Adam(),
        'batch_size': 32,
        'color': 'mediumseagreen'
    }
}

# Create experiments folder
exp_folder = os.path.join(os.path.dirname(__file__), 'experiments')
os.makedirs(exp_folder, exist_ok=True)

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(scenario_name, config, train_df, test_df):
    print(f"\n{'='*80}")
    print(f"Training: {config['name']}")
    print(f"{'='*80}")
    
    # Create generators
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    batch_size = config['batch_size']
    
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df, x_col='Filepath', y_col='Age',
        target_size=(120, 120), batch_size=batch_size,
        class_mode='raw', subset='training', shuffle=True, seed=42
    )
    
    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df, x_col='Filepath', y_col='Age',
        target_size=(120, 120), batch_size=batch_size,
        class_mode='raw', subset='validation', shuffle=True, seed=42
    )
    
    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df, x_col='Filepath', y_col='Age',
        target_size=(120, 120), batch_size=batch_size,
        class_mode='raw', shuffle=False
    )
    
    # Build improved model with BatchNormalization and Dropout
    inputs = tf.keras.Input(shape=(120, 120, 3))
    
    # Conv Block 1
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    
    # Conv Block 2 - Increased filters to 64
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    
    # Flatten & Regularization
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)  # Dropout to reduce overfitting
    
    # Dense Layers
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    
    # Output
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile with scenario's optimizer
    model.compile(optimizer=config['optimizer'], loss='mse')
    
    # Train
    print(f"\nStarting training with batch_size={batch_size}...")
    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=50,  # Reduced for faster experimentation
        verbose=1
    )
    
    # Evaluate
    predicted_ages = np.squeeze(model.predict(test_images))
    true_ages = test_images.labels
    r2 = r2_score(true_ages, predicted_ages)
    final_val_loss = history.history['val_loss'][-1]
    
    print(f"\n✓ Test R² Score: {r2:.5f}")
    print(f"✓ Final Val Loss: {final_val_loss:.5f}")
    
    # Save model
    model_path = os.path.join(exp_folder, scenario_name)
    model.save(model_path)
    print(f"✓ Model saved: {model_path}")
    
    return {
        'history': history.history,
        'r2_score': r2,
        'final_val_loss': final_val_loss,
        'model_path': model_path
    }

# ============================================================================
# RUN ALL EXPERIMENTS
# ============================================================================

print(f"\n[2/4] Running {len(experiments)} experiments...")
results = {}

for scenario_name, config in experiments.items():
    results[scenario_name] = train_model(scenario_name, config, train_df, test_df)
    results[scenario_name]['config'] = config

# ============================================================================
# COMPARISON & VISUALIZATION
# ============================================================================

print(f"\n[3/4] Generating comparison visualization...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Training Loss
for scenario_name, result in results.items():
    config = result['config']
    ax1.plot(result['history']['loss'], label=config['name'], 
             color=config['color'], linewidth=2, alpha=0.8)
ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Validation Loss
for scenario_name, result in results.items():
    config = result['config']
    ax2.plot(result['history']['val_loss'], label=config['name'], 
             color=config['color'], linewidth=2, alpha=0.8)
ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Val Loss (MSE)')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: R² Score Comparison
scenarios = list(results.keys())
r2_scores = [results[s]['r2_score'] for s in scenarios]
colors = [results[s]['config']['color'] for s in scenarios]
labels = [results[s]['config']['name'] for s in scenarios]

bars = ax3.bar(range(len(scenarios)), r2_scores, color=colors, alpha=0.7)
ax3.set_title('R² Score Comparison (Higher is Better)', fontsize=14, fontweight='bold')
ax3.set_ylabel('R² Score')
ax3.set_xticks(range(len(scenarios)))
ax3.set_xticklabels([f"S{i+1}" for i in range(len(scenarios))], rotation=0)
ax3.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, r2_scores)):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Final Val Loss Comparison
val_losses = [results[s]['final_val_loss'] for s in scenarios]
bars = ax4.bar(range(len(scenarios)), val_losses, color=colors, alpha=0.7)
ax4.set_title('Final Validation Loss (Lower is Better)', fontsize=14, fontweight='bold')
ax4.set_ylabel('Val Loss (MSE)')
ax4.set_xticks(range(len(scenarios)))
ax4.set_xticklabels([f"S{i+1}" for i in range(len(scenarios))], rotation=0)
ax4.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, loss) in enumerate(zip(bars, val_losses)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             f'{loss:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(exp_folder, 'comparison_results.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Comparison plot saved: {plot_path}")

# ============================================================================
# DETERMINE BEST MODEL
# ============================================================================

print(f"\n[4/4] Analysis & Recommendation...")

# Find best by R² score
best_scenario = max(results.items(), key=lambda x: x[1]['r2_score'])
best_name = best_scenario[0]
best_result = best_scenario[1]

print(f"\n{'='*80}")
print("RECOMMENDATION: BEST MODEL")
print(f"{'='*80}")
print(f"Winner: {best_result['config']['name']}")
print(f"  • R² Score: {best_result['r2_score']:.5f}")
print(f"  • Final Val Loss: {best_result['final_val_loss']:.5f}")
print(f"  • Model Path: {best_result['model_path']}")
print(f"{'='*80}\n")

# Save summary to file
summary = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'best_scenario': best_name,
    'best_config': best_result['config']['name'],
    'best_r2': float(best_result['r2_score']),
    'best_val_loss': float(best_result['final_val_loss']),
    'all_results': {
        s: {'r2': float(r['r2_score']), 'val_loss': float(r['final_val_loss'])}
        for s, r in results.items()
    }
}

summary_path = os.path.join(exp_folder, 'experiment_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Summary saved: {summary_path}")

print("\n✅ All experiments completed successfully!")
print(f"📊 Check '{exp_folder}' folder for detailed results\n")
