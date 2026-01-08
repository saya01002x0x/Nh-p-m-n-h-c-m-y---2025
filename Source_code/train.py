import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load images
image_dir = Path('D:/chuyen_nganh/20221/machine_learning/project/Nhap_mom_ML-hao/Nhap_mom_ML-hao/age_prediction/train')
filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name='Filepath').astype(str)
ages = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='Age').astype(np.int)
images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)

train_df, test_df = train_test_split(images, train_size=0.7, shuffle=True, random_state=1)

# Create ImageDataGenerator
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

# Change iamges to dataframe
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
    batch_size=64,
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
    batch_size=64,
    shuffle=False,
    save_to_dir=None,
    save_prefix='',
    save_format='png',
    interpolation='nearest',
    validate_filenames=True,
)

# Create Model
inputs = tf.keras.Input(shape=(120, 120, 3))
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='mse'
)

# Train Model
history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=74
)

# Save Model
model.save('models8')

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