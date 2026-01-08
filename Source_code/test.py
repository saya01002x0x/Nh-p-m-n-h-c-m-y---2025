import numpy as np
import pandas as pd
from pathlib import Path
import os.path

from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf

from sklearn.metrics import r2_score

# Load Model
model = keras.models.load_model('D:/chuyen_nganh/20221/machine_learning/project/Nhap_mom_ML-hao/Nhap_mom_ML-hao/models4')
print('ok')

# Load images from folder
image_dir = Path('D:/chuyen_nganh/20221/machine_learning/project/Nhap_mom_ML-hao/Nhap_mom_ML-hao/age_prediction/train/025')
filepaths = pd.Series(list(image_dir.glob(r'*.jpg')), name='Filepath').astype(str)
ages = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name='Age').astype(np.int)
images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)


# Load images
# image_dir = Path('D:/chuyen_nganh/20221/machine_learning/project/Nhap_mom_ML-hao/Nhap_mom_ML-hao/age_prediction/test/047')
# filepaths = pd.Series((image_dir), name='Filepath').astype(str)
# ages = pd.Series(name='Age')
# images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)

# Create ImageDataGenerator
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

# Change images to dataframe
test_images = train_generator.flow_from_dataframe(
    dataframe=images,
    x_col='Filepath',
    y_col='Age',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='raw',
    batch_size=64,
    shuffle=False
)

# Predict images
# print(model.evaluate(test_images, verbose=2))
predicted_ages = np.squeeze(model.predict(test_images))
true_ages = test_images.labels

# rmse = np.sqrt(model.evaluate(test_images, verbose=0))
# print("     Test RMSE: {:.5f}".format(rmse))
#
# r2 = r2_score(true_ages, predicted_ages)
# print("Test R^2 Score: {:.5f}".format(r2))
print(predicted_ages)
print(np.mean(predicted_ages))
# print(true_ages[0:100])
# print(model.evaluate(test_images, verbose=1))