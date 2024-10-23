import tensorflow as tf
from tensorflow.keras.utils import to_categorical, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Directories
TRAIN_DIR = 'images/train'
TEST_DIR = 'images/test'

# Load images and labels into a DataFrame
def createDataFrame(dir):
    imagePaths = []
    labels = []
    for label in os.listdir(dir):
        if not label.startswith('.'):
            for imageName in os.listdir(os.path.join(dir, label)):
                imagePaths.append(os.path.join(dir, label, imageName))
                labels.append(label)
            print(f'Finished loading images from {label}')
    return imagePaths, labels

train = pd.DataFrame()
train['image'], train['label'] = createDataFrame(TRAIN_DIR)

test = pd.DataFrame()
test['image'], test['label'] = createDataFrame(TEST_DIR)

# Extract features from images
def extractFeatures(images):
    print('Extracting features...')
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode="grayscale")  # Load as grayscale
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)  # Reshape for Conv2D input
    return features

trainFeatures = extractFeatures(train['image'])
testFeatures = extractFeatures(test['image'])

# Normalize the images
X_train = trainFeatures / 255.0
X_test = testFeatures / 255.0

# Encode the labels
label_encoder = LabelEncoder()
Y_train = label_encoder.fit_transform(train['label'])
Y_test = label_encoder.transform(test['label'])

# Convert labels to one-hot encoding
Y_train = to_categorical(Y_train, num_classes=7)
Y_test = to_categorical(Y_test, num_classes=7)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Use a Pretrained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Modify the model for grayscale images by adding the grayscale channels to 3 channels
model = Sequential()

model.add(tf.keras.layers.Conv2D(3, (1, 1), padding='same', input_shape=(48, 48, 1)))  # To convert grayscale to 3 channels
model.add(base_model)

# Add new fully connected layers on top
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))  # Reduced dropout rate
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))  # Reduced dropout rate
model.add(Dense(7, activation='softmax'))

# Freeze the base model layers
base_model.trainable = False

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Learning rate scheduler to adjust learning rate during training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

# Unfreeze the last few VGG16 layers for fine-tuning
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Compile the model with a lower learning rate for fine-tuning
model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              metrics=['accuracy'])

# Train the model with both callbacks: reduce_lr and early_stopping
with tf.device('/GPU:0'):
    history = model.fit(datagen.flow(X_train, Y_train, batch_size=64), 
                        epochs=200, 
                        validation_data=(X_test, Y_test), 
                        callbacks=[reduce_lr, early_stopping])

# Save the model
model.save("models/VGG16/emotionCheck_VGG16.h5")
