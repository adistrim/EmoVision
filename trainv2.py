import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import os
import pandas as pd
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 100
NUM_CLASSES = 7

def create_dataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        if not label.startswith('.'):
            label_path = os.path.join(dir, label)
            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    image_paths.append(os.path.join(label_path, image_name))
                    labels.append(label)
    return pd.DataFrame({'image': image_paths, 'label': labels})

def create_data_generator():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1./255,
        validation_split=0.2
    )

def build_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # First block
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Second block
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Third block
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # Dense layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def main():
    # Load and prepare data
    train_df = create_dataframe('images/train')
    test_df = create_dataframe('images/test')
    
    # Create data generators
    data_gen = create_data_generator()
    
    # Create model
    model = build_model()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        data_gen.flow_from_dataframe(
            train_df,
            x_col='image',
            y_col='label',
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode='grayscale',
            class_mode='categorical',
            batch_size=BATCH_SIZE,
            subset='training'
        ),
        validation_data=data_gen.flow_from_dataframe(
            train_df,
            x_col='image',
            y_col='label',
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode='grayscale',
            class_mode='categorical',
            batch_size=BATCH_SIZE,
            subset='validation'
        ),
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()