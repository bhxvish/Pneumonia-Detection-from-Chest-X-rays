import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

data_dir = "chest_xray"
train_path = os.path.join(data_dir,"train")
val_path = os.path.join(data_dir,"val")
train_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)
train_data = train_gen.flow_from_directory(train_path, target_size=(150,150), batch_size=32, class_mode= 'binary')
val_data = val_gen.flow_from_directory(val_path, target_size=(150,150), batch_size= 32, class_mode= 'binary')
model = Sequential([
    Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128,(3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
                    ])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
checkpoint = ModelCheckpoint("pneumonia_model.h5", save_best_only=True)
model.fit(train_data, epochs=5, validation_data = val_data, callbacks=[checkpoint])

