import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("pneumonia_model.h5")
def predict(img_path):
    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis =0)
    prediction = model.predict(img_array)[0][0]
    return "PNEUMONIA" if prediction > 0.5 else "NORMAL"