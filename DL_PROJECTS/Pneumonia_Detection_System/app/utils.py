import tensorflow as tf
import numpy as np

IMG_SIZE = 224

def preprocess_image(image_bytes):
    img = tf.image.decode_image(image_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return tf.expand_dims(img, axis=0)