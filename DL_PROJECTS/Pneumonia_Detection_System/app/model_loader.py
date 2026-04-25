import tensorflow as tf

def load_model():
    return tf.keras.models.load_model("model/pneumonia_model.h5")