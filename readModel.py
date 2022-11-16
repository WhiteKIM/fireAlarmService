import torch
import tensorflow as tf

model = tf.saved_model.load("./deep_sort/model_weights/")

print(model)