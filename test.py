import tensorflow as tf
import numpy as np


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

X_train_mean = np.mean(X_train, axis=0)
X_train -= X_train_mean
X_test -= X_train_mean


X_train.shape


from template_architectures import Transformer

input_shape = X_train.shape[1:]
n_classes = y_train.shape[1]
projection_dim = 64
num_heads = []
for i in range(10):
    num_heads.append(4)

model = Transformer(input_shape, projection_dim, num_heads, n_classes)