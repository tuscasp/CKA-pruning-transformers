import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sys
import argparse
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.data import Dataset

import template_architectures

sys.path.insert(0, './utils')
import custom_functions as func
import custom_callbacks

def fine_tuning(model, X_train, y_train, X_test, y_test):
    batch_size = 256
    # lr = 0.001
    # schedule = [(100, lr / 10), (150, lr / 100)]
    # lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=lr, schedule=schedule)
    # callbacks = [lr_scheduler]

    # sgd = keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    for ep in range(0, 50):
        y_tmp = np.concatenate((y_train, y_train, y_train))
        X_tmp = np.concatenate(
            (func.data_augmentation(X_train),
             func.data_augmentation(X_train),
             func.data_augmentation(X_train)))

        with tf.device("CPU"):
            X_tmp = Dataset.from_tensor_slices((X_tmp, y_tmp)).shuffle(4 * batch_size).batch(batch_size)

        model.fit(X_tmp, batch_size=batch_size, verbose=2,
                  epochs=ep, initial_epoch=ep - 1)

        if ep % 5 == 0: # % 5
            acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test, verbose=0), axis=1))
            print('Accuracy [{:.4f}]'.format(acc), flush=True)
            #func.save_model('TransformerViT_epoch[{}]'.format(ep), model)
    return model

if __name__ == '__main__':
    np.random.seed(12227)
    random.seed(12227)

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--blocks', type=int,default=4)
    parser.add_argument('--dim', type=int, default=64)

    args = parser.parse_args()
    heads = args.heads
    blocks = args.blocks
    dim = args.dim

    print(args, flush=False)

    X_train, y_train, X_test, y_test = func.cifar_resnet_data(debug=False)

    input_shape = X_train.shape[1:]
    n_classes = y_train.shape[1]


    model = template_architectures.Transformer(input_shape, dim, [heads]*blocks, n_classes)
    model = fine_tuning(model, X_train, y_train, X_test, y_test)

    acc = accuracy_score(np.argmax(model.predict(X_test, verbose=0), axis=1), np.argmax(y_test, axis=1))
    print("Acc Unpruned: {}".format(acc))
    if (acc >= 0.70):
        func.save_model('./models/TransformerViT_{}_{}_{}_UNPRUNED'.format(dim, blocks, heads), model)
