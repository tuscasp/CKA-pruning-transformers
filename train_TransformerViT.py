import numpy as np
from sklearn.utils import gen_batches
import random
from sklearn.metrics._classification import accuracy_score
import sys
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.data import Dataset
import tensorflow as tf
sys.path.insert(0, '../utils')

import sys
sys.path.insert(0, '../architectures')
import architecture_Transformer as arch

import custom_functions as func
import custom_callbacks

if __name__ == '__main__':
    np.random.seed(12227)

    debug = False
    data_augmentation = True
    architecture = 'TransformerViT'

    X_train, y_train, X_test, y_test, = func.cifar_resnet_data(debug)

    #model = arch.load_model('TransformerViT')
    model = arch.TransformerViT(X_train.shape[1:], 64, [12]*6, 8)

    # model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    lr = 0.01
    schedule = [(100, lr / 10), (150, lr / 100)]
    lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=lr, schedule=schedule)
    callbacks = [lr_scheduler]

    sgd = keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    arch.save_model('ViT_scratch_weights', model)

    batch_size = 1024
    for ep in range(0, 200):
        y_tmp = np.concatenate((y_train, y_train, y_train))
        X_tmp = np.concatenate(
            (func.data_augmentation(X_train),
             func.data_augmentation(X_train),
             func.data_augmentation(X_train)))

        with tf.device("CPU"):
            X_tmp = Dataset.from_tensor_slices((X_tmp, y_tmp)).shuffle(4 * batch_size).batch(batch_size)

        model.fit(X_tmp, batch_size=batch_size, verbose=2,
                 callbacks=callbacks,
                  epochs=ep, initial_epoch=ep - 1)

        if ep % 5 == 0: # % 5
            acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test, verbose=0), axis=1))
            print('Accuracy [{:.4f}]'.format(acc), flush=True)
            func.save_model('TransformerViT_epoch[{}]'.format(ep), model)
    
    func.save_model('TransformerViT', model)
    y_pred = model.predict(X_test)

    acc_test = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

    print('Testing Accuracy [{:.4f}]'.format(acc_test))
