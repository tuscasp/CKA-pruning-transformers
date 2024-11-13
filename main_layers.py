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

import rebuild_heads as rh
import rebuild_layers as rl

import template_architectures
from pruning_criteria import criteria_head as ch
from pruning_criteria import criteria_layer as cl

sys.path.insert(0, './utils')
import custom_functions as func
import custom_callbacks

def flops(model, verbose=False):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function([tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops

def statistics(model):
    n_params = model.count_params()
    n_heads = func.count_filters(model)
    memory = func.memory_usage(1, model)
    tmp = [layer._num_heads for layer in model.layers if isinstance(layer, layers.MultiHeadAttention)]

    #
    print('#Heads {} Params [{}]  FLOPS [{}] Memory [{:.6f}]'.format(tmp, n_params, 0, memory), flush=True)

def fine_tuning(model, X_train, y_train, X_test, y_test):
    batch_size = 1024
    lr = 0.001
    schedule = [(100, lr / 10), (150, lr / 100)]
    lr_scheduler = custom_callbacks.LearningRateScheduler(init_lr=lr, schedule=schedule)
    callbacks = [lr_scheduler]

    sgd = keras.optimizers.SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
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
            #func.save_model('TransformerViT_epoch[{}]'.format(ep), model)
    return model

if __name__ == '__main__':
    np.random.seed(12227)
    random.seed(12227)

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture_name', type=str, default='TransformerViT')
    parser.add_argument('--criterion_layer', type=str, default='CKA')
    parser.add_argument('--p_layer', type=int, default=1)

    args = parser.parse_args()
    architecture_name = args.architecture_name
    criterion_layer = args.criterion_layer
    p_layer = args.p_layer

    print(args, flush=False)
    print('Architecture [{}] p_layer[{}]'.format(architecture_name, p_layer), flush=True)

    X_train, y_train, X_test, y_test = func.cifar_resnet_data(debug=True)

    input_shape = X_train.shape[1:]
    n_classes = y_train.shape[1]

    model = func.load_transformer_model('TransformerViT', 'TransformerViT')

    y_pred = model.predict(X_test, verbose=0)
    acc = accuracy_score(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1))
    print('Accuracy Unpruned [{}]'.format(acc))
    #statistics(model)
    print("FLOPs: {}".format(flops(model)))
    '''
    max_iterations = len(rl.layers_to_prune(model))-1
    for i in range(0, max_iterations):
        layer_method = cl.criteria(criterion_layer)
        scores = layer_method.scores(model, X_train, y_train, rl.layers_to_prune(model))
        model = rl.rebuild_network(model, scores, p_layer)


        #Uncomment to real experiments
        #model = fine_tuning(model, X_train, y_train, X_test, y_test)
        acc = accuracy_score(np.argmax(model.predict(X_test, verbose=0), axis=1), np.argmax(y_test, axis=1))
        statistics(model)
        print('Acc [{}]'.format(acc))
        print('Iteration [{}] Accuracy [{}]'.format(i, acc))
        #func.save_model('{}_{}_p[{}]_iterations[{}]'.format(architecture_name, criterion_layer, p_layer, i), model)
    '''