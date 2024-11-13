import numpy as np
import random

def load_model(architecture_file='', weights_file=''):
    import keras
    from keras.utils.generic_utils import CustomObjectScope
    from keras import backend as K

    def _hard_swish(x):
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    if '.json' not in architecture_file:
        architecture_file = architecture_file+'.json'

    with open(architecture_file, 'r') as f:
        with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,
                                'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D,
                               '_hard_swish': _hard_swish}):
            model = keras.models.model_from_json(f.read())

    if weights_file != '':
        if '.h5' not in weights_file:
            weights_file = weights_file + '.h5'
        model.load_weights(weights_file)
        print('Load architecture [{}]. Load weights [{}]'.format(architecture_file, weights_file), flush=True)
    else:
        print('Load architecture [{}]'.format(architecture_file), flush=True)

    return model

def save_model(file_name='', model=None):
    import keras
    print('Salving architecture and weights in {}'.format(file_name))

    model.save_weights(file_name + '.h5')
    with open(file_name + '.json', 'w') as f:
        f.write(model.to_json())

def cifar_vgg_data(debug=True, validation_set=False, cifar_type=10, train_size=1.0, test_size=1.0):
    import keras
    from sklearn.model_selection import train_test_split
    print('Debuging Mode') if debug is True else print('Real Mode')

    if cifar_type == 10:
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    if cifar_type == 100:
        (X_train, y_train), (X_test, y_test) = keras.datasets.cifar100.load_data()

    if train_size!=1.0:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, random_state=42, train_size=train_size)
    if test_size!=1.0:
        _, X_test, _, y_test = train_test_split(X_test, y_test, random_state=42, test_size=test_size)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    #Works only for CIFAR-10
    if debug:
        idx_train = [4, 5, 32, 6, 24, 41, 38, 39, 59, 58, 28, 20, 27, 40, 51, 95, 103, 104, 84, 85, 87, 62, 8, 92, 67,
                     71, 76, 93, 129, 76]
        idx_test = [9, 25, 0, 22, 24, 4, 20, 1, 11, 3]

        X_train = X_train[idx_train]
        y_train = y_train[idx_train]

        X_test = X_test[idx_test]
        y_test = y_test[idx_test]

    y_train = keras.utils.to_categorical(y_train, cifar_type)
    y_test = keras.utils.to_categorical(y_test, cifar_type)
    #y_test = np.argmax(y_test, axis=1)

    mean = 120.707
    std = 64.15
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    if validation_set is False:
        return X_train, y_train, X_test, y_test
    else:
        datagen = generate_data_augmentation(X_train)
        for X_val, y_val in datagen.flow(X_train, y_train, batch_size=5000):
            break
        return X_train, y_train, X_test, y_test, X_val, y_val

def cifar_resnet_data(debug=True, validation_set=False):
    import keras
    print('Debuging Mode') if debug is True else print('Real Mode')

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    if debug:
        idx_train = [4, 5, 32, 6, 24, 41, 38, 39, 59, 58, 28, 20, 27, 40, 51, 95, 103, 104, 84, 85, 87, 62, 8, 92, 67,
                     71, 76, 93, 129, 76]
        idx_test = [9, 25, 0, 22, 24, 4, 20, 1, 11, 3]

        x_train = x_train[idx_train]
        y_train = y_train[idx_train]

        x_test = x_test[idx_test]
        y_test = y_test[idx_test]

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    #y_test = np.argmax(y_test, axis=1)

    if validation_set is False:
        return x_train, y_train, x_test, y_test
    else:
        datagen = generate_data_augmentation(x_train)
        for x_val, y_val in datagen.flow(x_train, y_train, batch_size=5000):
            break
        return x_train, y_train, x_test, y_test, x_val, y_val

def food_data(subtract_pixel_mean=False,
                   path='../Food-101/food-101_128x128.npz', train_size=1.0, test_size=1.0):
    import keras
    from sklearn.model_selection import train_test_split

    tmp = np.load(path)
    X_train, y_train, X_test, y_test = (tmp['X_train'], tmp['y_train'], tmp['X_test'], tmp['y_test'])

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    y_train = keras.utils.to_categorical(y_train, 101)
    y_test = keras.utils.to_categorical(y_test, 101)

    if train_size != 1.0:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, random_state=42, train_size=train_size)

    if test_size != 1.0:
        _, X_test, _, y_test = train_test_split(X_test, y_test, random_state=42, test_size=test_size)

    if subtract_pixel_mean is True:
        X_train_mean = np.mean(X_train, axis=0)
        X_train -= X_train_mean
        X_test -= X_train_mean

    print('#Training Samples [{}]'.format(X_train.shape[0]))
    print('#Testing Samples [{}]'.format(X_test.shape[0]))
    return X_train, y_train, X_test, y_test

def image_net_data(load_train=True, load_test=True, subtract_pixel_mean=False,
                   path='', train_size=1.0, test_size=1.0):
    import keras
    from sklearn.model_selection import train_test_split
    X_train, y_train, X_test, y_test = (None, None, None, None)
    if load_train is True:
        tmp = np.load(path+'imagenet_train.npz')
        X_train = tmp['X']
        y_train = tmp['y']

        if train_size != 1.0:
            X_train, _, y_train, _ = train_test_split(X_train, y_train, random_state=42, train_size=train_size)

        X_train = X_train.astype('float32') / 255
        y_train = keras.utils.to_categorical(y_train, 1000)

    if load_test is True:
        tmp = np.load(path + 'imagenet_val.npz')
        X_test = tmp['X']
        y_test = tmp['y']

        if test_size != 1.0:
            _, X_test, _, y_test = train_test_split(X_test, y_test, random_state=42, test_size=test_size)

        X_test = X_test.astype('float32') / 255
        y_test = keras.utils.to_categorical(y_test, 1000)

    if subtract_pixel_mean is True:
        X_train_mean = np.load(path + 'x_train_mean.npz')['X']

        if load_train is True:
            X_train -= X_train_mean#X_train_mean = np.mean(X_train, axis=0)
        if load_test is True:
            X_test -= X_train_mean
    print('#Training Samples [{}]'.format(X_train.shape[0])) if X_train is not None else print('#Training Samples [0]')
    print('#Testing Samples [{}]'.format(X_test.shape[0])) if X_test is not None else print('#Testing Samples [0]')
    return X_train, y_train, X_test, y_test

def image_net_tiny_data(subtract_pixel_mean=False,
                   path='../../datasets/ImageNetTiny/TinyImageNet.npz', train_size=1.0, test_size=1.0):
    import keras
    from sklearn.model_selection import train_test_split

    tmp = np.load(path)
    X_train, y_train, X_test, y_test = (tmp['X_train'], tmp['y_train'], tmp['X_test'], tmp['y_test'])

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    y_train = keras.utils.to_categorical(y_train, 200)
    y_test = keras.utils.to_categorical(y_test, 200)

    if train_size != 1.0:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, random_state=42, train_size=train_size)

    if test_size != 1.0:
        _, X_test, _, y_test = train_test_split(X_test, y_test, random_state=42, test_size=test_size)

    if subtract_pixel_mean is True:
        X_train_mean = np.mean(X_train, axis=0)
        X_train -= X_train_mean
        X_test -= X_train_mean

    print('#Training Samples [{}]'.format(X_train.shape[0]))
    print('#Testing Samples [{}]'.format(X_test.shape[0]))
    return X_train, y_train, X_test, y_test

def optimizer_compile(model, model_type='VGG16'):
    import keras
    if model_type == 'VGG16':
        sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    if model_type == 'ResNetV1':
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])
    return model

def lr_schedule(epoch, init_lr=0.01, schedule=[(25, 0.001), (50, 0.0001)]):

    for i in range(0, len(schedule)-1):
        if epoch > schedule[i][0] and epoch < schedule[i+1][0]:
            print('Learning rate: ', schedule[i][0])
            return schedule[i][0]

    if epoch > schedule[-1][0]:
        print('Learning rate: ', schedule[-1][0])
        return schedule[-1][1]

    print('Learning rate: ', init_lr)
    return init_lr

def generate_data_augmentation(X_train):
    print('Using real-time data augmentation.')
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(X_train)
    return datagen

def save_data_augmentation(X_train, y_train, batch_size=256, file_name=''):
    datagen = generate_data_augmentation(X_train)
    X = None
    y = None

    for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=batch_size):
        if X is not None:
            X = np.concatenate((X, X_batch))
            y = np.concatenate((y, y_batch))
        else:
            X = X_batch
            y = y_batch

        X_train = np.concatenate((X_train, X))
        y_train = np.concatenate((y_train, y))

    if file_name == '':
        return X_train, y_train
    else:
        np.savez_compressed(file_name, X=X_train, y=y_train)

def count_filters(model):
    import keras
    from keras.applications.mobilenet import DepthwiseConv2D
    n_filters = 0
    #Model contains only Conv layers
    for layer_idx in range(1, len(model.layers)):

        layer = model.get_layer(index=layer_idx)

        if isinstance(layer, keras.layers.Conv2D) and not isinstance(layer, DepthwiseConv2D):
            config = layer.get_config()
            n_filters += config['filters']

        if isinstance(layer, DepthwiseConv2D):
            n_filters += layer.output_shape[-1]

    #Todo: Model contains Conv and Fully Connected layers
    # for layer_idx in range(1, len(model.get_layer(index=1))):
    #     layer = model.get_layer(index=1).get_layer(index=layer_idx)
    #     if isinstance(layer, keras.layers.Conv2D) == True:
    #         config = layer.get_config()
    #     n_filters += config['filters']
    return n_filters

def count_filters_layer(model):
    import keras
    from keras.applications.mobilenet import DepthwiseConv2D
    n_filters = ''
    #Model contains only Conv layers
    for layer_idx in range(1, len(model.layers)):

        layer = model.get_layer(index=layer_idx)
        if isinstance(layer, keras.layers.Conv2D) and not isinstance(layer, DepthwiseConv2D):
            config = layer.get_config()
            n_filters += str(config['filters']) + ' '

        if isinstance(layer, DepthwiseConv2D):
            n_filters += str(layer.output_shape[-1])

    return n_filters

def compute_flops(model):
    import keras
    total_flops =0
    flops_per_layer = []

    try:
        layer = model.get_layer(index=1).layers #Just for discover the model type
        for layer_idx in range(1, len(model.get_layer(index=1).layers)):
            layer = model.get_layer(index=1).get_layer(index=layer_idx)
            if isinstance(layer, keras.layers.Conv2D) is True:
                _, output_map_H, output_map_W, current_layer_depth = layer.output_shape

                _, _, _, previous_layer_depth = layer.input_shape
                kernel_H, kernel_W = layer.kernel_size

                flops = output_map_H * output_map_W * previous_layer_depth * current_layer_depth * kernel_H * kernel_W
                total_flops += flops
                flops_per_layer.append(flops)

        for layer_idx in range(1, len(model.layers)):
            layer = model.get_layer(index=layer_idx)
            if isinstance(layer, keras.layers.Dense) is True:
                _, current_layer_depth = layer.output_shape

                _, previous_layer_depth = layer.input_shape

                flops = current_layer_depth * previous_layer_depth
                total_flops += flops
                flops_per_layer.append(flops)
    except:
        for layer_idx in range(1, len(model.layers)):
            layer = model.get_layer(index=layer_idx)
            if isinstance(layer, keras.layers.Conv2D) is True:
                _, output_map_H, output_map_W, current_layer_depth = layer.output_shape

                _, _, _, previous_layer_depth = layer.input_shape
                kernel_H, kernel_W = layer.kernel_size

                flops = output_map_H * output_map_W * previous_layer_depth * current_layer_depth * kernel_H * kernel_W
                total_flops += flops
                flops_per_layer.append(flops)

            if isinstance(layer, keras.layers.Dense) is True:
                _, current_layer_depth = layer.output_shape

                _, previous_layer_depth = layer.input_shape

                flops = current_layer_depth * previous_layer_depth
                total_flops += flops
                flops_per_layer.append(flops)

    return total_flops, flops_per_layer

def generate_conv_model(model, input_shape=(32, 32,3), model_type='VGG16'):
    from keras.layers.pooling import GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D
    from keras.layers import Dense, Dropout, Conv2D, Flatten, Activation, BatchNormalization, Add
    from keras.layers import Input
    from keras.models import Model

    model = model.get_layer(index=1)

    if model_type == 'VGG16':
        inp = (model.inputs[0].shape.dims[1].value,
               model.inputs[0].shape.dims[2].value,
               model.inputs[0].shape.dims[3].value)

        H = Input(inp)
        inp = H

        for layer_idx in range(1, len(model.layers)):

            layer = model.get_layer(index=layer_idx)
            config = layer.get_config()

            if isinstance(layer, MaxPooling2D):
                H = MaxPooling2D.from_config(config)(H)

            if isinstance(layer, Dropout):
                H = Dropout.from_config(config)(H)

            if isinstance(layer, Activation):
                H = Activation.from_config(config)(H)

            if isinstance(layer, BatchNormalization):
                weights = layer.get_weights()
                H = BatchNormalization(weights=weights)(H)

            elif isinstance(layer, Conv2D):
                weights = layer.get_weights()

                config['filters'] = weights[1].shape[0]
                H = Conv2D(activation=config['activation'],
                           activity_regularizer=config['activity_regularizer'],
                           bias_constraint=config['bias_constraint'],
                           bias_regularizer=config['bias_regularizer'],
                           data_format=config['data_format'],
                           dilation_rate=config['dilation_rate'],
                           filters=config['filters'],
                           kernel_constraint=config['kernel_constraint'],
                           kernel_regularizer=config['kernel_regularizer'],
                           kernel_size=config['kernel_size'],
                           name=config['name'],
                           padding=config['padding'],
                           strides=config['strides'],
                           trainable=config['trainable'],
                           use_bias=config['use_bias'],
                           weights=weights
                           )(H)

    if model_type == 'ResNetV1':
        inp = input_shape
        H = Input(inp)
        inp = H
        def create_Conv2D_from_conf(config, input, weights):
            return Conv2D(activation=config['activation'],
                          activity_regularizer=config['activity_regularizer'],
                          bias_constraint=config['bias_constraint'],
                          bias_regularizer=config['bias_regularizer'],
                          data_format=config['data_format'],
                          dilation_rate=config['dilation_rate'],
                          filters=config['filters'],
                          kernel_constraint=config['kernel_constraint'],
                          kernel_regularizer=config['kernel_regularizer'],
                          kernel_size=config['kernel_size'],
                          name=config['name'],
                          padding=config['padding'],
                          strides=config['strides'],
                          trainable=config['trainable'],
                          use_bias=config['use_bias'],
                          weights=weights
                          )(input)

        H = create_Conv2D_from_conf(model.get_layer(index=1).get_config(), H,
                                         model.get_layer(index=1).get_weights())
        H = BatchNormalization(weights=model.get_layer(index=2).get_weights())(H)  # 2 batch_normalization_1
        H = Activation.from_config(model.get_layer(index=3).get_config())(H)  # 3 activation_1
        skip = H

        H = create_Conv2D_from_conf(model.get_layer(index=4).get_config(), H, model.get_layer(index=4).get_weights())  # 4 conv2d_2
        H = BatchNormalization(weights=model.get_layer(index=5).get_weights())(H)  # 5 batch_normalization_2
        H = Activation.from_config(model.get_layer(index=6).get_config())(H)  # 6 activation_2

        H = create_Conv2D_from_conf(model.get_layer(index=7).get_config(), H, model.get_layer(index=7).get_weights())  # 7 conv2d_3
        H = BatchNormalization(weights=model.get_layer(index=8).get_weights())(H)  # 8 batch_normalization_3

        H = Add()([skip, H])  # 9 add_1
        ##########Block1 Done######

        H = Activation.from_config(model.get_layer(index=10).get_config())(H)  # 10 activation_3
        skip = H  # 10 activation_3

        H = create_Conv2D_from_conf(model.get_layer(index=11).get_config(), H, model.get_layer(index=11).get_weights())   # 11 conv2d_4
        H = BatchNormalization(weights=model.get_layer(index=12).get_weights())(H)  # 12 batch_normalization_4
        H = Activation.from_config(model.get_layer(index=13).get_config())(H)  # 13 activation_4

        H = create_Conv2D_from_conf(model.get_layer(index=14).get_config(), H, model.get_layer(index=14).get_weights())  # 14 conv2d_5
        H = BatchNormalization(weights=model.get_layer(index=15).get_weights())(H)  # 15 batch_normalization_5

        H = Add()([skip, H])  # 16
        ##########Block2 Done######

        H = Activation.from_config(model.get_layer(index=17).get_config())(H)  # 17 activation_5
        skip = H  # 17 activation_5

        H = create_Conv2D_from_conf(model.get_layer(index=18).get_config(), H, model.get_layer(index=18).get_weights())  # 18 conv2d_6
        H = BatchNormalization(weights=model.get_layer(index=19).get_weights())(H)  # 19 batch_normalization_6
        H = Activation.from_config(model.get_layer(index=20).get_config())(H)  # 20 activation_6

        H = create_Conv2D_from_conf(model.get_layer(index=21).get_config(), H, model.get_layer(index=21).get_weights())  # 21 conv2d_7
        H = BatchNormalization(weights=model.get_layer(index=22).get_weights())(H)  # 22 batch_normalization_7

        H = Add()([skip, H])  # 23 add_3
        ##########Block3 Done######

        H = Activation.from_config(model.get_layer(index=24).get_config())(H)  # 24 activation_7
        skip = create_Conv2D_from_conf(model.get_layer(index=29).get_config(), H, model.get_layer(index=29).get_weights())  # 29 conv2d_10

        H = create_Conv2D_from_conf(model.get_layer(index=25).get_config(), H, model.get_layer(index=25).get_weights())  # 25 conv2d_8
        H = BatchNormalization(weights=model.get_layer(index=26).get_weights())(H)  # 26 batch_normalization_8
        H = Activation.from_config(model.get_layer(index=27).get_config())(H)  # 27 activation_8

        H = create_Conv2D_from_conf(model.get_layer(index=28).get_config(), H, model.get_layer(index=28).get_weights())  # 28 conv2d_9
        H = BatchNormalization(weights=model.get_layer(index=30).get_weights())(H)  # 30 batch_normalization_9

        H = Add()([skip, H])
        ##########Block4 Done######

        H = Activation.from_config(model.get_layer(index=32).get_config())(H)  # 32 activation_9
        skip = H

        H = create_Conv2D_from_conf(model.get_layer(index=33).get_config(), H, model.get_layer(index=33).get_weights())  # 33 conv2d_11
        H = BatchNormalization(weights=model.get_layer(index=34).get_weights())(H)  # 34 batch_normalization_10
        H = Activation.from_config(model.get_layer(index=35).get_config())(H)  # 35 activation_10

        H = create_Conv2D_from_conf(model.get_layer(index=36).get_config(), H, model.get_layer(index=36).get_weights())  # 36 conv2d_12
        H = BatchNormalization(weights=model.get_layer(index=37).get_weights())(H)  # 37 batch_normalization_11

        H = Add()([skip, H])  # 38 add_5
        ##########Block5 Done######

        H = Activation.from_config(model.get_layer(index=39).get_config())(H)  # 39 activation_11
        skip = H  # 39 activation_11

        H = create_Conv2D_from_conf(model.get_layer(index=40).get_config(), H, model.get_layer(index=40).get_weights())  # 40 conv2d_13
        H = BatchNormalization(weights=model.get_layer(index=41).get_weights())(H)  # 41 batch_normalization_12
        H = Activation.from_config(model.get_layer(index=42).get_config())(H)  # 42 activation_12

        H = create_Conv2D_from_conf(model.get_layer(index=43).get_config(), H, model.get_layer(index=43).get_weights())  # 43 conv2d_14
        H = BatchNormalization(weights=model.get_layer(index=44).get_weights())(H)  # 44 batch_normalization_13

        H = Add()([skip, H])
        ##########Block5 Done######

        H = Activation.from_config(model.get_layer(index=46).get_config())(H)  # 46 activation_13

        skip = create_Conv2D_from_conf(model.get_layer(index=51).get_config(), H, model.get_layer(index=51).get_weights())  # 51 conv2d_17

        H = create_Conv2D_from_conf(model.get_layer(index=47).get_config(), H, model.get_layer(index=47).get_weights())  # 47 conv2d_15
        H = BatchNormalization(weights=model.get_layer(index=48).get_weights())(H)  # 48 batch_normalization_14
        H = Activation.from_config(model.get_layer(index=49).get_config())(H)  # 49 activation_14

        H = create_Conv2D_from_conf(model.get_layer(index=50).get_config(), H, model.get_layer(index=50).get_weights())  # 50 conv2d_16
        H = BatchNormalization(weights=model.get_layer(index=52).get_weights())(H)  # 52 batch_normalization_15

        H = Add()([skip, H])  # 53 add_7
        ##########Block6 Done######

        H = Activation.from_config(model.get_layer(index=54).get_config())(H)  # 54 activation_15
        skip = H  # 54 activation_15

        H = create_Conv2D_from_conf(model.get_layer(index=55).get_config(), H, model.get_layer(index=55).get_weights())  # 55 conv2d_18
        H = BatchNormalization(weights=model.get_layer(index=56).get_weights())(H)  # 56 batch_normalization_16
        H = Activation.from_config(model.get_layer(index=57).get_config())(H)  # 57 activation_16

        H = create_Conv2D_from_conf(model.get_layer(index=58).get_config(), H, model.get_layer(index=58).get_weights())  # 58 conv2d_19
        H = BatchNormalization(weights=model.get_layer(index=59).get_weights())(H)  # 59 batch_normalization_17

        H = Add()([skip, H])  # 60 add_8
        ##########Block7 Done######

        H = Activation.from_config(model.get_layer(index=61).get_config())(H)  # 61 activation_17
        skip = H

        H = create_Conv2D_from_conf(model.get_layer(index=62).get_config(), H, model.get_layer(index=62).get_weights())  # 62 conv2d_20
        H = BatchNormalization(weights=model.get_layer(index=63).get_weights())(H)  # 63 batch_normalization_18
        H = Activation.from_config(model.get_layer(index=64).get_config())(H)  # 64 activation_18

        H = create_Conv2D_from_conf(model.get_layer(index=65).get_config(), H, model.get_layer(index=65).get_weights())  # 65 conv2d_21
        H = BatchNormalization(weights=model.get_layer(index=66).get_weights())(H)  # 66 batch_normalization_19

        H = Add()([skip, H])  # 67 add_9
        ##########Block8 Done######

        H = Activation.from_config(model.get_layer(index=68).get_config())(H)  # 68 activation_19
        H = AveragePooling2D.from_config(model.get_layer(index=69).get_config())(H)

    return Model(inp, H)

def convert_model(model, inp=None):
    #This fuction convertes a model from Input->Model -> Dense -> Dese
    # to Input -> Conv2D->...->Dense->Dense
    import keras
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
    from keras.layers import Input, BatchNormalization, Activation

    if inp is None:
        inp = (model.inputs[0].shape.dims[1].value,
               model.inputs[0].shape.dims[2].value,
               model.inputs[0].shape.dims[3].value)

    H = Input(inp)
    inp = H
    new_idx = 1
    #Check if the convolutional layers are a layer in current model
    if isinstance(model.get_layer(index=1), keras.models.Model):
        cnn_model = model.get_layer(index=1)

        for layer in cnn_model.layers:
            config = layer.get_config()
            new_idx = new_idx+1
            if isinstance(layer, MaxPooling2D):
                H = MaxPooling2D.from_config(config)(H)

            if isinstance(layer, Dropout):
                H = Dropout.from_config(config)(H)

            if isinstance(layer, Activation):
                H = Activation.from_config(config)(H)

            if isinstance(layer, BatchNormalization):
                weights = layer.get_weights()
                H = BatchNormalization(weights=weights)(H)

            if isinstance(layer, Conv2D):
                weights = layer.get_weights()
                H = Conv2D(activation=config['activation'],
                           activity_regularizer=config['activity_regularizer'],
                           bias_constraint=config['bias_constraint'],
                           bias_regularizer=config['bias_regularizer'],
                           data_format=config['data_format'],
                           dilation_rate=config['dilation_rate'],
                           filters=config['filters'],
                           kernel_constraint=config['kernel_constraint'],
                           kernel_regularizer=config['kernel_regularizer'],
                           kernel_size=config['kernel_size'],
                           name=config['name'],
                           padding=config['padding'],
                           strides=config['strides'],
                           trainable=config['trainable'],
                           use_bias=config['use_bias'],
                           weights=weights
                           )(H)

    for layer in model.layers:
        config = layer.get_config()

        layer_id = config['name'].split('_')[-1]
        config['name'] = config['name'].replace(layer_id, str(new_idx))
        new_idx = new_idx+1

        if isinstance(layer, Dropout):
            H = Dropout.from_config(config)(H)

        if isinstance(layer, Activation):
            H = Activation.from_config(config)(H)

        if isinstance(layer, Flatten):
            H = Flatten()(H)

        if isinstance(layer, Dense):
            weights = layer.get_weights()
            H = Dense(units=config['units'],
                      activation=config['activation'],
                      use_bias=config['use_bias'],
                      kernel_initializer=config['kernel_initializer'],
                      bias_initializer=config['bias_initializer'],
                      kernel_regularizer=config['kernel_regularizer'],
                      bias_regularizer=config['bias_regularizer'],
                      activity_regularizer=config['activity_regularizer'],
                      kernel_constraint=config['kernel_constraint'],
                      bias_constraint=config['bias_constraint'],
                      weights=weights)(H)

    return keras.models.Model(inp, H)

def gini_index(y_predict, y_expected):
    gini = []
    c1 = np.where(y_expected != 1)#Negative samples
    c2 = np.where(y_expected == 1) #Positive samples
    n = y_expected.shape[0]
    thresholds = y_predict
    for th in thresholds:

        tmp = np.where( y_predict[c1] < th )[0] #Predict correctly the negative sample
        c1c1 = tmp.shape[0]
        n1 = c1c1

        tmp = np.where( y_predict[c1] >= th )[0] #Predict the negative sample as positive
        c1c2 = tmp.shape[0]
        n2 = c1c2

        tmp = np.where( y_predict[c2] >= th )[0] #Predict correctly the positive sample
        c2c2 = tmp.shape[0]
        n2 = n2 + c2c2
        tmp = np.where( y_predict[c2] < th )[0] #Predict the positive samples as negative
        c2c1 = tmp.shape[0]
        n1 = n1 + c2c1

        if n1 == 0 or n2 == 0:
            gini.append(9999)
            continue
        else:
            gini1 = (c1c1/n1)**2 - (c2c1/n1)**2
            gini1 = 1 - gini1
            gini1 = (n1/n) * gini1

            gini2 = (c2c2/n2)**2 - (c1c2/n2)**2
            gini2 = 1 - gini2
            gini2 = (n2/n) * gini2


            gini.append(gini1+gini2)
    if len(gini)>0:
        idx = gini.index(min(gini))
        best_th = thresholds[idx]
    else:
        print('Gini threshold not found')
        best_th = 0

    return best_th

def top_k_accuracy(y_true, y_pred, k):
    top_n = np.argsort(y_pred, axis=1)[:,-k:]
    idx_class = np.argmax(y_true, axis=1)
    hit = 0
    for i in range(idx_class.shape[0]):
      if idx_class[i] in top_n[i,:]:
        hit = hit + 1
    return float(hit)/idx_class.shape[0]

def center_crop(image, crop_size=224):
    h, w, _ = image.shape

    top = (h - crop_size) // 2
    left = (w - crop_size) // 2

    bottom = top + crop_size
    right = left + crop_size

    return image[top:bottom, left:right, :]

# def random_crop(img=None, random_crop_size=(32, 32)):
#     #Code taken from https://jkjung-avt.github.io/keras-image-cropping/
#     height, width = img.shape[0], img.shape[1]
#     dy, dx = random_crop_size
#     x = np.random.randint(0, width - dx + 1)
#     y = np.random.randint(0, height - dy + 1)
#     return img[y:(y+dy), x:(x+dx), :]
#
# def data_augmentation(X):
#     X_out = np.zeros((X.shape), dtype=X.dtype)
#     n_samples = X.shape[0]
#     padded_sample = np.zeros((40, 40, 3), dtype=X.dtype)
#     for i in range(0, n_samples):
#         p = random.random()
#         padded_sample[4:36, 4:36, :] = X[i][:, :, :]
#         if p >= 0.5: #random crop on the original image
#             X_out[i] = random_crop(padded_sample)
#         else: #random crop on the flipped image
#             X_out[i] = random_crop(np.flip(padded_sample, axis=1))
#
#     return X_out

def random_crop(img=None, random_crop_size=(64, 64)):
    #Code taken from https://jkjung-avt.github.io/keras-image-cropping/
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]

def data_augmentation(X, padding=4):

    X_out = np.zeros(X.shape, dtype=X.dtype)
    n_samples, x, y, _ = X.shape

    padded_sample = np.zeros((x+padding*2, y+padding*2, 3), dtype=X.dtype)

    for i in range(0, n_samples):
        p = random.random()
        padded_sample[padding:x+padding, padding:y+padding, :] = X[i][:, :, :]
        if p >= 0.5: #random crop on the original image
            X_out[i] = random_crop(padded_sample, (x, y))
        else: #random crop on the flipped image
            X_out[i] = random_crop(np.flip(padded_sample, axis=1), (x, y))

        # import matplotlib.pyplot as plt
        # plt.imshow(X_out[i])

    return X_out

def cutout(img):
    MAX_CUTS = 5  # chance to get more cuts
    MAX_LENGTH_MULTIPLIER = 5  # change to get larger cuts ; 16 for cifar 10, 8 for cifar 100

    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

    mask = np.ones((height, width, channels), np.float32)
    nb_cuts = np.random.randint(0, MAX_CUTS + 1)

    for i in range(nb_cuts):
        y = np.random.randint(height)
        x = np.random.randint(width)
        length = 4 * np.random.randint(1, MAX_LENGTH_MULTIPLIER + 1)

        y1 = np.clip(y - length // 2, 0, height)
        y2 = np.clip(y + length // 2, 0, height)
        x1 = np.clip(x - length // 2, 0, width)
        x2 = np.clip(x + length // 2, 0, width)

        mask[y1: y2, x1: x2, :] = 0.

    # apply mask
    img = img * mask

    return img

def memory_usage(batch_size, model):
    from keras import backend as K
    #Taken from #https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = total_memory / (1024.0 ** 3)
    return gbytes

def count_depth(model):
    import keras
    depth = 0
    for i in range(0, len(model.layers)):
        layer = model.get_layer(index=i)
        if isinstance(layer, keras.layers.Conv2D) == True:
            depth = depth + 1
    print('Depth: [{}]'.format(depth))
    return depth