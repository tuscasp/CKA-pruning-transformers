import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import random
import copy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.data import Dataset
from keras import Model
import rebuild_layers as rl


class CKA():
    __name__ = 'CKA'
    def __init__(self):
        pass

    def _debiased_dot_product_similarity_helper(self, xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n):
        return ( xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y) + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))

    def feature_space_linear_cka(self, features_x, features_y, debiased=False):
        features_x = features_x - np.mean(features_x, 0, keepdims=True)
        features_y = features_y - np.mean(features_y, 0, keepdims=True)

        dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
        normalization_x = np.linalg.norm(features_x.T.dot(features_x))
        normalization_y = np.linalg.norm(features_y.T.dot(features_y))

        if debiased:
            n = features_x.shape[0]
            # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
            sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
            sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
            squared_norm_x = np.sum(sum_squared_rows_x)
            squared_norm_y = np.sum(sum_squared_rows_y)

            dot_product_similarity = self._debiased_dot_product_similarity_helper(
                dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
                squared_norm_x, squared_norm_y, n)
            normalization_x = np.sqrt(self._debiased_dot_product_similarity_helper(
                normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
                squared_norm_x, squared_norm_x, n))
            normalization_y = np.sqrt(self._debiased_dot_product_similarity_helper(
                normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
                squared_norm_y, squared_norm_y, n))

        return dot_product_similarity / (normalization_x * normalization_y)

    def scores(self, model, X_train=None, y_train=None, allowed_layers=[], n_samples=None):
        scores = []

        if n_samples is not None:
            y_ = np.argmax(y_train, axis=1)
            sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in np.unique(y_)]
            sub_sampling = np.array(sub_sampling).reshape(-1)
        else:
            sub_sampling = np.arange(X_train.shape[0])

        F = Model(model.input, model.get_layer(index=-2).output)#Flatten
        features_F = F.predict(X_train[sub_sampling], verbose=0)

        for layer_idx in allowed_layers:

            idx_dense_and_multihead = []

            #Given a layer_normalization (allowed_layers), we need to find
            #the first and second add layers and zeroed-out the dense and multi head attention
            i = layer_idx
            while len(idx_dense_and_multihead) < 2:
                if isinstance(model.get_layer(index=i), layers.Add):
                    idx_dense_and_multihead.append(i-1)
                i = i + 1

            multi_head = model.get_layer(index=idx_dense_and_multihead[0])
            w1 = multi_head.get_weights()
            original_w1 = copy.deepcopy(w1)

            dense = model.get_layer(index=idx_dense_and_multihead[1])
            w2 = dense.get_weights()
            original_w2 = copy.deepcopy(w2)

            for i in range(0, len(w1)):
                w1[i] = np.zeros(w1[i].shape)

            for i in range(0, len(w2)):
                w2[i] = np.zeros(w2[i].shape)

            F_line = Model(model.input, model.get_layer(index=-2).output)
            features_line = F_line.predict(X_train, verbose=0)

            model.get_layer(index=idx_dense_and_multihead[0]).set_weights(original_w1)
            model.get_layer(index=idx_dense_and_multihead[1]).set_weights(original_w2)

            score = self.feature_space_linear_cka(features_F, features_line)
            scores.append((layer_idx, 1 - score))

        return scores
    
def calculate_flops(model, verbose=False):
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
    tmp = [layer._num_heads for layer in model.layers if isinstance(layer, layers.MultiHeadAttention)]
    flops = calculate_flops(model)
    print('Heads {} FLOPS {}'.format(tmp, flops), flush=True)
    # print('#Heads {} Params [{}]  FLOPS [{}] Memory [{:.6f}]'.format(tmp, n_params, 0, memory), flush=True)

def load_transformer_model(architecture_file='', weights_file=''):
    import keras
    import tensorflow as tf
    from keras import layers
    from tensorflow.keras.utils import CustomObjectScope

    class Patches(layers.Layer):
        def __init__(self, patch_size, **kwargs):
            super(Patches, self).__init__()
            self.patch_size = patch_size

        def call(self, images):
            batch_size = tf.shape(images)[0]
            patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'patch_size': self.patch_size,
            })
            return config

    class PatchEncoder(layers.Layer):
        def __init__(self, num_patches, projection_dim, **kwargs):
            super(PatchEncoder, self).__init__()
            self.num_patches = num_patches
            self.projection_dim = projection_dim
            self.projection = layers.Dense(units=self.projection_dim)

            # if weights is not None:
            #     self.projection = layers.Dense(units=projection_dim, weights=weights)

            self.position_embedding = layers.Embedding(
                input_dim=num_patches,
                output_dim=self.projection_dim
            )

        def call(self, patch):
            positions = tf.range(start=0, limit=self.num_patches, delta=1)
            encoded = self.projection(patch) + self.position_embedding(positions)
            return encoded

        def get_config(self):
            config = super().get_config().copy()
            config.update({
                'num_patches': self.num_patches,
                'projection_dim': self.projection_dim
            })
            return config

    if '.json' not in architecture_file:
        architecture_file = architecture_file+'.json'

    with open(architecture_file, 'r') as f:
        with CustomObjectScope({'PatchEncoder': PatchEncoder},
                               {'Patches':Patches}):
            model = keras.models.model_from_json(f.read())

    if weights_file != '':
        if '.h5' not in weights_file:
            weights_file = weights_file + '.h5'
        model.load_weights(weights_file)
        print('Load architecture [{}]. Load weights [{}]'.format(architecture_file, weights_file), flush=True)
    else:
        print('Load architecture [{}]'.format(architecture_file), flush=True)

    return model


if __name__ == '__main__':
    np.random.seed(21)

    debug = True

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    X_train_mean = np.mean(X_train, axis=0)
    X_train -= X_train_mean
    X_test -= X_train_mean

    if debug:
        n_samples = 10
        n_classes = len(np.unique(y_train, axis=0))
        n_samples = n_samples * n_classes
        y_ = np.argmax(y_train, axis=1)
        sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in np.unique(y_)]
        sub_sampling = np.array(sub_sampling).reshape(-1)

        X_train = X_train[sub_sampling]
        y_train = y_train[sub_sampling]

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = load_transformer_model('TransformerViT', 'TransformerViT')
    print("Unpruned:")
    statistics(model)

    for i in range(4):
        layer_method = CKA()
        scores = layer_method.scores(model, X_train, y_train, rl.layers_to_prune(model))
        model = rl.rebuild_network(model, scores, p=1)
        statistics(model)