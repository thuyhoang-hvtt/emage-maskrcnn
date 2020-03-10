"""EfficientNet models for Keras.

# Reference paper

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks]
  (https://arxiv.org/abs/1905.11946) (ICML 2019)

# Reference implementation

- [TensorFlow]
  (https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

from . import correct_pad
from . import utils
from .utils import _obtain_input_shape
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

BASE_WEIGHTS_PATH = (
    'https://github.com/Callidior/keras-applications/'
    'releases/download/efficientnet/')
WEIGHTS_HASHES = {
    'b0': ('e9e877068bd0af75e0a36691e03c072c',
           '345255ed8048c2f22c793070a9c1a130'),
    'b1': ('8f83b9aecab222a9a2480219843049a1',
           'b20160ab7b79b7a92897fcb33d52cc61'),
    'b2': ('b6185fdcd190285d516936c09dceeaa4',
           'c6e46333e8cddfa702f4d8b8b6340d70'),
    'b3': ('b2db0f8aac7c553657abb2cb46dcbfbb',
           'e0cf8654fad9d3625190e30d70d0c17d'),
    'b4': ('ab314d28135fe552e2f9312b31da6926',
           'b46702e4754d2022d62897e0618edc7b'),
    'b5': ('8d60b903aff50b09c6acf8eaba098e09',
           '0a839ac36e46552a881f2975aaab442f'),
    'b6': ('a967457886eac4f5ab44139bdd827920',
           '375a35c17ef70d46f9c664b03b4437f2'),
    'b7': ('e964fd6e26e9a4c144bcb811f2a10f20',
           'd55674cc46b805f4382d18bc08ed43c1')
}

DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def swish(x):
    """Swish activation function.

    # Arguments
        x: Input tensor.

    # Returns
        The Swish activation: `x * sigmoid(x)`.

    # References
        [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
    """
    if K.backend() == 'tensorflow':
        try:
            # The native TF implementation has a more
            # memory-efficient gradient implementation
            return K.tf.nn.swish(x)
        except AttributeError:
            pass

    return x * K.sigmoid(x)


def block(inputs, activation_fn=swish, drop_rate=0., name='',
          filters_in=32, filters_out=16, kernel_size=3, strides=1,
          expand_ratio=1, se_ratio=0., id_skip=True):
    """A mobile inverted residual block.

    # Arguments
        inputs: input tensor.
        activation_fn: activation function.
        drop_rate: float between 0 and 1, fraction of the input units to drop.
        name: string, block label.
        filters_in: integer, the number of input filters.
        filters_out: integer, the number of output filters.
        kernel_size: integer, the dimension of the convolution window.
        strides: integer, the stride of the convolution.
        expand_ratio: integer, scaling coefficient for the input filters.
        se_ratio: float between 0 and 1, fraction to squeeze the input filters.
        id_skip: boolean.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    # Expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = KL.Conv2D(filters, 1,
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=name + 'expand_conv')(inputs)
        x = KL.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
        x = KL.Activation(activation_fn, name=name + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    if strides == 2:
        x = KL.ZeroPadding2D(padding=correct_pad(K, x, kernel_size),
                             name=name + 'dwconv_pad')(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'
    x = KL.DepthwiseConv2D(kernel_size,
                           strides=strides,
                           padding=conv_pad,
                           use_bias=False,
                           depthwise_initializer=CONV_KERNEL_INITIALIZER,
                           name=name + 'dwconv')(x)
    x = KL.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
    x = KL.Activation(activation_fn, name=name + 'activation')(x)

    # Squeeze and Excitation phase
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = KL.GlobalAveragePooling2D(name=name + 'se_squeeze')(x)
        se = KL.Reshape((1, 1, filters), name=name + 'se_reshape')(se)
        se = KL.Conv2D(filters_se, 1,
                       padding='same',
                       activation=activation_fn,
                       kernel_initializer=CONV_KERNEL_INITIALIZER,
                       name=name + 'se_reduce')(se)
        se = KL.Conv2D(filters, 1,
                       padding='same',
                       activation='sigmoid',
                       kernel_initializer=CONV_KERNEL_INITIALIZER,
                       name=name + 'se_expand')(se)
        if K.backend() == 'theano':
            # For the Theano backend, we have to explicitly make
            # the excitation weights broadcastable.
            se = KL.Lambda(
                lambda x: K.pattern_broadcast(x, [True, True, True, False]),
                output_shape=lambda input_shape: input_shape,
                name=name + 'se_broadcast')(se)
        x = KL.multiply([x, se], name=name + 'se_excite')

    # Output phase
    x = KL.Conv2D(filters_out, 1,
                  padding='same',
                  use_bias=False,
                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                  name=name + 'project_conv')(x)
    x = KL.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)
    if id_skip is True and strides == 1 and filters_in == filters_out:
        if drop_rate > 0:
            x = KL.Dropout(drop_rate,
                           noise_shape=(None, 1, 1, 1),
                           name=name + 'drop')(x)
        x = KL.add([x, inputs], name=name + 'add')

    return x


def efficient_graph(width_coefficient,
                    depth_coefficient,
                    default_size,
                    dropout_rate=0.2,
                    drop_connect_rate=0.2,
                    depth_divisor=8,
                    activation_fn=swish,
                    blocks_args=DEFAULT_BLOCKS_ARGS,
                    model_name='efficientnet',
                    include_top=False,
                    weights='imagenet',
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    classes=1000,
                    **kwargs):
    """Instantiates the EfficientNet architecture using given scaling coefficients.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        width_coefficient: float, scaling coefficient for network width.
        depth_coefficient: float, scaling coefficient for network depth.
        default_size: integer, default input image size.
        dropout_rate: float, dropout rate before final classifier layer.
        drop_connect_rate: float, dropout rate at skip connections.
        depth_divisor: integer, a unit of network width.
        activation_fn: activation function.
        blocks_args: list of dicts, parameters to construct block modules.
        model_name: string, model name.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False.
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = KL.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = KL.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    # Build stem - C1
    x = img_input
    x = KL.ZeroPadding2D(padding=correct_pad(K, x, 3),
                         name='stem_conv_pad')(x)
    x = KL.Conv2D(round_filters(32), 3,
                  strides=2,
                  padding='valid',
                  use_bias=False,
                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                  name='stem_conv')(x)
    x = KL.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = KL.Activation(activation_fn, name='stem_activation')(x)

    # Build blocks
    from copy import deepcopy
    blocks_args = deepcopy(blocks_args)

    # C2-C8
    C = [x]
    b = 0
    blocks = float(sum(args['repeats'] for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']
            x = block(x, activation_fn, drop_connect_rate * b / blocks,
                      name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
            b += 1

        C.append(x)

    # Build top
    x = KL.Conv2D(round_filters(1280), 1,
                  padding='same',
                  use_bias=False,
                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                  name='top_conv')(x)
    x = KL.BatchNormalization(axis=bn_axis, name='top_bn')(x)
    x = KL.Activation(activation_fn, name='top_activation')(x)
    if include_top:
        x = KL.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate > 0:
            x = KL.Dropout(dropout_rate, name='top_dropout')(x)
        x = KL.Dense(classes,
                     activation='softmax',
                     kernel_initializer=DENSE_KERNEL_INITIALIZER,
                     name='probs')(x)
    else:
        if pooling == 'avg':
            x = KL.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = KL.GlobalMaxPooling2D(name='max_pool')(x)

    C.append(x)

    return C

    # # Ensure that the model takes into account
    # # any potential predecessors of `input_tensor`.
    # if input_tensor is not None:
    #     inputs = utils.get_source_inputs(input_tensor)
    # else:
    #     inputs = img_input

    # # Create model.
    # model = KM.Model(inputs, x, name=model_name)
    #
    # # Load weights.
    # if weights == 'imagenet':
    #     if include_top:
    #         file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment.h5'
    #         file_hash = WEIGHTS_HASHES[model_name[-2:]][0]
    #     else:
    #         file_suff = '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
    #         file_hash = WEIGHTS_HASHES[model_name[-2:]][1]
    #     file_name = model_name + file_suff
    #     weights_path = utils.get_file(file_name,
    #                                   BASE_WEIGHTS_PATH + file_name,
    #                                   cache_subdir='models',
    #                                   file_hash=file_hash)
    #     model.load_weights(weights_path)
    # elif weights is not None:
    #     model.load_weights(weights)
    #
    # return model


EFFICIENT_ARCHITECTURES = {
    'efficientnetb0': {
        'width_coefficient': 1.0,
        'depth_coefficient': 1.0,
        'default_size': 224,
        'dropout_rate': 0.2,
        'model_name': 'efficientnet-b0'
    },
    'efficientnetb1': {
        'width_coefficient': 1.0,
        'depth_coefficient': 1.1,
        'default_size': 256,    # 240
        'dropout_rate': 0.2,
        'model_name': 'efficientnet-b1'
    },
    'efficientnetb2': {
        'width_coefficient': 1.1,
        'depth_coefficient': 1.2,
        'default_size': 288,    # 260
        'dropout_rate': 0.3,
        'model_name': 'efficientnet-b2'
    },
    'efficientnetb3': {
        'width_coefficient': 1.2,
        'depth_coefficient': 1.4,
        'default_size': 320,    # 300
        'dropout_rate': 0.3,
        'model_name': 'efficientnet-b3'
    },
    'efficientnetb4': {
        'width_coefficient': 1.4,
        'depth_coefficient': 1.8,
        'default_size': 384,    # 380
        'dropout_rate': 0.4,
        'model_name': 'efficientnet-b4'
    },
    'efficientnetb5': {
        'width_coefficient': 1.6,
        'depth_coefficient': 2.2,
        'default_size': 416,    # 456
        'dropout_rate': 0.4,
        'model_name': 'efficientnet-b5'
    },
    'efficientnetb6': {
        'width_coefficient': 1.8,
        'depth_coefficient': 2.6,
        'default_size': 544,    # 528
        'dropout_rate': 0.5,
        'model_name': 'efficientnet-b6'
    },
    'efficientnetb7': {
        'width_coefficient': 2.0,
        'depth_coefficient': 3.1,
        'default_size': 608,    # 600
        'dropout_rate': 0.5,
        'model_name': 'efficientnet-b7'
    },
}
