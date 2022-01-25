# coding=utf-8
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines reset that are compatible with learnable stridding."""

import functools
from typing import Optional, Sequence, Tuple, Union

import gin
import tensorflow as tf

Number = Union[float, int]
Stride = Union[Number, Tuple[Number, Number]]


def data_format(channels_first: bool = True) -> str:
  return 'channels_first' if channels_first else 'channels_last'


def conv2d(
    *args, channels_first: bool = True, weight_decay: float = 0.0, **kwargs):
  return tf.keras.layers.Conv2D(
      *args,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
      data_format=data_format(channels_first),
      use_bias=False,
      **kwargs)


def batch_norm(channels_first: bool = True):
  axis = 1 if channels_first else 3
  return tf.keras.layers.BatchNormalization(
      momentum=0.9, epsilon=1e-5, axis=axis)


@gin.configurable
class ResidualLayer(tf.keras.layers.Layer):
  """A generic residual layer for Resnet.

  This resnet can represent an `IdBlock` or a `ProjBlock` by setting the
  `project` parameter and can be compatible with Spectral or Learnable poolings
  by setting the `pooling_cls` parameter.

  The pooling_cls and strides will be overwritten automatically in case of an
  ID block.
  """

  def __init__(self,
               filters: int,
               kernel_size: int = gin.REQUIRED,
               strides: Stride = (1, 1),
               pooling_cls=None,
               project: bool = False,
               channels_first: bool = True,
               weight_decay: float = 5e-3,
               **kwargs):
    super().__init__(**kwargs)

    # If we are in an Id Layer there is no striding of any kind.
    pooling_cls = None if not project else pooling_cls
    strides = (1, 1) if not project else strides
    # DiffStride compatibility: the strides go into the pooling layer.
    if pooling_cls is not None:
      conv_strides = (1, 1)
      self._pooling = pooling_cls(
          strides=strides, data_format=data_format(channels_first))
    else:
      self._pooling = tf.identity
      conv_strides = strides

    self._strided_conv = conv2d(
        filters, kernel_size, strides=conv_strides, padding='same',
        channels_first=channels_first, weight_decay=weight_decay)
    # The second convolution is a regular one with no strides, no matter what.
    self._unstrided_conv = conv2d(
        filters, kernel_size, strides=(1, 1), padding='same',
        channels_first=channels_first, weight_decay=weight_decay)
    self._bns = tuple(batch_norm(channels_first) for _ in range(2))

    self._shortcut_conv, self._shortcut_bn = None, None
    if project:
      self._shortcut_conv = conv2d(
          filters, kernel_size=1, strides=conv_strides, padding='valid',
          channels_first=channels_first, weight_decay=weight_decay)
      self._shortcut_bn = batch_norm(channels_first)

  def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
    x = self._strided_conv(inputs)
    x = self._pooling(x)
    x = self._bns[0](x, training=training)
    x = tf.nn.relu(x)
    x = self._unstrided_conv(x)
    x = self._bns[1](x, training=training)
    if self._shortcut_conv is not None:
      shortcut_x = self._shortcut_conv(inputs)
      shortcut_x = self._pooling(shortcut_x)
      shortcut_x = self._shortcut_bn(shortcut_x, training=training)
    else:
      shortcut_x = inputs
    return tf.nn.relu(x + shortcut_x)


@gin.configurable
class ResnetBlock(tf.keras.Sequential):
  """A block of residual layers sharing the same number of filters.

  The first residual layer of the block and only this one might be strided.
  This parameter is controlled by the `project_first` parameters.

  The kwargs are passed down to the ResidualLayer.
  """

  def __init__(self,
               filters: int = gin.REQUIRED,
               strides: Stride = gin.REQUIRED,
               num_layers: int = 2,
               project_first: bool = True,
               **kwargs):
    residual_fn = functools.partial(
        ResidualLayer, filters=filters, strides=strides, **kwargs)
    blocks = [residual_fn(project=True)] if project_first else []
    num_left_layers = num_layers - int(project_first)
    blocks.extend([residual_fn(project=False) for i in range(num_left_layers)])
    super().__init__(blocks)


@gin.configurable
class Resnet(tf.keras.Sequential):
  """A generic Resnet class.

  Depending on the number of blocks and the used filters, it can easily
  instantiate a Resnet18 or Resnet56.

  The kwargs are passed down to the ResnetBlock layer.
  """

  def __init__(self,
               filters: Sequence[int],
               strides: Sequence[Stride],
               num_output_classes: int = gin.REQUIRED,
               output_activation: Optional[str] = None,
               id_only: Sequence[int] = (),
               channels_first: bool = True,
               weight_decay: float = 5e-3,
               **kwargs):
    if len(filters) != len(strides):
      raise ValueError(f'The number of `filters` ({len(filters)}) should match'
                       f' the number of strides ({len(strides)})')
    layers = [tf.keras.layers.Permute((3, 1, 2))] if channels_first else []
    layers.extend([
        tf.keras.layers.ZeroPadding2D(
            padding=(1, 1), data_format=data_format(channels_first)),
        conv2d(filters[0], 3, strides=strides[0], padding='valid',
               channels_first=channels_first, weight_decay=weight_decay),
        batch_norm(channels_first),
        tf.keras.layers.ReLU(),
    ])
    for i, (num_filters, stride) in enumerate(zip(filters[1:], strides[1:])):
      layers.append(ResnetBlock(filters=num_filters,
                                strides=stride,
                                project_first=(i not in id_only),
                                channels_first=channels_first,
                                weight_decay=weight_decay,
                                **kwargs))
    layers.extend([
        tf.keras.layers.GlobalAveragePooling2D(
            data_format=data_format(channels_first)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            num_output_classes,
            activation=output_activation,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
            bias_regularizer=tf.keras.regularizers.L2(weight_decay),
            ),
        ])
    super().__init__(layers)


@gin.configurable
def resnet18(strides=None, **kwargs):
  strides = [1, 1, 2, 2, 2] if strides is None else strides
  filters = [64, 64, 128, 256, 512]
  return Resnet(
      filters, strides, id_only=[0], num_layers=2, kernel_size=3, **kwargs)


@gin.configurable
def resnet56(strides=None, **kwargs):
  filters = [16, 16, 32, 64]
  strides = [1, 1, 2, 2] if strides is None else strides
  return Resnet(filters, strides, num_layers=9, kernel_size=3, **kwargs)
