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

"""Pooling functions: spectral or spatial, with learnable stride or not.

Warning: This module runs faster in channels_first data format due to the use of
FFT. In case of channels_last, the tensors will be transposed to channels_first
and then transposed back, which increases the time and memory overhead
significantly. It is therefore highly recommended to run with channels_first on
a GPU.
"""

from typing import Optional, Tuple, Union

import gin
import tensorflow as tf

OptionalDim = Union[int, tf.Tensor, None]
Number = Union[float, int]
Stride = Union[Number, Tuple[Number, Number]]
CHANNELS_FIRST = 'channels_first'


def compute_adaptive_span_mask(threshold: tf.float32,
                               ramp_softness: tf.float32,
                               pos: tf.Tensor) -> tf.Tensor:
  """Adaptive mask as proposed in https://arxiv.org/pdf/1905.07799.pdf.

  Args:
    threshold: Threshold that starts the ramp.
    ramp_softness: Smoothness of the ramp.
    pos: Position indices.

  Returns:
   A tf.Tensor<tf.complex64> containing the
   thresholdings for the mask with the same size of pos.
  """
  output = (1.0 / ramp_softness) * (ramp_softness + threshold - pos)
  return tf.cast(tf.clip_by_value(output, 0.0, 1.0), dtype=tf.complex64)


def fixed_spectral_pooling(inputs: tf.Tensor,
                           lower_height: OptionalDim = None,
                           upper_height: OptionalDim = None,
                           upper_width: OptionalDim = None) -> Tuple[tf.Tensor]:
  """Fixed spectral pooling in 2D. Expects channels_first data format.

  Args:
   inputs: tf.Tensor<float>[batch_size, channels_in, height, width] of input
     sequences, obtained from the tf.signal.rfft2d.
   lower_height: Lower height limit to apply in the Fourier domain. This limit
     represents the upper bound for the lower corner.
   upper_height: Upper height limit to apply in the Fourier domain. This limit
     represents the lower bound for the upper corner.
   upper_width: Width limit to apply in the Fourier domain.

  Returns:
   A tf.Tensor<float>[batch_size, channels_out, height, width] containing the
     cropped coefficients of the Fourier transform.
  """
  return tf.concat([inputs[:, :, :lower_height, :upper_width],
                    inputs[:, :, upper_height:, :upper_width]],
                   axis=2)


@gin.configurable
class SpatialPooling(tf.keras.layers.AveragePooling2D):
  """Fixed pooling layer, computed in the spatial domain."""

  def __init__(self,
               pool_size: Union[int, Tuple[int, int]] = (1, 1),
               strides: Stride = (2, 2),
               **kwargs):
    super().__init__(
        pool_size=pool_size, strides=strides, padding='same', **kwargs)


@gin.configurable
class FixedSpectralPooling(tf.keras.layers.Layer):
  """Fixed Spectral pooling layer, computed in the Fourier domain."""

  def __init__(self,
               strides: Stride = (2.0, 2.0),
               data_format: str = CHANNELS_FIRST,
               **kwargs):
    """Fixed Spectral pooling layer.

    Args:
      strides: Fractional strides to apply via the Fourier domain.
      data_format: either 'channels_first' or 'channels_last'. Be aware that
        providing the data in channels_last format will significantly increase
        the overhead due to the need to transpose temporarily to channels_first.
      **kwargs: Additional arguments for parent class.
    """
    super().__init__(**kwargs)
    self._channels_first = data_format == CHANNELS_FIRST
    strides = (
        (strides, strides) if isinstance(strides, (int, float)) else strides)
    strides = tuple(map(float, strides))
    self._strides = strides
    if not strides[0] >= 1 and strides[1] >= 1:
      raise ValueError('Strides params need to be above 1, not ({}, {})'.format(
          str(strides[0]), str(strides[1])))

  def build(self, input_shape):
    if self._channels_first:
      height, width = input_shape[2], input_shape[3]
    else:
      height, width = input_shape[1], input_shape[2]
    self.strides = self.add_weight(
        shape=(2,),
        initializer=tf.initializers.Constant(self._strides),
        trainable=False,
        dtype=tf.float32,
        name='strides')
    strided_height = height // self.strides[0]
    strided_height -= strided_height % 2
    strided_width = width // self.strides[1]
    # The parameter 2 is the minimum to avoid collapse of the feature map.
    strided_height = tf.math.maximum(strided_height, 2)
    strided_width = tf.math.maximum(strided_width, 2)
    lower_height = strided_height // 2
    upper_height = height - lower_height
    upper_width = strided_width // 2 + 1
    self._output_shape = [int(strided_height), int(strided_width)]
    self._limits = [int(lower_height), int(upper_height), int(upper_width)]

  def call(self, inputs: tf.Tensor, training: bool = False):
    if not self._channels_first:
      inputs = tf.transpose(inputs, (0, 3, 1, 2))
    batch_size, input_chans = inputs.shape.as_list()[:2]
    lh, uh, uw = self._limits
    output_height, output_width = self._output_shape
    f_inputs = tf.signal.rfft2d(inputs)
    output = fixed_spectral_pooling(
        f_inputs, lower_height=lh, upper_height=uh, upper_width=uw)
    result = tf.ensure_shape(
        tf.signal.irfft2d(output, fft_length=[output_height, output_width]),
        [batch_size, input_chans, output_height, output_width])
    if not self._channels_first:
      result = tf.transpose(result, (0, 2, 3, 1))
    return result


class StrideConstraint(tf.keras.constraints.Constraint):
  """Constraint strides.

  Strides are constrained in [1,+infty) as default as smoothness factor
  always leave some feature map by default.
  """

  def __init__(self,
               lower_limit: Optional[float] = None,
               upper_limit: Optional[float] = None,
               **kwargs):
    """Constraint strides.

    Args:
      lower_limit: Lower limit for the stride.
      upper_limit: Upper limit for the stride.
      **kwargs: Additional arguments for parent class.
    """
    super().__init__(**kwargs)
    self._lower_limit = lower_limit if lower_limit is not None else 1.0
    self._upper_limit = (
        upper_limit if upper_limit is not None else tf.float32.max)

  def __call__(self, kernel):
    return tf.clip_by_value(kernel, self._lower_limit, self._upper_limit)


@gin.configurable
class DiffStride(tf.keras.layers.Layer):
  """Learnable Spectral pooling layer, computed in the Fourier domain.

  The adaptive window function is inspired from
  https://arxiv.org/pdf/1905.07799.pdf.
  """

  def __init__(self,
               strides: Stride = (2.0, 2.0),
               smoothness_factor: float = 4.0,
               cropping: bool = True,
               trainable: bool = True,
               shared_stride: bool = False,
               lower_limit_stride: Optional[float] = None,
               upper_limit_stride: Optional[float] = None,
               data_format: str = CHANNELS_FIRST,
               **kwargs):
    """Learnable Spectral pooling layer.

    Vertical and horizontal positions are the indices of the feature map. It
    allows to selectively weight the output of the fourier transform based
    on these positions.
    Args:
      strides: Fractional strides to init before learning the reduction in the
        Fourier domain.
      smoothness_factor: Smoothness factor to reduce/crop the input feature map
        in the Fourier domain.
      cropping: Boolean to specify if the layer crops or set to 0 the
        coefficients outside the cropping window in the Fourier domain.
      trainable: Boolean to specify if the stride is learnable.
      shared_stride: If `True`, a single parameter is shared for vertical and
        horizontal strides.
      lower_limit_stride: Lower limit for the stride. It can be useful when
        there are memory issues, it avoids the stride converge to small values.
      upper_limit_stride: Upper limit for the stride.
      data_format: either `channels_first` or `channels_last`. Be aware that
        channels_last will increase the memory cost due transformation to
        channels_first.
      **kwargs: Additional arguments for parent class.
    """
    super().__init__(**kwargs)
    self._cropping = cropping
    self._smoothness_factor = smoothness_factor
    self._shared_stride = shared_stride
    self.trainable = trainable
    self._lower_limit_stride = lower_limit_stride
    self._upper_limit_stride = upper_limit_stride
    self._channels_first = data_format == CHANNELS_FIRST

    # Ensures a tuple of floats.
    strides = (
        (strides, strides) if isinstance(strides, (int, float)) else strides)
    strides = tuple(map(float, strides))
    if strides[0] != strides[1] and shared_stride:
      raise ValueError('shared_stride requires the same initialization for '
                       f'vertical and horizontal strides but got {strides}')
    if strides[0] < 1 or strides[1] < 1:
      raise ValueError(f'Both strides should be >=1 but got {strides}')
    if smoothness_factor < 0.0:
      raise ValueError('Smoothness factor should be >= 0 but got '
                       f'{smoothness_factor}.')
    self._strides = strides

  def build(self, input_shape):
    del input_shape
    init = self._strides[0] if self._shared_stride else self._strides
    self.strides = self.add_weight(
        shape=(1,) if self._shared_stride else (2,),
        initializer=tf.initializers.Constant(init),
        trainable=self.trainable,
        dtype=tf.float32,
        name='strides',
        constraint=StrideConstraint(
            lower_limit=self._lower_limit_stride,
            upper_limit=self._upper_limit_stride))

  def call(self, inputs: tf.Tensor, training: bool = False):
    if not self._channels_first:
      inputs = tf.transpose(inputs, (0, 3, 1, 2))
    batch_size, channels = inputs.shape.as_list()[:2]
    height, width = tf.shape(inputs)[2], tf.shape(inputs)[3]

    horizontal_positions = tf.range(width // 2 + 1, dtype=tf.float32)
    vertical_positions = tf.range(
        height // 2 + height % 2, dtype=tf.float32)
    vertical_positions = tf.concat([
        tf.reverse(vertical_positions[(height % 2):], axis=[0]),
        vertical_positions], axis=0)
    # This clipping by .assign is performed to allow gradient to flow,
    # even when the stride becomes too small, i.e. close to 1.
    min_vertical_stride = tf.cast(height, tf.float32) / (
        tf.cast(height, tf.float32) - self._smoothness_factor)
    min_horizontal_stride = tf.cast(width, tf.float32) / (
        tf.cast(width, tf.float32) - self._smoothness_factor)
    if self._shared_stride:
      min_stride = tf.math.maximum(min_vertical_stride, min_horizontal_stride)
      self.strides[0].assign(tf.math.maximum(self.strides[0], min_stride))
      vertical_stride, horizontal_stride = self.strides[0], self.strides[0]
    else:
      self.strides[0].assign(
          tf.math.maximum(self.strides[0], min_vertical_stride))
      self.strides[1].assign(
          tf.math.maximum(self.strides[1], min_horizontal_stride))
      vertical_stride, horizontal_stride = self.strides[0], self.strides[1]

    # Explicitly calls the stride constraints on strides.
    vertical_stride = self.strides.constraint(vertical_stride)
    horizontal_stride = self.strides.constraint(horizontal_stride)

    strided_height = tf.cast(height, tf.float32) / vertical_stride
    strided_width = tf.cast(width, tf.float32) / horizontal_stride
    # Warning: Little discrepancy for the init of strided_height with
    #   FixedSpectralPooling. As the gradient of the operation below is 0, it
    #   is removed for DiffStride.
    # strided_height = strided_height - tf.math.floormod(strided_height, 2)
    # The parameter 2 is the minimum to avoid collapse of the feature map.
    strided_height = tf.math.maximum(strided_height, 2.0)
    strided_width = tf.math.maximum(strided_width, 2.0)
    lower_height = strided_height / 2.0
    upper_width = strided_width / 2.0 + 1.0

    f_inputs = tf.signal.rfft2d(inputs)
    horizontal_mask = compute_adaptive_span_mask(
        upper_width, self._smoothness_factor, horizontal_positions)
    vertical_mask = compute_adaptive_span_mask(
        lower_height, self._smoothness_factor, vertical_positions)

    vertical_mask = tf.signal.fftshift(vertical_mask)
    output = f_inputs * horizontal_mask[None, None, None, :]
    output = output * vertical_mask[None, None, :, None]
    if self._cropping:
      horizontal_to_keep = tf.stop_gradient(
          tf.where(tf.cast(horizontal_mask, tf.float32) > 0.)[:, 0])
      vertical_to_keep = tf.stop_gradient(
          tf.where(tf.cast(vertical_mask, tf.float32) > 0.)[:, 0])

      output = tf.gather(output, indices=vertical_to_keep, axis=2)
      output = tf.gather(output, indices=horizontal_to_keep, axis=3)

    result = tf.ensure_shape(
        tf.signal.irfft2d(output), [batch_size, channels, None, None])
    if not self._channels_first:
      result = tf.transpose(result, (0, 2, 3, 1))
    return result

  def compute_output_shape(self, input_shape):
    batch_size, channels = input_shape[:2]
    return (batch_size, channels, None, None)
