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

"""Tests for pooling."""

from absl.testing import parameterized
from diffstride import pooling
import tensorflow as tf


class PoolingTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    tf.random.set_seed(0)

  def test_spatial_pooling(self):
    shape = (1, 64, 64, 3)   # Because CPU.
    pool = pooling.SpatialPooling(strides=(2, 4), data_format='channels_last')
    inputs = tf.random.uniform(shape)
    output = pool(inputs)
    self.assertEqual(output.shape, (1, 32, 16, 3))

  @parameterized.parameters(['channels_first', 'channels_last'])
  def test_spectral_pooling(self, data_format):
    is_channels_last = data_format == 'channels_last'
    shape = (1, 64, 64, 3) if is_channels_last else (1, 3, 64, 64)
    pool = pooling.FixedSpectralPooling(strides=(2, 4), data_format=data_format)
    inputs = tf.random.uniform(shape)
    output = pool(inputs)
    output_shape = (1, 32, 16, 3) if is_channels_last else (1, 3, 32, 16)
    self.assertEqual(output.shape, output_shape)

  @parameterized.parameters(['channels_first', 'channels_last'])
  def test_learnable_spectral_pooling(self, data_format):
    is_channels_last = data_format == 'channels_last'
    shape = (1, 64, 64, 3) if is_channels_last else (1, 3, 64, 64)
    pool = pooling.LearnableSpectralPooling(
        strides=(2, 4), data_format=data_format)
    inputs = tf.random.uniform(shape)
    output = pool(inputs)
    output_shape = (1, 40, 24, 3) if is_channels_last else (1, 3, 40, 24)
    self.assertEqual(output.shape, output_shape)


if __name__ == '__main__':
  tf.test.main()
