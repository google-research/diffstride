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

"""Tests for reset."""

from absl.testing import parameterized
from diffstride import resnet
import tensorflow as tf


class ResnetTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for resnets."""

  def setUp(self):
    super().setUp()
    tf.random.set_seed(0)

  def test_residual_layer(self):
    num_filters = 7
    shape = (2, 32, 32, 3)
    proj_layer = resnet.ResidualLayer(
        filters=num_filters, kernel_size=3, strides=2, channels_first=False,
        pooling_cls=None, project=True)
    inputs = tf.random.uniform(shape)
    output = proj_layer(inputs)
    self.assertEqual(output.shape, (2, 16, 16, 7))

    # Should have the same number of features.
    id_layer = resnet.ResidualLayer(
        filters=num_filters, kernel_size=3, channels_first=False,
        pooling_cls=None, project=False)
    output2 = id_layer(output)
    self.assertEqual(output2.shape, output.shape)

  def test_resnet_block(self):
    num_filters = 7
    num_layers = 10
    block = resnet.ResnetBlock(
        filters=num_filters, kernel_size=3, strides=(2, 4),
        num_layers=num_layers, project_first=True, channels_first=False)
    self.assertLen(block.layers, num_layers)

    shape = (2, 64, 64, 3)
    inputs = tf.random.uniform(shape)
    output = block(inputs)
    self.assertEqual(output.shape, (2, 32, 16, num_filters))

    block = resnet.ResnetBlock(
        filters=7, strides=(2, 4), kernel_size=3, channels_first=False,
        num_layers=num_layers, project_first=False)
    self.assertLen(block.layers, num_layers)
    output2 = block(output)
    self.assertEqual(output2.shape, output.shape)


if __name__ == '__main__':
  tf.test.main()
