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

"""Makes some classes and function gin configurable."""

import gin
import gin.tf.external_configurables
import tensorflow as tf
import tensorflow_addons.optimizers as tfa_optimizers

configurables = {
    'tf.keras.layers': (
        tf.keras.layers.Conv1D,
        tf.keras.layers.Conv1DTranspose,
        tf.keras.layers.Conv2D,
        tf.keras.layers.Conv2DTranspose,
        tf.keras.layers.Dense,
        tf.keras.layers.Flatten,
        tf.keras.layers.Reshape,
        tf.keras.layers.MaxPooling2D,
        tf.keras.layers.GlobalMaxPooling2D,
        tf.keras.layers.BatchNormalization,
        tf.keras.layers.LayerNormalization,
    ),
    'tf.keras.regularizers': (
        tf.keras.regularizers.L1,
        tf.keras.regularizers.L2,
        tf.keras.regularizers.L1L2,
    ),
    'tf.keras.initializers': (
        tf.keras.initializers.Constant,
    ),
    'tf.keras.losses': (
        tf.keras.losses.CategoricalCrossentropy,
    ),
    'tf.keras.optimizers.schedules': (
        tf.keras.optimizers.schedules.CosineDecay,
        tf.keras.optimizers.schedules.PiecewiseConstantDecay,
    ),
    'tfa.optimizers': (
        tfa_optimizers.MovingAverage,
    ),
}

for module in configurables:
  for v in configurables[module]:
    gin.config.external_configurable(v, module=module)
