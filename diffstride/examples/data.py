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

"""Prepares data."""

from typing import Tuple


import gin
import tensorflow as tf
import tensorflow_datasets as tfds


def convert(image: tf.Tensor) -> tf.Tensor:
  return tf.cast(image, tf.float32) / tf.uint8.max


@gin.configurable
def augment(image: tf.Tensor,
            shift: float = 0.9,
            flip: bool = True) -> tf.Tensor:
  """Data augmentation: Random shifts and flips."""
  if flip:
    image = tf.image.random_flip_left_right(image)

  h, w = image.shape[:2]
  offset_h, offset_w = int(shift * h), int(shift * w)
  padded = tf.image.pad_to_bounding_box(
      image, offset_h, offset_w, h + offset_h, w + offset_w)
  return tf.image.random_crop(padded, image.shape)


def prepare(ds: tf.data.Dataset,
            batch_size: int = 32,
            training: bool = True,
            augment_fn=augment) -> tf.data.Dataset:
  """Prepares a dataset for train/test."""
  def transform(
      image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = convert(image)
    if training and augment_fn is not None:
      image = augment_fn(image)
    return image, label

  ds = ds.map(transform)
  if training:
    ds = ds.shuffle(1000)
  ds = ds.batch(batch_size, drop_remainder=training)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


@gin.configurable
def load_datasets(name: str, batch_size: int = 32, augment_fn=None):
  """Loads a tf.Dataset corresponding to the given name."""
  datasets, info = tfds.load(name, as_supervised=True, with_info=True)
  ds_train = prepare(
      datasets['train'], batch_size, training=True, augment_fn=augment_fn)
  ds_test = prepare(
      datasets['test'], batch_size, training=False, augment_fn=None)
  return ds_train, ds_test, info
