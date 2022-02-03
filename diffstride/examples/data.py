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
  mean = tf.constant([[[0.4914, 0.4822, 0.4465]]])
  std = tf.constant([[[0.2023, 0.1994, 0.2010]]])
  image = tf.cast(image, tf.float32) / tf.uint8.max
  return (image - mean) / std


@gin.configurable
def augment(image: tf.Tensor, pad_size: int = 2) -> tf.Tensor:
  """Data augmentation: Random shifts and flips."""
  shape = image.shape
  height, width = shape[:2]
  image = tf.image.pad_to_bounding_box(image, pad_size, pad_size,
                                       height + 2 * pad_size,
                                       width + 2 * pad_size)
  image = tf.image.random_crop(image, shape)
  image = tf.image.random_flip_left_right(image)
  return image


def sample_beta_distribution(size: int,
                             concentration_0: float = 0.2,
                             concentration_1: float = 0.2) -> tf.Tensor:
  """Samples from a beta distribution."""
  gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
  gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
  return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(images: tf.Tensor,
           labels: tf.Tensor,
           alpha: float = 0.2) -> Tuple[tf.Tensor, tf.Tensor]:
  """Applies mixup to two examples."""
  batch_size = images.shape[0]
  if batch_size != 2:
    raise ValueError(f'Mixup expects batch_size == 2 but got {batch_size}.')

  # Sample lambda and reshape it to do the mixup
  weight = sample_beta_distribution(1, alpha, alpha)[0]

  # Perform mixup on both images and labels by combining a pair of images/labels
  # (one from each dataset) into one image/label
  mixed_image = images[0] * weight + images[1] * (1 - weight)
  mixed_label = labels[0] * weight + labels[1] * (1 - weight)
  return (mixed_image, mixed_label)


def prepare(ds: tf.data.Dataset,
            num_classes: int,
            batch_size: int = 32,
            training: bool = True,
            augment_fn=augment,
            mixup_alpha: float = 0.0) -> tf.data.Dataset:
  """Prepares a dataset for train/test."""
  def transform(
      image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = convert(image)
    if training and augment_fn is not None:
      image = augment_fn(image)
    label = tf.one_hot(label, depth=num_classes)
    return image, label

  ds = ds.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
  if training:
    ds = ds.shuffle(1000)
    if mixup_alpha > 0.0:
      ds = ds.batch(2, drop_remainder=True)
      ds = ds.map(mix_up)
  ds = ds.batch(batch_size, drop_remainder=training)
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


@gin.configurable
def load_datasets(name: str,
                  batch_size: int = 32,
                  augment_fn=None,
                  mixup_alpha: float = 0.0):
  """Loads a tf.Dataset corresponding to the given name."""
  datasets, info = tfds.load(name, as_supervised=True, with_info=True)
  _, label_key = info.supervised_keys
  num_classes = info.features[label_key].num_classes
  ds_train = prepare(
      datasets['train'],
      num_classes=num_classes,
      batch_size=batch_size,
      training=True,
      augment_fn=augment_fn,
      mixup_alpha=mixup_alpha)
  ds_test = prepare(
      datasets['test'],
      num_classes=num_classes,
      batch_size=batch_size,
      training=False,
      augment_fn=None,
      mixup_alpha=0.0)
  return ds_train, ds_test, num_classes
